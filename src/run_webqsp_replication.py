#!/usr/bin/env python3
"""Gold-free WebQSP replication of budgeted path selection with Qwen3-8B-AWQ."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kgqa_core import (
    Edge,
    GraphPath,
    bm25_scores,
    budget_aware_mmr,
    evidence_tokens,
    pack_in_order,
    rank_by_scores,
    relation_bm25_scores,
    standalone_evidence_tokens,
)
from run_kgqa_inference import (
    atomic_json,
    build_prompt,
    normalize_entity,
    parse_entities,
    running_gpu_compute_processes,
)


MODEL_ID = "Qwen/Qwen3-8B-AWQ"
MODEL_REVISION = "cb7d6a337aadb4d2082ed0dcef1032e4f8645194"
NON_THINKING_ASSISTANT_PREFILL = "<think>\n\n</think>\n\n"
CONDITIONS = ("direct", "bfs", "bm25", "ra_mmr", "oracle")
FB_RE = re.compile(r"^<fb:(.+)>$")
META_PREFIXES = (
    "type.object.",
    "common.topic.",
    "common.document.",
    "freebase.",
)


def normalize_fb(value: str) -> str:
    value = value.strip()
    match = FB_RE.match(value)
    if match:
        return match.group(1)
    if value.startswith("/"):
        return value[1:].replace("/", ".")
    return value


def relation_text(value: str) -> str:
    return value.replace(".", "_").replace("/", "_")


def display_value(identifier: str, names: dict[str, str]) -> str:
    name = names.get(identifier)
    if name and normalize_entity(name) != normalize_entity(identifier):
        return f"{name} [{identifier}]"
    return identifier


def load_question_graph(path: Path):
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    names: dict[str, str] = {}
    triple_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            fields = raw.rstrip("\n").split("\t")
            if len(fields) != 3:
                continue
            head, relation, tail = map(normalize_fb, fields)
            triple_count += 1
            if relation == "type.object.name":
                names[head] = tail
            elif not relation.startswith(META_PREFIXES):
                adjacency[head].append((relation, tail))
    return adjacency, names, triple_count


def enumerate_paths(
    adjacency: dict[str, list[tuple[str, str]]],
    names: dict[str, str],
    topics: list[str],
    max_paths: int,
):
    paths: list[GraphPath] = []
    endpoint_ids: dict[GraphPath, str] = {}
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    first_prefixes: list[tuple[str, str, str, GraphPath]] = []
    truncated = False

    for topic in topics:
        topic_display = display_value(topic, names)
        for relation, tail in adjacency.get(topic, ()):
            if tail == topic:
                continue
            tail_display = display_value(tail, names)
            edge = Edge(topic_display, relation_text(relation), tail_display)
            path = GraphPath((topic_display, tail_display), (edge,))
            key = ((topic, tail), (relation,))
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
            endpoint_ids[path] = tail
            first_prefixes.append((topic, tail, relation, path))
            if len(paths) >= max_paths:
                return paths, endpoint_ids, True

    for topic, middle, first_relation, prefix in first_prefixes:
        for second_relation, tail in adjacency.get(middle, ()):
            if tail in {topic, middle}:
                continue
            tail_display = display_value(tail, names)
            edge = Edge(
                prefix.nodes[-1], relation_text(second_relation), tail_display
            )
            path = GraphPath(prefix.nodes + (tail_display,), prefix.edges + (edge,))
            key = ((topic, middle, tail), (first_relation, second_relation))
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
            endpoint_ids[path] = tail
            if len(paths) >= max_paths:
                truncated = True
                return paths, endpoint_ids, truncated
    return paths, endpoint_ids, truncated


def parse_question(question: dict) -> dict[str, object] | None:
    parses = []
    topics = []
    gold_ids: set[str] = set()
    gold_aliases: dict[str, str] = {}
    chain_lengths = []
    for parse in question.get("Parses", []):
        topic = normalize_fb(str(parse.get("TopicEntityMid") or ""))
        chain = parse.get("InferentialChain") or []
        answers = parse.get("Answers") or []
        if not topic or not chain or not answers:
            continue
        topics.append(topic)
        chain_lengths.append(len(chain))
        parse_gold = []
        for answer in answers:
            identifier = normalize_fb(str(answer.get("AnswerArgument") or ""))
            if not identifier:
                continue
            gold_ids.add(identifier)
            parse_gold.append(identifier)
            gold_aliases[normalize_entity(identifier)] = identifier
            entity_name = str(answer.get("EntityName") or "").strip()
            if entity_name:
                gold_aliases[normalize_entity(entity_name)] = identifier
        parses.append({"topic": topic, "chain_length": len(chain), "answers": parse_gold})
    if not parses:
        return None
    return {
        "id": str(question["QuestionId"]),
        "question": str(question["RawQuestion"]),
        "topics": list(dict.fromkeys(topics)),
        "gold_ids": sorted(gold_ids),
        "gold_aliases": gold_aliases,
        "annotated_max_hops": max(chain_lengths),
        "parse_count": len(parses),
    }


def id_metrics(predicted: list[str], gold: list[str]) -> dict[str, float]:
    predicted_set = set(predicted)
    gold_set = set(gold)
    overlap = len(predicted_set & gold_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hits_at_1": float(bool(predicted) and predicted[0] in gold_set),
        "exact_set_match": float(predicted_set == gold_set),
    }


def aliases_for_paths(
    selected: list[GraphPath], endpoint_ids: dict[GraphPath, str], names: dict[str, str]
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for path in selected:
        identifier = endpoint_ids[path]
        aliases[normalize_entity(path.endpoint)] = identifier
        aliases[normalize_entity(identifier)] = identifier
        if identifier in names:
            aliases[normalize_entity(names[identifier])] = identifier
    return aliases


def map_predictions(predictions: list[str], aliases: dict[str, str]) -> list[str]:
    mapped = []
    seen = set()
    for prediction in predictions:
        identifier = aliases.get(normalize_entity(prediction))
        if identifier and identifier not in seen:
            seen.add(identifier)
            mapped.append(identifier)
    return mapped


def render_labeled_evidence(
    selected: list[GraphPath], endpoint_ids: dict[GraphPath, str]
) -> tuple[str, dict[str, str]]:
    """Render paths with short answer codes so Freebase IDs need not be copied."""
    lines = []
    labels: dict[str, str] = {}
    for index, path in enumerate(selected, start=1):
        label = f"A{index}"
        lines.append(f"- {path.text()} => {label}")
        labels[label.lower()] = endpoint_ids[path]
    return "\n".join(lines), labels


def fit_labeled_budget(
    selected: list[GraphPath],
    endpoint_ids: dict[GraphPath, str],
    tokenizer,
    budget: int,
) -> tuple[list[GraphPath], str, dict[str, str], int]:
    """Trim only the tail until the exact labeled evidence fits the token cap."""
    fitted = list(selected)
    while True:
        evidence, labels = render_labeled_evidence(fitted, endpoint_ids)
        tokens = len(tokenizer.encode(evidence, add_special_tokens=False)) if evidence else 0
        if tokens <= budget or not fitted:
            return fitted, evidence, labels, tokens
        fitted.pop()


def build_labeled_prompt(tokenizer, question: str, evidence: str) -> str:
    system = (
        "Answer the knowledge-graph question using only the candidate paths. "
        "Each path ends with a short answer code such as A1. Return the code or codes "
        "whose rightmost entities answer the question. Separate multiple codes with |. "
        "Return each code at most once. If no candidate is valid, return NONE. "
        "Return codes only, with no explanation."
    )
    user = f"Evidence:\n{evidence}\n\nQuestion: {question}\n/no_think"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if not prompt.endswith(NON_THINKING_ASSISTANT_PREFILL):
        prompt += NON_THINKING_ASSISTANT_PREFILL
    if not prompt.endswith(NON_THINKING_ASSISTANT_PREFILL):
        raise RuntimeError("Failed to construct the non-thinking labeled prompt")
    return prompt


def map_answer_codes(response: str, labels: dict[str, str]) -> list[str]:
    mapped = []
    seen = set()
    for code in re.findall(r"(?i)\bA\d+\b", response):
        identifier = labels.get(code.lower())
        if identifier and identifier not in seen:
            seen.add(identifier)
            mapped.append(identifier)
    return mapped


def mean_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    def mean(key: str) -> float:
        return round(statistics.fmean(float(row[key]) for row in rows), 6)

    return {
        "questions": len(rows),
        "macro_f1": mean("f1"),
        "hits_at_1": mean("hits_at_1"),
        "exact_set_match": mean("exact_set_match"),
        "candidate_answer_recall": mean("candidate_answer_recall"),
        "evidence_answer_recall": mean("evidence_answer_recall"),
        "mean_evidence_tokens": mean("evidence_tokens"),
        "mean_total_input_tokens": mean("total_input_tokens"),
        "mean_selected_paths": mean("selected_paths"),
        "mean_retrieval_seconds": mean("retrieval_seconds"),
        "mean_generation_seconds": mean("generation_seconds"),
        "output_cap_rate": mean("output_hit_max_new_tokens"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-test", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--subgraph-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--max-paths", type=int, default=100_000)
    parser.add_argument("--rerank-cap", type=int, default=20_000)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.sample_size < 1 or args.budget < 1 or args.max_paths < 1:
        raise ValueError("sample-size, budget, and max-paths must be positive")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or not all(audit.get("gate", {}).values()):
        raise ValueError("WebQSP resource gate did not pass")
    official = json.loads(args.official_test.read_text(encoding="utf-8"))
    audit_rows = {row["question_id"]: row for row in audit["records"]}
    population = []
    for question in official.get("Questions", []):
        parsed = parse_question(question)
        if not parsed:
            continue
        audited = audit_rows.get(parsed["id"], {})
        if audited.get("subgraph_exists") and audited.get("topic_present"):
            population.append(parsed)
    generator = random.Random(args.seed)
    sampled = generator.sample(population, min(args.sample_size, len(population)))
    sampled.sort(key=lambda row: int(str(row["id"]).rsplit("-", 1)[-1]))

    if not args.retrieval_only:
        processes = running_gpu_compute_processes()
        if processes:
            raise RuntimeError("GPU safety check failed: " + "; ".join(processes))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir.resolve(), local_files_only=True, trust_remote_code=False
    )
    model = None
    model_load_seconds = 0.0
    if not args.retrieval_only:
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir.resolve(),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.eval()
        torch.cuda.synchronize(0)
        model_load_seconds = time.perf_counter() - load_started
        warmup, _ = build_prompt(tokenizer, "What is the capital of France?", "- France --[capital]--> Paris")
        warmup_inputs = tokenizer(warmup, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            model.generate(**warmup_inputs, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize(0)
        print("gpu_safety_check=PASS generation_warmup=PASS", flush=True)

    payload: dict[str, object] = {
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_path": str(args.model_dir.resolve()),
            "class": model.__class__.__name__ if model else None,
            "dtype": str(model.dtype) if model else None,
        },
        "configuration": {
            "dataset": "WebQSP official v1.0",
            "freebase_version": official.get("FreebaseVersion"),
            "population": len(population),
            "sample_size": len(sampled),
            "seed": args.seed,
            "budget": args.budget,
            "budget_definition": "tokenized labeled evidence body only; answer codes included",
            "max_paths": args.max_paths,
            "rerank_cap": args.rerank_cap,
            "path_depths": [1, 2],
            "path_direction": "outgoing",
            "conditions": CONDITIONS,
            "ra_mmr_lambda": 0.90,
            "ra_cost_exponent": 1.00,
            "gold_policy": "gold answers used only for metrics and oracle upper bound",
            "retrieval_only": args.retrieval_only,
            "sample_ids": [row["id"] for row in sampled],
        },
        "model_load_seconds": round(model_load_seconds, 6),
        "results": [],
    }
    atomic_json(args.output.resolve(), payload)

    for index, row in enumerate(sampled, start=1):
        retrieval_started = time.perf_counter()
        graph_path = args.subgraph_dir / f"{row['id']}.nxhd"
        adjacency, names, triple_count = load_question_graph(graph_path)
        paths, endpoint_ids, truncated = enumerate_paths(
            adjacency, names, list(row["topics"]), args.max_paths
        )
        candidate_recall = (
            len(set(endpoint_ids.values()) & set(row["gold_ids"])) / len(row["gold_ids"])
            if row["gold_ids"]
            else 0.0
        )

        bfs_pool = paths[: min(len(paths), args.rerank_cap)]
        bm25_all_scores = bm25_scores(str(row["question"]), paths)
        bm25_ranked = rank_by_scores(paths, bm25_all_scores)[: args.rerank_cap]
        relation_all_scores = relation_bm25_scores(str(row["question"]), paths)
        relation_ranked = rank_by_scores(paths, relation_all_scores)[: args.rerank_cap]
        relation_score_by_path = dict(zip(paths, relation_all_scores))

        cost_paths = list(dict.fromkeys(bfs_pool + bm25_ranked + relation_ranked))
        costs = standalone_evidence_tokens(cost_paths, tokenizer)
        cost_by_path = dict(zip(cost_paths, costs))
        bfs_costs = [cost_by_path[path] for path in bfs_pool]
        bm25_costs = [cost_by_path[path] for path in bm25_ranked]
        relation_costs = [cost_by_path[path] for path in relation_ranked]
        relation_scores = [relation_score_by_path[path] for path in relation_ranked]
        signatures = [path.directional_signature() for path in relation_ranked]

        oracle_by_id: dict[str, GraphPath] = {}
        for path in paths:
            identifier = endpoint_ids[path]
            if identifier not in set(row["gold_ids"]):
                continue
            previous = oracle_by_id.get(identifier)
            if previous is None or len(path.text()) < len(previous.text()):
                oracle_by_id[identifier] = path
        oracle_paths = sorted(oracle_by_id.values(), key=lambda path: (len(path.text()), path.text()))
        oracle_costs = standalone_evidence_tokens(oracle_paths, tokenizer)

        selections = {
            "direct": [],
            "bfs": pack_in_order(bfs_pool, args.budget, tokenizer, bfs_costs),
            "bm25": pack_in_order(bm25_ranked, args.budget, tokenizer, bm25_costs),
            "ra_mmr": budget_aware_mmr(
                relation_ranked,
                relation_scores,
                args.budget,
                tokenizer,
                relevance_weight=0.90,
                cost_exponent=1.00,
                standalone_costs=relation_costs,
                signatures=signatures,
            ),
            "oracle": pack_in_order(oracle_paths, args.budget, tokenizer, oracle_costs),
        }
        retrieval_seconds = time.perf_counter() - retrieval_started

        for condition in CONDITIONS:
            selected, labeled_evidence, answer_labels, tokens = fit_labeled_budget(
                selections[condition], endpoint_ids, tokenizer, args.budget
            )
            if tokens > args.budget:
                raise AssertionError(f"budget violation {row['id']} {condition}")
            endpoint_list = list(dict.fromkeys(endpoint_ids[path] for path in selected))
            evidence_recall = (
                len(set(endpoint_list) & set(row["gold_ids"])) / len(row["gold_ids"])
                if row["gold_ids"]
                else 0.0
            )
            response = ""
            predictions: list[str] = []
            mapped: list[str] = []
            total_input_tokens = 0
            output_tokens = 0
            generation_seconds = 0.0
            hit_cap = False
            if model is not None:
                if condition == "direct":
                    prompt, _ = build_prompt(tokenizer, str(row["question"]), "")
                else:
                    prompt = build_labeled_prompt(
                        tokenizer, str(row["question"]), labeled_evidence
                    )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                total_input_tokens = int(inputs["input_ids"].shape[-1])
                torch.cuda.synchronize(0)
                generation_started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.10,
                        no_repeat_ngram_size=3,
                    )
                torch.cuda.synchronize(0)
                generation_seconds = time.perf_counter() - generation_started
                output_ids = generated[0, total_input_tokens:]
                output_tokens = int(output_ids.shape[-1])
                hit_cap = output_tokens == args.max_new_tokens
                response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                if condition == "direct":
                    predictions = parse_entities(response)
                    mapped = map_predictions(predictions, dict(row["gold_aliases"]))
                else:
                    predictions = re.findall(r"(?i)\bA\d+\b", response)
                    mapped = map_answer_codes(response, answer_labels)
            metrics = id_metrics(mapped, list(row["gold_ids"]))
            result = {
                "id": row["id"],
                "annotated_max_hops": row["annotated_max_hops"],
                "condition": condition,
                "question": row["question"],
                "topics": row["topics"],
                "gold_ids": row["gold_ids"],
                "raw_response": response,
                "parsed_predictions": predictions,
                "mapped_prediction_ids": mapped,
                **metrics,
                "subgraph_triples": triple_count,
                "candidate_paths": len(paths),
                "candidate_truncated": truncated,
                "candidate_answer_recall": candidate_recall,
                "selected_paths": len(selected),
                "selected_endpoint_ids": endpoint_list,
                "selected_evidence": [path.text() for path in selected],
                "rendered_labeled_evidence": labeled_evidence,
                "answer_code_map": answer_labels,
                "evidence_tokens": tokens,
                "evidence_answer_recall": evidence_recall,
                "total_input_tokens": total_input_tokens,
                "output_tokens": output_tokens,
                "output_hit_max_new_tokens": hit_cap,
                "retrieval_seconds": round(retrieval_seconds, 6),
                "generation_seconds": round(generation_seconds, 6),
            }
            payload["results"].append(result)
            print(
                f"{index}/{len(sampled)} {row['id']} condition={condition} "
                f"f1={metrics['f1']:.3f} evidence_recall={evidence_recall:.3f} "
                f"paths={len(paths)} truncated={truncated} gen_s={generation_seconds:.3f}",
                flush=True,
            )
        if index % args.checkpoint_every == 0 or index == len(sampled):
            atomic_json(args.output.resolve(), payload)
            print(f"checkpoint={index}/{len(sampled)} records={len(payload['results'])}", flush=True)

    payload["summary"] = {
        condition: mean_summary([row for row in payload["results"] if row["condition"] == condition])
        for condition in CONDITIONS
    }
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    payload["status"] = "PASS"
    atomic_json(args.output.resolve(), payload)
    print("status=PASS")
    for condition, values in payload["summary"].items():
        print(
            f"result={condition} f1={values['macro_f1']} hits1={values['hits_at_1']} "
            f"candidate_recall={values['candidate_answer_recall']} evidence_recall={values['evidence_answer_recall']} "
            f"input_tokens={values['mean_total_input_tokens']} retrieval_s={values['mean_retrieval_seconds']} "
            f"gen_s={values['mean_generation_seconds']}",
            flush=True,
        )
    print(f"report={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"status=FAIL error={error}", file=sys.stderr, flush=True)
        raise
