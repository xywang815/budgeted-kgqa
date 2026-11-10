#!/usr/bin/env python3
"""Prospectively frozen week-2 component attribution on MetaQA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kgqa_core import (
    answer_recall,
    bm25_scores,
    budget_aware_mmr,
    enumerate_simple_paths,
    evidence_tokens,
    load_jsonl,
    load_kg,
    pack_in_order,
    rank_by_scores,
    relation_bm25_scores,
    render_evidence,
    standalone_evidence_tokens,
)
from week2_selectors import (
    relation_only,
    shortest_first,
    unique_relation_chain,
    unit_endpoint,
)


MODEL_ID = "Qwen/Qwen3-8B-AWQ"
MODEL_REVISION = "cb7d6a337aadb4d2082ed0dcef1032e4f8645194"
NON_THINKING_ASSISTANT_PREFILL = "<think>\n\n</think>\n\n"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def mib(value: int) -> float:
    return round(value / (1024 * 1024), 3)


def running_gpu_compute_processes() -> list[str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_entity(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"^\s*(?:answer|answers)\s*:\s*", "", value, flags=re.I)
    value = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", value)
    value = value.strip().strip("`\"'[](){}")
    value = value.rstrip(".。")
    return " ".join(value.casefold().split())


def parse_entities(response: str) -> list[str]:
    cleaned = THINK_BLOCK_RE.sub("", response).strip()
    pieces = re.split(r"\s*(?:\||;|\n)\s*", cleaned)
    normalized: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        value = normalize_entity(piece)
        if value in {"none", "no answer", "unknown", "n/a"}:
            continue
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def answer_metrics(predictions: list[str], gold_answers: list[str]) -> dict[str, float]:
    predicted = set(predictions)
    gold = {normalize_entity(value) for value in gold_answers}
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hits_at_1": float(bool(predictions) and predictions[0] in gold),
        "exact_set_match": float(predicted == gold),
    }


def build_prompt(tokenizer, question: str, evidence: str) -> tuple[str, bool]:
    if evidence:
        task_instruction = (
            "Each evidence line is a candidate reasoning path. The entity at "
            "the far right end of that line is its candidate answer. Select "
            "only final path entities that answer the question; never return "
            "the topic entity, an intermediate entity, or a relation label. "
            "Return each selected entity exactly once, separated by the | "
            "character. If no candidate is valid, return NONE exactly once."
        )
    else:
        task_instruction = (
            "Answer from your internal knowledge. Return every answer entity "
            "exactly once, separated by the | character. If you cannot answer, "
            "return NONE exactly once."
        )
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the knowledge-graph question. "
                + task_instruction
                + " Return entities only, with no explanation or sentence. "
                "Preserve the spelling shown in the evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Evidence:\n{evidence if evidence else '(none)'}\n\n"
                f"Question: {question}\n/no_think"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    manual_prefill = not prompt.endswith(NON_THINKING_ASSISTANT_PREFILL)
    if manual_prefill:
        prompt += NON_THINKING_ASSISTANT_PREFILL
    if not prompt.endswith(NON_THINKING_ASSISTANT_PREFILL):
        raise RuntimeError("Failed to construct the non-thinking prompt")
    return prompt, manual_prefill


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    def mean(name: str) -> float:
        return round(statistics.fmean(float(row[name]) for row in rows), 6)

    return {
        "questions": len(rows),
        "macro_f1": mean("f1"),
        "mean_precision": mean("precision"),
        "mean_answer_recall": mean("recall"),
        "hits_at_1": mean("hits_at_1"),
        "exact_set_match": mean("exact_set_match"),
        "mean_evidence_answer_recall": mean("evidence_answer_recall"),
        "mean_evidence_tokens": mean("evidence_tokens"),
        "mean_total_input_tokens": mean("total_input_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "mean_generation_seconds": mean("generation_seconds"),
        "mean_selected_paths": mean("selected_paths"),
        "mean_selected_endpoint_precision": mean("selected_endpoint_precision"),
        "mean_selected_endpoint_recall": mean("selected_endpoint_recall"),
        "mean_selected_endpoint_f1": mean("selected_endpoint_f1"),
        "selected_endpoint_hits_at_1": mean("selected_endpoint_hits_at_1"),
        "mean_retrieval_seconds": mean("retrieval_seconds"),
        "candidate_constrained_macro_f1": mean("candidate_constrained_f1"),
        "candidate_constrained_hits_at_1": mean(
            "candidate_constrained_hits_at_1"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--sample-per-hop", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--max-paths", type=int, default=100_000)
    parser.add_argument("--standard-mmr-lambda", type=float, default=0.90)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.sample_per_hop < 1 or args.budget < 1 or args.max_paths < 1:
        raise ValueError("sample-per-hop, budget, and max-paths must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    if not 0.0 <= args.standard_mmr_lambda <= 1.0:
        raise ValueError("standard-mmr-lambda must be in [0, 1]")
    if args.split != "test" or args.seed != 20260818:
        raise ValueError("The frozen protocol requires split=test and seed=20260818")
    if args.budget != 256 or args.max_paths != 100_000:
        raise ValueError("The frozen protocol requires B256 and max_paths=100000")
    if args.sample_per_hop > 300:
        raise ValueError("The frozen manifest contains 300 questions per hop")

    reference_path = args.reference_results.resolve()
    manifest_path = args.sample_manifest.resolve()
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_configuration = reference_payload.get("configuration", {})
    reference_model = reference_payload.get("model", {})
    if reference_payload.get("status") != "PASS":
        raise ValueError("Reference result status is not PASS")
    expected_reference = {
        "split": "test",
        "sample_per_hop": 300,
        "seed": 20260818,
        "budget": 256,
        "max_paths": 100_000,
    }
    observed_reference = {
        key: reference_configuration.get(key) for key in expected_reference
    }
    if observed_reference != expected_reference:
        raise ValueError(
            f"Reference configuration mismatch: {observed_reference} "
            f"!= {expected_reference}"
        )
    if (
        reference_model.get("id") != MODEL_ID
        or reference_model.get("revision") != MODEL_REVISION
    ):
        raise ValueError("Reference model identity or revision mismatch")

    reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_payload.get("status") != "PASS":
        raise ValueError("Sample manifest status is not PASS")
    if manifest_payload.get("split") != "test":
        raise ValueError("Sample manifest split is not test")
    if manifest_payload.get("sampling_seed") != 20260818:
        raise ValueError("Sample manifest seed mismatch")
    if manifest_payload.get("gold_answers_included") is not False:
        raise ValueError("Sample manifest unexpectedly contains gold answers")
    if manifest_payload.get("source_result_sha256") != reference_sha256:
        raise ValueError("Sample manifest reference SHA256 mismatch")
    if manifest_payload.get("questions_per_hop") != {
        "1": 300,
        "2": 300,
        "3": 300,
    }:
        raise ValueError("Sample manifest hop counts mismatch")

    reference_results = reference_payload.get("results", [])
    reference_rows = [
        row for row in reference_results if row.get("condition") == "ra_mmr"
    ]
    if len(reference_results) != 300 * 3 * 6:
        raise ValueError("Reference does not contain six conditions per question")
    if len(reference_rows) != 300 * 3:
        raise ValueError("Reference RA-MMR question count mismatch")
    if len({str(row["id"]) for row in reference_rows}) != len(reference_rows):
        raise ValueError("Reference RA-MMR question identifiers are not unique")

    reference_by_id = {str(row["id"]): row for row in reference_rows}
    manifest_questions = manifest_payload.get("questions", [])
    if len(manifest_questions) != 900:
        raise ValueError("Sample manifest does not contain 900 questions")
    manifest_ids = [str(row["id"]) for row in manifest_questions]
    if len(set(manifest_ids)) != 900:
        raise ValueError("Sample manifest identifiers are not unique")
    if set(manifest_ids) != set(reference_by_id):
        raise ValueError("Sample manifest identifiers do not match the reference")

    selected_manifest_rows: list[dict[str, object]] = []
    for hop in (1, 2, 3):
        hop_rows = [row for row in manifest_questions if int(row["hop"]) == hop]
        if len(hop_rows) != 300:
            raise ValueError(f"Manifest {hop}-hop count is not 300")
        selected_manifest_rows.extend(hop_rows[: args.sample_per_hop])

    sampled_rows: list[dict[str, object]] = []
    for manifest_row in selected_manifest_rows:
        row = reference_by_id[str(manifest_row["id"])]
        if any(
            row.get(key) != manifest_row.get(key)
            for key in ("hop", "line_number", "question", "topic_entity")
        ):
            raise ValueError(f"Manifest/reference metadata mismatch for {row['id']}")
        sampled_rows.append(
            {
                "id": row["id"],
                "hop": int(row["hop"]),
                "line_number": int(row["line_number"]),
                "question": row["question"],
                "topic_entity": row["topic_entity"],
                "gold_answers": row["gold_answers"],
            }
        )
    hop_counts = {
        hop: sum(int(row["hop"]) == hop for row in sampled_rows)
        for hop in (1, 2, 3)
    }
    if any(count != args.sample_per_hop for count in hop_counts.values()):
        raise ValueError(f"Reference hop counts mismatch: {hop_counts}")

    if args.output.resolve().exists():
        raise FileExistsError(f"Refusing to overwrite {args.output.resolve()}")
    if not (args.data_dir.resolve() / "kg.tsv").is_file():
        raise FileNotFoundError("processed KG is missing")
    if not (args.model_dir.resolve() / "config.json").is_file():
        raise FileNotFoundError("model config is missing")
    if args.preflight_only:
        print("status=PASS")
        print("mode=preflight_only")
        print(f"reference={reference_path}")
        print(f"reference_sha256={reference_sha256}")
        print(f"sample_manifest={manifest_path}")
        print(f"sample_manifest_sha256={manifest_sha256}")
        print(f"questions={len(sampled_rows)} hop_counts={hop_counts}")
        print(
            "conditions="
            f"{('relation_only', 'standard_mmr_no_cost', 'shortest_first', 'unit_endpoint', 'unique_relation_chain')}"
        )
        print("equivalent_reuse=relation_plus_cost:ra_no_diversity")
        return 0

    existing_processes = running_gpu_compute_processes()
    if existing_processes:
        raise RuntimeError(
            "GPU safety check failed; existing compute processes: "
            + "; ".join(existing_processes)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    started = time.perf_counter()
    data_dir = args.data_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    print("gpu_safety_check=PASS", flush=True)
    print("loading_processed_kg...", flush=True)
    adjacency = load_kg(data_dir / "kg.tsv")
    print(f"kg_entities={len(adjacency)}", flush=True)
    print("loading_tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    print("loading_model...", flush=True)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
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
    print(f"model_load_seconds={model_load_seconds:.3f}", flush=True)

    # Exclude one-time CUDA/AWQ kernel initialization from per-question timing.
    warmup_prompt, _ = build_prompt(
        tokenizer,
        "What is the capital of France?",
        "- France --[capital]--> Paris",
    )
    warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        model.generate(
            **warmup_inputs,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize(0)
    del warmup_inputs
    print("generation_warmup=PASS", flush=True)

    payload: dict[str, object] = {
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_path": str(model_dir),
            "class": model.__class__.__name__,
            "dtype": str(model.dtype),
            "memory_footprint_mib": mib(int(model.get_memory_footprint())),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "autoawq": package_version("autoawq"),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "configuration": {
            "split": args.split,
            "sample_per_hop": args.sample_per_hop,
            "seed": args.seed,
            "budget": args.budget,
            "budget_definition": "tokenized evidence body only",
            "path_mode": "exact",
            "max_paths": args.max_paths,
            "component_ablation": {
                "status": "prospectively_frozen_internal_protocol_20260821",
                "relation_only": {
                    "ranking": "direction-aware relation BM25",
                    "cost_utility": False,
                    "diversity_utility": False,
                },
                "relation_plus_cost_reused": {
                    "source_condition": "ra_no_diversity",
                    "source_file": str(reference_path),
                    "equivalence": "normalized relation BM25 divided by token cost",
                },
                "standard_mmr_no_cost": {
                    "relevance": "BM25 over complete entity-relation paths",
                    "diversity": "entity-relation signature Jaccard",
                    "mmr_lambda": args.standard_mmr_lambda,
                    "cost_exponent": 0.00,
                },
                "shortest_first": {
                    "ranking": "ascending standalone path token cost",
                    "tie_break": "candidate enumeration order",
                },
                "unit_endpoint": {
                    "ranking": "direction-aware relation BM25",
                    "constraint": "one feasible path per endpoint before relevance fill",
                },
                "unique_relation_chain": {
                    "ranking": "direction-aware relation BM25",
                    "constraint": "one feasible directed relation chain before relevance fill",
                },
            },
            "reference_results": str(reference_path),
            "reference_results_sha256": reference_sha256,
            "sample_manifest": str(manifest_path),
            "sample_manifest_sha256": manifest_sha256,
            "thinking": False,
            "prompt_protocol": "endpoint_instruction_v3_frozen_retrieval",
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "checkpoint_every_questions": args.checkpoint_every,
            "repetition_penalty": 1.10,
            "no_repeat_ngram_size": 3,
            "raw_generation_metrics": "primary during prompt validation",
            "candidate_constrained_metrics": "secondary diagnostic; raw metrics retained",
            "sampling": "ordered subset of prospectively frozen identifier manifest",
            "analysis_status": "prospectively_frozen_internal_protocol_20260821",
        },
        "model_load_seconds": round(model_load_seconds, 6),
        "results": [],
    }
    atomic_json(output_path, payload)

    condition_order = (
        "relation_only",
        "standard_mmr_no_cost",
        "shortest_first",
        "unit_endpoint",
        "unique_relation_chain",
    )
    for question_index, row in enumerate(sampled_rows, start=1):
        hop = int(row["hop"])
        question = str(row["question"])
        gold_answers = [str(value) for value in row["gold_answers"]]
        enumeration_started = time.perf_counter()
        paths, truncated = enumerate_simple_paths(
            adjacency,
            str(row["topic_entity"]),
            hop,
            args.max_paths,
            exact_hops_only=True,
        )
        candidate_enumeration_seconds = time.perf_counter() - enumeration_started

        cost_started = time.perf_counter()
        standalone_costs = standalone_evidence_tokens(paths, tokenizer)
        evidence_cost_seconds = time.perf_counter() - cost_started

        bm25_started = time.perf_counter()
        scores = bm25_scores(question, paths)
        bm25_scoring_seconds = time.perf_counter() - bm25_started

        relation_started = time.perf_counter()
        relation_scores = relation_bm25_scores(question, paths)
        relation_scoring_seconds = time.perf_counter() - relation_started

        signature_started = time.perf_counter()
        signatures = [path.signature() for path in paths]
        signature_seconds = time.perf_counter() - signature_started

        selections = {}
        selection_seconds = {}

        selection_started = time.perf_counter()
        selections["relation_only"] = relation_only(
            paths,
            relation_scores,
            args.budget,
            tokenizer,
            standalone_costs,
        )
        selection_seconds["relation_only"] = (
            time.perf_counter() - selection_started
        )

        selection_started = time.perf_counter()
        selections["standard_mmr_no_cost"] = budget_aware_mmr(
            paths,
            scores,
            args.budget,
            tokenizer,
            relevance_weight=args.standard_mmr_lambda,
            cost_exponent=0.00,
            standalone_costs=standalone_costs,
            signatures=signatures,
        )
        selection_seconds["standard_mmr_no_cost"] = (
            time.perf_counter() - selection_started
        )

        selection_started = time.perf_counter()
        selections["shortest_first"] = shortest_first(
            paths,
            args.budget,
            tokenizer,
            standalone_costs,
        )
        selection_seconds["shortest_first"] = (
            time.perf_counter() - selection_started
        )

        selection_started = time.perf_counter()
        selections["unit_endpoint"] = unit_endpoint(
            paths,
            relation_scores,
            args.budget,
            tokenizer,
            standalone_costs,
        )
        selection_seconds["unit_endpoint"] = time.perf_counter() - selection_started

        selection_started = time.perf_counter()
        selections["unique_relation_chain"] = unique_relation_chain(
            paths,
            relation_scores,
            args.budget,
            tokenizer,
            standalone_costs,
        )
        selection_seconds["unique_relation_chain"] = (
            time.perf_counter() - selection_started
        )

        shared_base_seconds = candidate_enumeration_seconds + evidence_cost_seconds
        retrieval_seconds_by_condition = {
            "relation_only": (
                shared_base_seconds
                + relation_scoring_seconds
                + selection_seconds["relation_only"]
            ),
            "standard_mmr_no_cost": (
                shared_base_seconds
                + bm25_scoring_seconds
                + signature_seconds
                + selection_seconds["standard_mmr_no_cost"]
            ),
            "shortest_first": (
                shared_base_seconds
                + selection_seconds["shortest_first"]
            ),
            "unit_endpoint": (
                shared_base_seconds
                + relation_scoring_seconds
                + selection_seconds["unit_endpoint"]
            ),
            "unique_relation_chain": (
                shared_base_seconds
                + relation_scoring_seconds
                + selection_seconds["unique_relation_chain"]
            ),
        }

        for condition in condition_order:
            selected = selections[condition]
            evidence_token_count = evidence_tokens(selected, tokenizer)
            if evidence_token_count > args.budget:
                raise AssertionError(
                    f"Budget violation for {row['id']} {condition}: "
                    f"{evidence_token_count}>{args.budget}"
                )
            evidence = render_evidence(selected)
            prompt, manual_prefill = build_prompt(tokenizer, question, evidence)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            total_input_tokens = int(inputs["input_ids"].shape[-1])

            torch.cuda.reset_peak_memory_stats(0)
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
            response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            predictions = parse_entities(response)
            metrics = answer_metrics(predictions, gold_answers)
            allowed_endpoints = {
                normalize_entity(path.endpoint) for path in selected
            }
            constrained_predictions = (
                predictions
                if condition == "direct"
                else [value for value in predictions if value in allowed_endpoints]
            )
            constrained_metrics = answer_metrics(
                constrained_predictions, gold_answers
            )
            selected_endpoint_predictions = list(
                dict.fromkeys(normalize_entity(path.endpoint) for path in selected)
            )
            selected_endpoint_metrics = answer_metrics(
                selected_endpoint_predictions, gold_answers
            )

            result = {
                "id": row["id"],
                "hop": hop,
                "line_number": row["line_number"],
                "condition": condition,
                "budget": args.budget,
                "question": question,
                "topic_entity": row["topic_entity"],
                "gold_answers": gold_answers,
                "raw_response": response,
                "parsed_predictions": predictions,
                **metrics,
                "candidate_constrained_predictions": constrained_predictions,
                "candidate_constrained_precision": constrained_metrics["precision"],
                "candidate_constrained_recall": constrained_metrics["recall"],
                "candidate_constrained_f1": constrained_metrics["f1"],
                "candidate_constrained_hits_at_1": constrained_metrics["hits_at_1"],
                "candidate_constrained_exact_set_match": constrained_metrics[
                    "exact_set_match"
                ],
                "candidate_paths": len(paths),
                "candidate_truncated": truncated,
                "candidate_answer_recall": answer_recall(paths, gold_answers),
                "selected_paths": len(selected),
                "selected_endpoints": [path.endpoint for path in selected],
                "selected_endpoint_precision": selected_endpoint_metrics[
                    "precision"
                ],
                "selected_endpoint_recall": selected_endpoint_metrics["recall"],
                "selected_endpoint_f1": selected_endpoint_metrics["f1"],
                "selected_endpoint_hits_at_1": selected_endpoint_metrics[
                    "hits_at_1"
                ],
                "selected_evidence": [path.text() for path in selected],
                "evidence_tokens": evidence_token_count,
                "evidence_answer_recall": answer_recall(selected, gold_answers),
                "total_input_tokens": total_input_tokens,
                "output_tokens": output_tokens,
                "output_hit_max_new_tokens": output_tokens == args.max_new_tokens,
                "candidate_enumeration_seconds": round(
                    candidate_enumeration_seconds, 6
                ),
                "evidence_cost_seconds": round(evidence_cost_seconds, 6),
                "selector_only_seconds": round(selection_seconds[condition], 6),
                "retrieval_seconds": round(
                    retrieval_seconds_by_condition[condition], 6
                ),
                "generation_seconds": round(generation_seconds, 6),
                "generation_peak_mib": mib(torch.cuda.max_memory_allocated(0)),
                "manual_non_thinking_prefill": manual_prefill,
            }
            payload["results"].append(result)
            print(
                f"{question_index}/{len(sampled_rows)} {row['id']} "
                f"condition={condition} f1={metrics['f1']:.3f} "
                f"constrained_f1={constrained_metrics['f1']:.3f} "
                f"evidence_recall={result['evidence_answer_recall']:.3f} "
                f"input_tokens={total_input_tokens} gen_s={generation_seconds:.3f}",
                flush=True,
            )

        if (
            question_index % args.checkpoint_every == 0
            or question_index == len(sampled_rows)
        ):
            atomic_json(output_path, payload)
            print(
                f"checkpoint={question_index}/{len(sampled_rows)} "
                f"records={len(payload['results'])}",
                flush=True,
            )

    summary: dict[str, dict[str, float]] = {}
    results = payload["results"]
    for hop in (1, 2, 3):
        for condition in condition_order:
            rows = [
                row
                for row in results
                if int(row["hop"]) == hop and row["condition"] == condition
            ]
            summary[f"{hop}-hop/{condition}"] = summarize(rows)
    for condition in condition_order:
        rows = [row for row in results if row["condition"] == condition]
        summary[f"all-hops/{condition}"] = summarize(rows)
    payload["summary"] = summary
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    payload["status"] = "PASS"
    atomic_json(output_path, payload)

    print("status=PASS", flush=True)
    for name, values in summary.items():
        print(
            f"result={name} f1={values['macro_f1']} "
            f"constrained_f1={values['candidate_constrained_macro_f1']} "
            f"hits1={values['hits_at_1']} "
            f"evidence_recall={values['mean_evidence_answer_recall']} "
            f"endpoint_f1={values['mean_selected_endpoint_f1']} "
            f"input_tokens={values['mean_total_input_tokens']} "
            f"retrieval_s={values['mean_retrieval_seconds']} "
            f"gen_s={values['mean_generation_seconds']}",
            flush=True,
        )
    print(f"report={output_path}", flush=True)
    print(f"elapsed_seconds={payload['elapsed_seconds']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"status=FAIL error={error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
