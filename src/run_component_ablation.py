#!/usr/bin/env python3
"""Post-hoc Qwen3 component ablation on frozen MetaQA questions."""

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
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--sample-per-hop", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--max-paths", type=int, default=100_000)
    parser.add_argument("--ba-mmr-lambda", type=float, default=0.90)
    parser.add_argument("--ba-cost-exponent", type=float, default=0.50)
    parser.add_argument("--ra-mmr-lambda", type=float, default=0.90)
    parser.add_argument("--ra-cost-exponent", type=float, default=1.00)
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
    if not 0.0 <= args.ba_mmr_lambda <= 1.0:
        raise ValueError("ba-mmr-lambda must be in [0, 1]")
    if not 0.0 <= args.ra_mmr_lambda <= 1.0:
        raise ValueError("ra-mmr-lambda must be in [0, 1]")
    if args.ba_cost_exponent < 0.0 or args.ra_cost_exponent < 0.0:
        raise ValueError("cost exponents must be non-negative")
    reference_path = args.reference_results.resolve()
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_configuration = reference_payload.get("configuration", {})
    reference_model = reference_payload.get("model", {})
    if reference_payload.get("status") != "PASS":
        raise ValueError("Reference result status is not PASS")
    expected_reference = {
        "split": args.split,
        "sample_per_hop": args.sample_per_hop,
        "seed": args.seed,
        "budget": args.budget,
        "max_paths": args.max_paths,
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
    reference_results = reference_payload.get("results", [])
    reference_rows = [
        row for row in reference_results if row.get("condition") == "ra_mmr"
    ]
    if len(reference_results) != args.sample_per_hop * 3 * 6:
        raise ValueError("Reference does not contain six conditions per question")
    if len(reference_rows) != args.sample_per_hop * 3:
        raise ValueError("Reference RA-MMR question count mismatch")
    if len({str(row["id"]) for row in reference_rows}) != len(reference_rows):
        raise ValueError("Reference RA-MMR question identifiers are not unique")

    sampled_rows: list[dict[str, object]] = []
    for row in reference_rows:
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
    sampled_rows.sort(key=lambda row: (int(row["hop"]), int(row["line_number"])))
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
        print(f"reference_sha256={hashlib.sha256(reference_path.read_bytes()).hexdigest()}")
        print(f"questions={len(sampled_rows)} hop_counts={hop_counts}")
        print(
            "conditions="
            + str(
                (
                    "racps_no_relation",
                    "racps_no_novelty",
                    "racps_no_cost",
                    "racps_raw_selector_labels",
                    "racps_raw_end_to_end",
                )
            )
        )
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
                "status": "post_hoc_exploratory",
                "frozen_full_condition": {
                    "name": "ra_mmr",
                    "mmr_lambda": args.ra_mmr_lambda,
                    "cost_exponent": args.ra_cost_exponent,
                },
                "racps_no_relation": {
                    "relevance": "BM25 over complete entity-relation paths",
                    "diversity": "unchanged direction-aware relation signature Jaccard",
                    "mmr_lambda": 0.90,
                    "cost_exponent": 1.00,
                    "isolated_component": "relation-only relevance representation",
                },
                "racps_no_novelty": {
                    "relevance": "BM25 over direction-aware relation verbalizations",
                    "novelty": "coefficient fixed to zero",
                    "mmr_lambda": 1.00,
                    "cost_exponent": 1.00,
                    "isolated_component": "novelty term",
                },
                "racps_no_cost": {
                    "relevance": "BM25 over direction-aware relation verbalizations",
                    "diversity": "direction-aware relation signature Jaccard",
                    "mmr_lambda": 0.90,
                    "cost_exponent": 0.00,
                    "isolated_component": "length-cost normalization",
                },
                "racps_raw_selector_labels": {
                    "relevance": "BM25 over direction-tagged raw KG relation identifiers",
                    "diversity": "raw direction-relation signature Jaccard",
                    "evidence_rendering": "manual relation verbalizations retained",
                    "mmr_lambda": 0.90,
                    "cost_exponent": 1.00,
                },
                "racps_raw_end_to_end": {
                    "relevance": "BM25 over direction-tagged raw KG relation identifiers",
                    "diversity": "raw direction-relation signature Jaccard",
                    "evidence_rendering": "raw KG relation identifiers",
                    "mmr_lambda": 0.90,
                    "cost_exponent": 1.00,
                },
            },
            "reference_results": str(reference_path),
            "reference_results_sha256": hashlib.sha256(
                reference_path.read_bytes()
            ).hexdigest(),
            "thinking": False,
            "prompt_protocol": "endpoint_instruction_v3_frozen_retrieval",
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "checkpoint_every_questions": args.checkpoint_every,
            "repetition_penalty": 1.10,
            "no_repeat_ngram_size": 3,
            "raw_generation_metrics": "primary during prompt validation",
            "candidate_constrained_metrics": "secondary diagnostic; raw metrics retained",
            "sampling": "exact question identifiers loaded from frozen reference",
            "analysis_status": "post_hoc_exploratory",
        },
        "model_load_seconds": round(model_load_seconds, 6),
        "results": [],
    }
    atomic_json(output_path, payload)

    condition_order = (
        "racps_no_relation",
        "racps_no_novelty",
        "racps_no_cost",
        "racps_raw_selector_labels",
        "racps_raw_end_to_end",
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
        raw_standalone_costs = standalone_evidence_tokens(
            paths, tokenizer, raw_labels=True
        )
        evidence_cost_seconds = time.perf_counter() - cost_started

        bm25_started = time.perf_counter()
        scores = bm25_scores(question, paths)
        bm25_scoring_seconds = time.perf_counter() - bm25_started

        relation_started = time.perf_counter()
        relation_scores = relation_bm25_scores(question, paths)
        relation_scoring_seconds = time.perf_counter() - relation_started

        raw_relation_started = time.perf_counter()
        raw_relation_scores = relation_bm25_scores(
            question, paths, raw_labels=True
        )
        raw_relation_scoring_seconds = time.perf_counter() - raw_relation_started

        signature_started = time.perf_counter()
        directional_signatures = [
            path.directional_signature() for path in paths
        ]
        raw_directional_signatures = [
            path.raw_directional_signature() for path in paths
        ]
        signature_seconds = time.perf_counter() - signature_started

        selections = {}
        selection_seconds = {}

        selection_started = time.perf_counter()
        selections["racps_no_relation"] = budget_aware_mmr(
            paths,
            scores,
            args.budget,
            tokenizer,
            relevance_weight=0.90,
            cost_exponent=1.00,
            standalone_costs=standalone_costs,
            signatures=directional_signatures,
        )
        selection_seconds["racps_no_relation"] = (
            time.perf_counter() - selection_started
        )

        selection_started = time.perf_counter()
        selections["racps_no_novelty"] = budget_aware_mmr(
            paths,
            relation_scores,
            args.budget,
            tokenizer,
            relevance_weight=1.00,
            cost_exponent=1.00,
            standalone_costs=standalone_costs,
            signatures=directional_signatures,
        )
        selection_seconds["racps_no_novelty"] = (
            time.perf_counter() - selection_started
        )

        selection_started = time.perf_counter()
        selections["racps_no_cost"] = budget_aware_mmr(
            paths,
            relation_scores,
            args.budget,
            tokenizer,
            relevance_weight=0.90,
            cost_exponent=0.00,
            standalone_costs=standalone_costs,
            signatures=directional_signatures,
        )
        selection_seconds["racps_no_cost"] = (
            time.perf_counter() - selection_started
        )

        selection_started = time.perf_counter()
        selections["racps_raw_selector_labels"] = budget_aware_mmr(
            paths,
            raw_relation_scores,
            args.budget,
            tokenizer,
            relevance_weight=0.90,
            cost_exponent=1.00,
            standalone_costs=standalone_costs,
            signatures=raw_directional_signatures,
        )
        selection_seconds["racps_raw_selector_labels"] = (
            time.perf_counter() - selection_started
        )

        selection_started = time.perf_counter()
        selections["racps_raw_end_to_end"] = budget_aware_mmr(
            paths,
            raw_relation_scores,
            args.budget,
            tokenizer,
            relevance_weight=0.90,
            cost_exponent=1.00,
            standalone_costs=raw_standalone_costs,
            signatures=raw_directional_signatures,
            raw_labels=True,
        )
        selection_seconds["racps_raw_end_to_end"] = (
            time.perf_counter() - selection_started
        )

        shared_base_seconds = candidate_enumeration_seconds + evidence_cost_seconds
        retrieval_seconds_by_condition = {
            "racps_no_relation": (
                shared_base_seconds
                + bm25_scoring_seconds
                + signature_seconds
                + selection_seconds["racps_no_relation"]
            ),
            "racps_no_novelty": (
                shared_base_seconds
                + relation_scoring_seconds
                + signature_seconds
                + selection_seconds["racps_no_novelty"]
            ),
            "racps_no_cost": (
                shared_base_seconds
                + relation_scoring_seconds
                + signature_seconds
                + selection_seconds["racps_no_cost"]
            ),
            "racps_raw_selector_labels": (
                shared_base_seconds
                + raw_relation_scoring_seconds
                + signature_seconds
                + selection_seconds["racps_raw_selector_labels"]
            ),
            "racps_raw_end_to_end": (
                shared_base_seconds
                + raw_relation_scoring_seconds
                + signature_seconds
                + selection_seconds["racps_raw_end_to_end"]
            ),
        }

        for condition in condition_order:
            selected = selections[condition]
            raw_labels = condition == "racps_raw_end_to_end"
            evidence_token_count = evidence_tokens(
                selected, tokenizer, raw_labels=raw_labels
            )
            if evidence_token_count > args.budget:
                raise AssertionError(
                    f"Budget violation for {row['id']} {condition}: "
                    f"{evidence_token_count}>{args.budget}"
                )
            evidence = render_evidence(selected, raw_labels=raw_labels)
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
                "selected_evidence": [
                    path.raw_text() if raw_labels else path.text()
                    for path in selected
                ],
                "relation_label_style": "raw" if raw_labels else "verbalized",
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
