#!/usr/bin/env python3
"""Frozen MetaQA hop-knowledge robustness experiment for week 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
import traceback
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
    load_kg,
    pack_in_order,
    rank_by_scores,
    relation_bm25_scores,
    render_evidence,
    standalone_evidence_tokens,
)
from run_week2_components import (
    MODEL_ID,
    MODEL_REVISION,
    answer_metrics,
    atomic_json,
    build_prompt,
    mib,
    normalize_entity,
    package_version,
    parse_entities,
    running_gpu_compute_processes,
)


CONDITIONS = ("bfs", "bm25", "ra_mmr")
SCENARIO_DEPTHS = {
    1: {"unknown_hop_1to3": (1, 2, 3), "wrong_long": (2,)},
    2: {
        "unknown_hop_1to3": (1, 2, 3),
        "wrong_short": (1,),
        "wrong_long": (3,),
    },
    3: {"unknown_hop_1to3": (1, 2, 3), "wrong_short_only": (2,)},
}


def load_frozen_rows(
    reference_path: Path,
    manifest_path: Path,
    sample_per_hop: int,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    if reference_payload.get("status") != "PASS":
        raise ValueError("Reference result status is not PASS")
    expected_configuration = {
        "split": "test",
        "sample_per_hop": 300,
        "seed": 20260818,
        "budget": 256,
        "max_paths": 100_000,
    }
    configuration = reference_payload.get("configuration", {})
    observed = {key: configuration.get(key) for key in expected_configuration}
    if observed != expected_configuration:
        raise ValueError(f"Reference configuration mismatch: {observed}")
    model = reference_payload.get("model", {})
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ValueError("Reference model identity or revision mismatch")

    if manifest_payload.get("status") != "PASS":
        raise ValueError("Sample manifest status is not PASS")
    if manifest_payload.get("split") != "test":
        raise ValueError("Sample manifest split mismatch")
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

    results = reference_payload.get("results", [])
    reference_rows = [row for row in results if row.get("condition") == "ra_mmr"]
    if len(results) != 5400 or len(reference_rows) != 900:
        raise ValueError("Reference result cardinality mismatch")
    reference_by_id = {str(row["id"]): row for row in reference_rows}
    if len(reference_by_id) != 900:
        raise ValueError("Reference RA-MMR identifiers are not unique")

    manifest_rows = manifest_payload.get("questions", [])
    manifest_ids = [str(row["id"]) for row in manifest_rows]
    if len(manifest_ids) != 900 or len(set(manifest_ids)) != 900:
        raise ValueError("Sample manifest identifier cardinality mismatch")
    if set(manifest_ids) != set(reference_by_id):
        raise ValueError("Sample manifest identifiers do not match reference")

    selected: list[dict[str, object]] = []
    for hop in (1, 2, 3):
        hop_rows = [row for row in manifest_rows if int(row["hop"]) == hop]
        if len(hop_rows) != 300:
            raise ValueError(f"Manifest {hop}-hop count mismatch")
        for manifest_row in hop_rows[:sample_per_hop]:
            row = reference_by_id[str(manifest_row["id"])]
            keys = ("hop", "line_number", "question", "topic_entity")
            if any(row.get(key) != manifest_row.get(key) for key in keys):
                raise ValueError(f"Manifest/reference mismatch for {row['id']}")
            selected.append(
                {
                    "id": row["id"],
                    "hop": int(row["hop"]),
                    "line_number": int(row["line_number"]),
                    "question": row["question"],
                    "topic_entity": row["topic_entity"],
                    "gold_answers": row["gold_answers"],
                }
            )
    return selected, {
        "reference_sha256": reference_sha256,
        "manifest_sha256": manifest_sha256,
    }


def enumerate_depth_pool(adjacency, topic: str, depths: tuple[int, ...], cap: int):
    """Pool exact-depth paths in ascending depth order under one global cap."""
    pooled = []
    counts: dict[str, int] = {}
    truncated = False
    for depth in depths:
        remaining = cap - len(pooled)
        if remaining <= 0:
            counts[str(depth)] = 0
            truncated = True
            continue
        paths, depth_truncated = enumerate_simple_paths(
            adjacency,
            topic,
            depth,
            remaining,
            exact_hops_only=True,
        )
        counts[str(depth)] = len(paths)
        pooled.extend(paths)
        truncated = truncated or depth_truncated
    return pooled, counts, truncated


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    def mean(key: str) -> float:
        return round(statistics.fmean(float(row[key]) for row in rows), 6)

    return {
        "questions": len(rows),
        "macro_f1": mean("f1"),
        "hits_at_1": mean("hits_at_1"),
        "exact_set_match": mean("exact_set_match"),
        "candidate_constrained_macro_f1": mean("candidate_constrained_f1"),
        "mean_evidence_answer_recall": mean("evidence_answer_recall"),
        "mean_candidate_answer_recall": mean("candidate_answer_recall"),
        "mean_selected_endpoint_f1": mean("selected_endpoint_f1"),
        "mean_evidence_tokens": mean("evidence_tokens"),
        "mean_total_input_tokens": mean("total_input_tokens"),
        "mean_retrieval_seconds": mean("retrieval_seconds"),
        "mean_generation_seconds": mean("generation_seconds"),
        "candidate_truncation_rate": mean("candidate_truncated"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--sample-per-hop", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--max-paths", type=int, default=100_000)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 1 <= args.sample_per_hop <= 300:
        raise ValueError("sample-per-hop must be in [1, 300]")
    if args.seed != 20260818 or args.budget != 256 or args.max_paths != 100_000:
        raise ValueError("Frozen protocol requires seed=20260818, B256, cap=100000")
    if args.max_new_tokens != 96:
        raise ValueError("Frozen protocol requires max_new_tokens=96")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    if args.output.resolve().exists():
        raise FileExistsError(f"Refusing to overwrite {args.output.resolve()}")
    if not (args.data_dir.resolve() / "kg.tsv").is_file():
        raise FileNotFoundError("Processed KG is missing")
    if not (args.model_dir.resolve() / "config.json").is_file():
        raise FileNotFoundError("Model config is missing")

    rows, hashes = load_frozen_rows(
        args.reference_results.resolve(),
        args.sample_manifest.resolve(),
        args.sample_per_hop,
    )
    expected_records = args.sample_per_hop * 7 * len(CONDITIONS)
    if args.preflight_only:
        print("status=PASS")
        print("mode=preflight_only")
        print(f"questions={len(rows)} expected_records={expected_records}")
        print(f"scenarios={SCENARIO_DEPTHS}")
        print(f"conditions={CONDITIONS}")
        print(f"reference_sha256={hashes['reference_sha256']}")
        print(f"sample_manifest_sha256={hashes['manifest_sha256']}")
        return 0

    processes = running_gpu_compute_processes()
    if processes:
        raise RuntimeError("GPU safety check failed: " + "; ".join(processes))
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
    print("gpu_safety_check=PASS", flush=True)
    print("loading_processed_kg...", flush=True)
    adjacency = load_kg(data_dir / "kg.tsv")
    print(f"kg_entities={len(adjacency)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
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
            "split": "test",
            "sample_per_hop": args.sample_per_hop,
            "seed": args.seed,
            "budget": args.budget,
            "budget_definition": "tokenized evidence body only",
            "max_paths": args.max_paths,
            "candidate_cap_scope": "one global cap per scenario pool",
            "unknown_pool_order": "exact depths 1,2,3 concatenated in ascending order",
            "scenarios": SCENARIO_DEPTHS,
            "conditions": CONDITIONS,
            "ra_mmr": {
                "relevance": "direction-aware relation BM25",
                "relevance_weight": 0.90,
                "cost_exponent": 1.00,
                "signature": "directional relation plus non-topic entities",
            },
            "reference_results": str(args.reference_results.resolve()),
            "reference_results_sha256": hashes["reference_sha256"],
            "sample_manifest": str(args.sample_manifest.resolve()),
            "sample_manifest_sha256": hashes["manifest_sha256"],
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": 1.10,
            "no_repeat_ngram_size": 3,
            "checkpoint_every_questions": args.checkpoint_every,
            "analysis_status": "prospectively_frozen_internal_protocol_20260821",
        },
        "model_load_seconds": round(model_load_seconds, 6),
        "results": [],
    }
    atomic_json(output_path, payload)

    for question_index, row in enumerate(rows, start=1):
        true_hop = int(row["hop"])
        question = str(row["question"])
        gold_answers = [str(value) for value in row["gold_answers"]]
        for scenario, depths in SCENARIO_DEPTHS[true_hop].items():
            enum_started = time.perf_counter()
            paths, depth_counts, truncated = enumerate_depth_pool(
                adjacency,
                str(row["topic_entity"]),
                depths,
                args.max_paths,
            )
            enumeration_seconds = time.perf_counter() - enum_started

            cost_started = time.perf_counter()
            costs = standalone_evidence_tokens(paths, tokenizer)
            cost_seconds = time.perf_counter() - cost_started
            bm25_score_started = time.perf_counter()
            lexical_scores = bm25_scores(question, paths)
            bm25_score_seconds = time.perf_counter() - bm25_score_started
            relation_score_started = time.perf_counter()
            relation_scores = relation_bm25_scores(question, paths)
            relation_score_seconds = time.perf_counter() - relation_score_started
            signature_started = time.perf_counter()
            signatures = [path.directional_signature() for path in paths]
            signature_seconds = time.perf_counter() - signature_started

            selection_started = time.perf_counter()
            bfs = pack_in_order(paths, args.budget, tokenizer, costs)
            bfs_seconds = time.perf_counter() - selection_started
            selection_started = time.perf_counter()
            ranked = rank_by_scores(paths, lexical_scores)
            cost_by_id = {id(path): cost for path, cost in zip(paths, costs)}
            ranked_costs = [cost_by_id[id(path)] for path in ranked]
            bm25 = pack_in_order(ranked, args.budget, tokenizer, ranked_costs)
            bm25_seconds = time.perf_counter() - selection_started
            selection_started = time.perf_counter()
            ra_mmr = budget_aware_mmr(
                paths,
                relation_scores,
                args.budget,
                tokenizer,
                relevance_weight=0.90,
                cost_exponent=1.00,
                standalone_costs=costs,
                signatures=signatures,
            )
            ra_seconds = time.perf_counter() - selection_started
            selections = {"bfs": bfs, "bm25": bm25, "ra_mmr": ra_mmr}
            retrieval_seconds = {
                "bfs": enumeration_seconds + cost_seconds + bfs_seconds,
                "bm25": (
                    enumeration_seconds
                    + cost_seconds
                    + bm25_score_seconds
                    + bm25_seconds
                ),
                "ra_mmr": (
                    enumeration_seconds
                    + cost_seconds
                    + relation_score_seconds
                    + signature_seconds
                    + ra_seconds
                ),
            }

            for condition in CONDITIONS:
                selected = selections[condition]
                evidence_token_count = evidence_tokens(selected, tokenizer)
                if evidence_token_count > args.budget:
                    raise AssertionError(
                        f"Budget violation {row['id']} {scenario}/{condition}"
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
                response = tokenizer.decode(
                    output_ids, skip_special_tokens=True
                ).strip()
                predictions = parse_entities(response)
                metrics = answer_metrics(predictions, gold_answers)
                allowed = {normalize_entity(path.endpoint) for path in selected}
                constrained = [value for value in predictions if value in allowed]
                constrained_metrics = answer_metrics(constrained, gold_answers)
                endpoints = list(
                    dict.fromkeys(normalize_entity(path.endpoint) for path in selected)
                )
                endpoint_metrics = answer_metrics(endpoints, gold_answers)
                result = {
                    "id": row["id"],
                    "hop": true_hop,
                    "true_hop": true_hop,
                    "line_number": row["line_number"],
                    "scenario": scenario,
                    "enumerated_depths": list(depths),
                    "condition": condition,
                    "budget": args.budget,
                    "question": question,
                    "topic_entity": row["topic_entity"],
                    "gold_answers": gold_answers,
                    "raw_response": response,
                    "parsed_predictions": predictions,
                    **metrics,
                    "candidate_constrained_predictions": constrained,
                    "candidate_constrained_f1": constrained_metrics["f1"],
                    "candidate_constrained_hits_at_1": constrained_metrics["hits_at_1"],
                    "candidate_paths": len(paths),
                    "candidate_paths_by_depth": depth_counts,
                    "candidate_truncated": truncated,
                    "candidate_answer_recall": answer_recall(paths, gold_answers),
                    "selected_paths": len(selected),
                    "selected_endpoints": [path.endpoint for path in selected],
                    "selected_endpoint_f1": endpoint_metrics["f1"],
                    "selected_evidence": [path.text() for path in selected],
                    "evidence_tokens": evidence_token_count,
                    "evidence_answer_recall": answer_recall(selected, gold_answers),
                    "total_input_tokens": total_input_tokens,
                    "output_tokens": output_tokens,
                    "output_hit_max_new_tokens": output_tokens == args.max_new_tokens,
                    "retrieval_seconds": round(retrieval_seconds[condition], 6),
                    "generation_seconds": round(generation_seconds, 6),
                    "generation_peak_mib": mib(torch.cuda.max_memory_allocated(0)),
                    "manual_non_thinking_prefill": manual_prefill,
                }
                payload["results"].append(result)
                print(
                    f"{question_index}/{len(rows)} {row['id']} "
                    f"scenario={scenario} condition={condition} "
                    f"f1={metrics['f1']:.3f} evidence_recall="
                    f"{result['evidence_answer_recall']:.3f} gen_s={generation_seconds:.3f}",
                    flush=True,
                )

        if question_index % args.checkpoint_every == 0 or question_index == len(rows):
            atomic_json(output_path, payload)
            print(
                f"checkpoint={question_index}/{len(rows)} records={len(payload['results'])}",
                flush=True,
            )

    results = payload["results"]
    if len(results) != expected_records:
        raise AssertionError(f"Expected {expected_records} records, got {len(results)}")
    summary: dict[str, dict[str, float]] = {}
    for hop in (1, 2, 3):
        for scenario in SCENARIO_DEPTHS[hop]:
            for condition in CONDITIONS:
                subset = [
                    row
                    for row in results
                    if int(row["hop"]) == hop
                    and row["scenario"] == scenario
                    and row["condition"] == condition
                ]
                summary[f"{hop}-hop/{scenario}/{condition}"] = summarize(subset)
    payload["summary"] = summary
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    payload["status"] = "PASS"
    atomic_json(output_path, payload)
    print("status=PASS", flush=True)
    print(f"records={len(results)}", flush=True)
    for name, values in summary.items():
        print(
            f"result={name} f1={values['macro_f1']} "
            f"evidence_recall={values['mean_evidence_answer_recall']} "
            f"candidate_recall={values['mean_candidate_answer_recall']} "
            f"tokens={values['mean_evidence_tokens']}",
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
