#!/usr/bin/env python3
"""Frozen decoding-sensitivity experiment on fixed MetaQA evidence."""

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


CONDITIONS = ("bm25", "ra_mmr", "oracle")
SAMPLING_SEEDS = (20260818, 20260819, 20260820)


def evidence_body(row: dict[str, object]) -> str:
    return "\n".join(f"- {value}" for value in row.get("selected_evidence", []))


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    def mean(key: str) -> float:
        return round(statistics.fmean(float(row[key]) for row in rows), 6)

    return {
        "records": len(rows),
        "macro_f1": mean("f1"),
        "constrained_macro_f1": mean("candidate_constrained_f1"),
        "hits_at_1": mean("hits_at_1"),
        "exact_set_match": mean("exact_set_match"),
        "mean_evidence_answer_recall": mean("evidence_answer_recall"),
        "mean_input_tokens": mean("total_input_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "output_cap_rate": mean("output_hit_max_new_tokens"),
        "mean_generation_seconds": mean("generation_seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--sample-per-hop", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 1 <= args.sample_per_hop <= 100:
        raise ValueError("sample-per-hop must be in [1, 100]")
    if args.seed != 20260818 or args.budget != 256:
        raise ValueError("Frozen protocol requires selection seed=20260818 and B256")
    if args.temperature != 0.7 or args.top_p != 0.9:
        raise ValueError("Frozen stochastic decoder requires temperature=0.7, top_p=0.9")
    if args.max_new_tokens != 96:
        raise ValueError("Frozen protocol requires max_new_tokens=96")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")

    reference_path = args.reference_results.resolve()
    manifest_path = args.sample_manifest.resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if reference.get("status") != "PASS" or manifest.get("status") != "PASS":
        raise ValueError("Reference or manifest status is not PASS")
    if manifest.get("source_result_sha256") != reference_sha:
        raise ValueError("Manifest/reference SHA mismatch")
    config = reference.get("configuration", {})
    expected_config = {
        "split": "test",
        "sample_per_hop": 300,
        "seed": 20260818,
        "budget": 256,
        "max_paths": 100_000,
    }
    if {key: config.get(key) for key in expected_config} != expected_config:
        raise ValueError("Reference configuration mismatch")
    metadata = reference.get("model", {})
    if metadata.get("id") != MODEL_ID or metadata.get("revision") != MODEL_REVISION:
        raise ValueError("Reference model identity mismatch")

    all_manifest_rows = manifest.get("questions", [])
    if len(all_manifest_rows) != 900:
        raise ValueError("Frozen manifest question count mismatch")
    manifest_rows = []
    for hop in (1, 2, 3):
        hop_rows = [row for row in all_manifest_rows if int(row["hop"]) == hop]
        if len(hop_rows) != 300:
            raise ValueError(f"Frozen manifest hop-{hop} count mismatch")
        manifest_rows.extend(hop_rows[: args.sample_per_hop])

    reference_index = {}
    for row in reference.get("results", []):
        key = (str(row["id"]), str(row["condition"]))
        if key in reference_index:
            raise ValueError(f"Duplicate reference row {key}")
        reference_index[key] = row
    selected_rows = []
    for manifest_row in manifest_rows:
        for condition in CONDITIONS:
            key = (str(manifest_row["id"]), condition)
            if key not in reference_index:
                raise ValueError(f"Missing reference selection {key}")
            row = reference_index[key]
            for field in ("hop", "line_number", "question", "topic_entity"):
                if row.get(field) != manifest_row.get(field):
                    raise ValueError(f"Manifest/reference mismatch {key}/{field}")
            if int(row["evidence_tokens"]) > args.budget:
                raise ValueError(f"Reference budget violation {key}")
            selected_rows.append(row)

    processes = running_gpu_compute_processes()
    if processes:
        raise RuntimeError("GPU safety check failed: " + "; ".join(processes))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model_dir = args.model_dir.resolve()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError("Model config is missing")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    started = time.perf_counter()
    print("gpu_safety_check=PASS", flush=True)
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
            "selection_seed": args.seed,
            "selection_budget": args.budget,
            "conditions": CONDITIONS,
            "selection_source": "exact frozen evidence strings; selection executed once",
            "greedy_source": "copied from the frozen primary result",
            "stochastic_seeds": SAMPLING_SEEDS,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": 1.10,
            "no_repeat_ngram_size": 3,
            "analysis_status": "prospectively_frozen_internal_protocol_20260821",
        },
        "sources": {
            "reference_results": str(reference_path),
            "reference_results_sha256": reference_sha,
            "sample_manifest": str(manifest_path),
            "sample_manifest_sha256": manifest_sha,
        },
        "model_load_seconds": round(model_load_seconds, 6),
        "results": [],
    }

    for source in selected_rows:
        copied = dict(source)
        copied["decoding"] = "greedy"
        copied["sampling_seed"] = None
        copied["result_source"] = "frozen_reference"
        payload["results"].append(copied)
    atomic_json(output_path, payload)
    print(f"greedy_rows_reused={len(selected_rows)}", flush=True)

    generated_count = 0
    for sampling_seed in SAMPLING_SEEDS:
        random.seed(sampling_seed)
        torch.manual_seed(sampling_seed)
        torch.cuda.manual_seed_all(sampling_seed)
        print(f"sampling_seed={sampling_seed}", flush=True)
        for source in selected_rows:
            evidence = evidence_body(source)
            prompt, manual_prefill = build_prompt(
                tokenizer, str(source["question"]), evidence
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            total_input_tokens = int(inputs["input_ids"].shape[-1])
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(0)
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
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
            gold = [str(value) for value in source["gold_answers"]]
            metrics = answer_metrics(predictions, gold)
            allowed = {normalize_entity(value) for value in source["selected_endpoints"]}
            constrained = [value for value in predictions if value in allowed]
            constrained_metrics = answer_metrics(constrained, gold)
            result = {
                "id": source["id"],
                "hop": int(source["hop"]),
                "line_number": int(source["line_number"]),
                "condition": source["condition"],
                "decoding": "temperature_0.7_top_p_0.9",
                "sampling_seed": sampling_seed,
                "result_source": "generated",
                "question": source["question"],
                "topic_entity": source["topic_entity"],
                "gold_answers": gold,
                "selected_evidence": source["selected_evidence"],
                "selected_endpoints": source["selected_endpoints"],
                "selected_paths": int(source["selected_paths"]),
                "evidence_tokens": int(source["evidence_tokens"]),
                "evidence_answer_recall": float(source["evidence_answer_recall"]),
                "raw_response": response,
                "parsed_predictions": predictions,
                **metrics,
                "candidate_constrained_predictions": constrained,
                "candidate_constrained_f1": constrained_metrics["f1"],
                "candidate_constrained_hits_at_1": constrained_metrics["hits_at_1"],
                "total_input_tokens": total_input_tokens,
                "output_tokens": output_tokens,
                "output_hit_max_new_tokens": output_tokens == args.max_new_tokens,
                "generation_seconds": round(generation_seconds, 6),
                "generation_peak_mib": mib(torch.cuda.max_memory_allocated(0)),
                "manual_non_thinking_prefill": manual_prefill,
            }
            payload["results"].append(result)
            generated_count += 1
            print(
                f"{generated_count}/{len(selected_rows) * len(SAMPLING_SEEDS)} "
                f"seed={sampling_seed} {source['id']} condition={source['condition']} "
                f"f1={metrics['f1']:.3f} gen_s={generation_seconds:.3f}",
                flush=True,
            )
            if generated_count % (args.checkpoint_every * len(CONDITIONS)) == 0:
                atomic_json(output_path, payload)
                print(f"checkpoint_generated={generated_count}", flush=True)

    expected_rows = len(selected_rows) * (1 + len(SAMPLING_SEEDS))
    results = payload["results"]
    if len(results) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, observed {len(results)}")
    summary = {}
    decoding_labels = ("greedy", "temperature_0.7_top_p_0.9")
    for decoding in decoding_labels:
        for condition in CONDITIONS:
            for hop in (1, 2, 3):
                subset = [
                    row
                    for row in results
                    if row["decoding"] == decoding
                    and row["condition"] == condition
                    and int(row["hop"]) == hop
                ]
                summary[f"{decoding}/{condition}/{hop}-hop"] = summarize(subset)
            subset = [
                row
                for row in results
                if row["decoding"] == decoding and row["condition"] == condition
            ]
            summary[f"{decoding}/{condition}/all-hops"] = summarize(subset)
    payload["summary"] = summary
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    payload["status"] = "PASS"
    atomic_json(output_path, payload)
    print("status=PASS", flush=True)
    for name, values in summary.items():
        if name.endswith("all-hops"):
            print(
                f"result={name} f1={values['macro_f1']} "
                f"hits1={values['hits_at_1']} cap={values['output_cap_rate']}",
                flush=True,
            )
    print(f"report={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"status=FAIL error={error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
