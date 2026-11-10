#!/usr/bin/env python3
"""Run a second reader on exact evidence selected by the frozen Qwen run."""

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

from run_kgqa_inference_multimodel import (
    answer_metrics,
    atomic_json,
    build_prompt,
    mib,
    normalize_entity,
    package_version,
    parse_entities,
    running_gpu_compute_processes,
)


QWEN_ID = "Qwen/Qwen3-8B-AWQ"
QWEN_REVISION = "cb7d6a337aadb4d2082ed0dcef1032e4f8645194"
PHI_ID = "microsoft/Phi-4-mini-instruct"
PHI_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
CONDITIONS = ("direct", "bfs", "bm25", "ba_mmr", "ra_mmr", "oracle")


def evidence_body(row: dict[str, object]) -> str:
    return "\n".join(f"- {value}" for value in row.get("selected_evidence", []))


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    def mean(key: str) -> float:
        return round(statistics.fmean(float(row[key]) for row in rows), 6)

    return {
        "questions": len(rows),
        "macro_f1": mean("f1"),
        "mean_precision": mean("precision"),
        "mean_answer_recall": mean("recall"),
        "hits_at_1": mean("hits_at_1"),
        "exact_set_match": mean("exact_set_match"),
        "candidate_constrained_macro_f1": mean("candidate_constrained_f1"),
        "mean_evidence_answer_recall": mean("evidence_answer_recall"),
        "mean_selector_tokenizer_evidence_tokens": mean("selector_tokenizer_evidence_tokens"),
        "mean_reader_tokenizer_evidence_tokens": mean("reader_tokenizer_evidence_tokens"),
        "reader_tokenizer_over_256_rate": mean("reader_tokenizer_over_256"),
        "mean_total_input_tokens": mean("total_input_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "mean_generation_seconds": mean("generation_seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--selector-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--reader-model-dir", type=Path, required=True)
    parser.add_argument("--sample-per-hop", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 1 <= args.sample_per_hop <= 300:
        raise ValueError("sample-per-hop must be in [1, 300]")
    if args.seed != 20260818:
        raise ValueError("Frozen protocol requires seed=20260818")
    if args.max_new_tokens != 96:
        raise ValueError("Frozen protocol requires max_new_tokens=96")
    if args.output.resolve().exists():
        raise FileExistsError(f"Refusing to overwrite {args.output.resolve()}")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")

    reference_path = args.reference_results.resolve()
    manifest_path = args.sample_manifest.resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if reference.get("status") != "PASS" or manifest.get("status") != "PASS":
        raise ValueError("Reference or manifest status is not PASS")
    config = reference.get("configuration", {})
    expected = {
        "split": "test",
        "sample_per_hop": 300,
        "seed": 20260818,
        "budget": 256,
        "max_paths": 100_000,
    }
    if {key: config.get(key) for key in expected} != expected:
        raise ValueError("Reference configuration mismatch")
    model_metadata = reference.get("model", {})
    if model_metadata.get("id") != QWEN_ID or model_metadata.get("revision") != QWEN_REVISION:
        raise ValueError("Reference selector model identity mismatch")
    if manifest.get("source_result_sha256") != reference_sha:
        raise ValueError("Manifest/reference SHA mismatch")

    all_manifest_rows = manifest.get("questions", [])
    if len(all_manifest_rows) != 900:
        raise ValueError("Manifest question count mismatch")
    manifest_rows = []
    for hop in (1, 2, 3):
        hop_rows = [row for row in all_manifest_rows if int(row["hop"]) == hop]
        if len(hop_rows) != 300:
            raise ValueError(f"Manifest hop-{hop} count mismatch")
        manifest_rows.extend(hop_rows[: args.sample_per_hop])
    reference_rows = reference.get("results", [])
    indexed = {}
    for row in reference_rows:
        key = (str(row["id"]), str(row["condition"]))
        if key in indexed:
            raise ValueError(f"Duplicate reference key: {key}")
        indexed[key] = row
    expected_keys = {
        (str(row["id"]), condition)
        for row in manifest_rows
        for condition in CONDITIONS
    }
    missing_keys = expected_keys - set(indexed)
    if missing_keys:
        raise ValueError(
            "Reference rows are missing manifest IDs or conditions: "
            f"{sorted(missing_keys)[:5]}"
        )

    ordered_rows = []
    for manifest_row in manifest_rows:
        for condition in CONDITIONS:
            row = indexed[(str(manifest_row["id"]), condition)]
            for field in ("hop", "line_number", "question", "topic_entity"):
                if row.get(field) != manifest_row.get(field):
                    raise ValueError(f"Manifest/reference mismatch: {row['id']}/{field}")
            ordered_rows.append(row)
    expected_rows = args.sample_per_hop * 3 * len(CONDITIONS)
    if len(ordered_rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} ordered reference rows")

    processes = running_gpu_compute_processes()
    if processes:
        raise RuntimeError("GPU safety check failed: " + "; ".join(processes))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    selector_dir = args.selector_tokenizer_dir.resolve()
    reader_dir = args.reader_model_dir.resolve()
    if not (selector_dir / "tokenizer_config.json").is_file():
        raise FileNotFoundError("Selector tokenizer is missing")
    if not (reader_dir / "config.json").is_file():
        raise FileNotFoundError("Reader model is missing")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    started = time.perf_counter()

    print("gpu_safety_check=PASS", flush=True)
    selector_tokenizer = AutoTokenizer.from_pretrained(
        selector_dir, local_files_only=True, trust_remote_code=False
    )
    reader_tokenizer = AutoTokenizer.from_pretrained(
        reader_dir, local_files_only=True, trust_remote_code=True
    )
    load_started = time.perf_counter()
    reader = AutoModelForCausalLM.from_pretrained(
        reader_dir,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    reader.eval()
    torch.cuda.synchronize(0)
    model_load_seconds = time.perf_counter() - load_started

    warmup_prompt, _ = build_prompt(
        reader_tokenizer,
        "What is the capital of France?",
        "- France --[capital]--> Paris",
        "generic",
    )
    warmup = reader_tokenizer(warmup_prompt, return_tensors="pt").to(reader.device)
    with torch.inference_mode():
        reader.generate(
            **warmup,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=reader_tokenizer.eos_token_id,
        )
    torch.cuda.synchronize(0)
    del warmup
    print("generation_warmup=PASS", flush=True)

    payload: dict[str, object] = {
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_source": {
            "model_id": QWEN_ID,
            "model_revision": QWEN_REVISION,
            "reference_results": str(reference_path),
            "reference_results_sha256": reference_sha,
            "sample_manifest": str(manifest_path),
            "sample_manifest_sha256": manifest_sha,
            "selector_tokenizer_path": str(selector_dir),
        },
        "model": {
            "id": PHI_ID,
            "revision": PHI_REVISION,
            "local_path": str(reader_dir),
            "class": reader.__class__.__name__,
            "dtype": str(reader.dtype),
            "memory_footprint_mib": mib(int(reader.get_memory_footprint())),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "configuration": {
            "split": "test",
            "sample_per_hop": args.sample_per_hop,
            "seed": args.seed,
            "selection_budget": 256,
            "selection_budget_tokenizer": QWEN_ID,
            "evidence_sharing": "exact path strings from the frozen Qwen selection",
            "conditions": CONDITIONS,
            "prompt_family": "generic",
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": 1.10,
            "no_repeat_ngram_size": 3,
            "checkpoint_every_rows": args.checkpoint_every * len(CONDITIONS),
            "analysis_status": "prospectively_frozen_internal_protocol_20260821",
        },
        "model_load_seconds": round(model_load_seconds, 6),
        "results": [],
    }
    atomic_json(args.output.resolve(), payload)

    for index, source in enumerate(ordered_rows, start=1):
        evidence = evidence_body(source)
        selector_tokens = token_count(selector_tokenizer, evidence)
        if selector_tokens != int(source["evidence_tokens"]):
            raise ValueError(
                f"Selector token recount mismatch {source['id']}/{source['condition']}: "
                f"{selector_tokens}!={source['evidence_tokens']}"
            )
        reader_tokens = token_count(reader_tokenizer, evidence)
        prompt, manual_prefill = build_prompt(
            reader_tokenizer,
            str(source["question"]),
            evidence,
            "generic",
        )
        inputs = reader_tokenizer(prompt, return_tensors="pt").to(reader.device)
        total_input_tokens = int(inputs["input_ids"].shape[-1])
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated = reader.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=reader_tokenizer.eos_token_id,
                repetition_penalty=1.10,
                no_repeat_ngram_size=3,
            )
        torch.cuda.synchronize(0)
        generation_seconds = time.perf_counter() - generation_started
        output_ids = generated[0, total_input_tokens:]
        output_tokens = int(output_ids.shape[-1])
        response = reader_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        predictions = parse_entities(response)
        gold = [str(value) for value in source["gold_answers"]]
        metrics = answer_metrics(predictions, gold)
        allowed = {normalize_entity(value) for value in source["selected_endpoints"]}
        constrained = (
            predictions
            if source["condition"] == "direct"
            else [value for value in predictions if value in allowed]
        )
        constrained_metrics = answer_metrics(constrained, gold)
        result = {
            "id": source["id"],
            "hop": int(source["hop"]),
            "line_number": int(source["line_number"]),
            "condition": source["condition"],
            "question": source["question"],
            "topic_entity": source["topic_entity"],
            "gold_answers": gold,
            "selection_reader": QWEN_ID,
            "generation_reader": PHI_ID,
            "selected_evidence": source["selected_evidence"],
            "selected_endpoints": source["selected_endpoints"],
            "selected_paths": int(source["selected_paths"]),
            "evidence_answer_recall": float(source["evidence_answer_recall"]),
            "selector_tokenizer_evidence_tokens": selector_tokens,
            "reader_tokenizer_evidence_tokens": reader_tokens,
            "reader_tokenizer_over_256": reader_tokens > 256,
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
        print(
            f"{index}/{len(ordered_rows)} {source['id']} "
            f"condition={source['condition']} f1={metrics['f1']:.3f} "
            f"qwen_tokens={selector_tokens} phi_tokens={reader_tokens} "
            f"gen_s={generation_seconds:.3f}",
            flush=True,
        )
        if index % (args.checkpoint_every * len(CONDITIONS)) == 0 or index == len(ordered_rows):
            atomic_json(args.output.resolve(), payload)
            print(f"checkpoint={index}/{len(ordered_rows)}", flush=True)

    results = payload["results"]
    if len(results) != expected_rows:
        raise AssertionError(
            f"Shared-evidence run did not produce {expected_rows} rows"
        )
    summary = {}
    for hop in (1, 2, 3):
        for condition in CONDITIONS:
            subset = [row for row in results if int(row["hop"]) == hop and row["condition"] == condition]
            summary[f"{hop}-hop/{condition}"] = summarize(subset)
    for condition in CONDITIONS:
        subset = [row for row in results if row["condition"] == condition]
        summary[f"all-hops/{condition}"] = summarize(subset)
    payload["summary"] = summary
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    payload["status"] = "PASS"
    atomic_json(args.output.resolve(), payload)
    print("status=PASS", flush=True)
    for name, values in summary.items():
        print(
            f"result={name} f1={values['macro_f1']} hits1={values['hits_at_1']} "
            f"qwen_tokens={values['mean_selector_tokenizer_evidence_tokens']} "
            f"phi_tokens={values['mean_reader_tokenizer_evidence_tokens']} "
            f"phi_over256={values['reader_tokenizer_over_256_rate']}",
            flush=True,
        )
    print(f"report={args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"status=FAIL error={error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
