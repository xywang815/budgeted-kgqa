#!/usr/bin/env python3
"""Paired statistical analysis for frozen MetaQA inference results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


METRICS = {
    "raw_f1": "f1",
    "constrained_f1": "candidate_constrained_f1",
    "hits_at_1": "hits_at_1",
    "evidence_recall": "evidence_answer_recall",
    "endpoint_f1": "selected_endpoint_f1",
    "input_tokens": "total_input_tokens",
    "retrieval_seconds": "retrieval_seconds",
    "generation_seconds": "generation_seconds",
}
BASELINES = ("bfs", "bm25", "ba_mmr")


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    family_size = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * p_values[name])
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def paired_permutation_p(
    differences: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
    chunk_size: int = 1000,
) -> float:
    observed = abs(float(differences.mean()))
    if np.allclose(differences, 0.0):
        return 1.0
    exceedances = 0
    completed = 0
    while completed < iterations:
        count = min(chunk_size, iterations - completed)
        signs = rng.integers(0, 2, size=(count, len(differences)), dtype=np.int8)
        signs = signs * 2 - 1
        permuted = np.abs((signs * differences).mean(axis=1))
        exceedances += int(np.count_nonzero(permuted >= observed - 1e-15))
        completed += count
    return (exceedances + 1.0) / (iterations + 1.0)


def stratified_bootstrap_difference(
    grouped_differences: list[np.ndarray],
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    total = sum(len(values) for values in grouped_differences)
    bootstrap = np.zeros(iterations, dtype=np.float64)
    for values in grouped_differences:
        indices = rng.integers(0, len(values), size=(iterations, len(values)))
        bootstrap += values[indices].mean(axis=1) * (len(values) / total)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return float(low), float(high)


def mean_metric(
    records: dict[tuple[int, str], dict[str, object]],
    keys: list[tuple[int, str]],
    field: str,
) -> float:
    return statistics.fmean(float(records[key][field]) for key in keys)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=50_000)
    args = parser.parse_args()

    if args.bootstrap_iterations < 1000 or args.permutation_iterations < 1000:
        raise ValueError("statistical iteration counts must each be at least 1000")

    input_path = args.input.resolve()
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if source.get("status") != "PASS":
        raise ValueError("input experiment is not marked PASS")
    rows = source.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("input experiment has no result rows")

    by_condition: dict[str, dict[tuple[int, str], dict[str, object]]] = {}
    for row in rows:
        key = (int(row["hop"]), str(row["id"]))
        condition = str(row["condition"])
        condition_rows = by_condition.setdefault(condition, {})
        if key in condition_rows:
            raise ValueError(f"duplicate result for condition={condition} key={key}")
        condition_rows[key] = row

    required = {"direct", "bfs", "bm25", "ba_mmr", "ra_mmr", "oracle"}
    missing = sorted(required - set(by_condition))
    if missing:
        raise ValueError(f"missing conditions: {missing}")
    reference_keys = set(by_condition["ra_mmr"])
    for condition in sorted(required):
        if set(by_condition[condition]) != reference_keys:
            raise ValueError(f"question mismatch for condition={condition}")

    # Development-frozen routing is derived from existing deterministic outputs;
    # no additional model generation or test-set parameter choice is performed.
    by_condition["dr_mmr"] = {
        key: (
            by_condition["ba_mmr"][key]
            if key[0] <= 2
            else by_condition["ra_mmr"][key]
        )
        for key in reference_keys
    }

    all_keys = sorted(reference_keys)
    scopes = {
        "1-hop": [key for key in all_keys if key[0] == 1],
        "2-hop": [key for key in all_keys if key[0] == 2],
        "3-hop": [key for key in all_keys if key[0] == 3],
        "all-hops": all_keys,
    }
    counts = {name: len(keys) for name, keys in scopes.items()}
    if any(value == 0 for value in counts.values()):
        raise ValueError(f"empty analysis scope: {counts}")

    method_summary: dict[str, dict[str, dict[str, float]]] = {}
    for scope, keys in scopes.items():
        method_summary[scope] = {}
        for condition in sorted(by_condition):
            method_summary[scope][condition] = {
                metric: round(
                    mean_metric(by_condition[condition], keys, field), 9
                )
                for metric, field in METRICS.items()
            }

    rng = np.random.default_rng(args.seed)
    comparisons: list[dict[str, object]] = []
    primary_p_values: dict[str, float] = {}
    for scope, keys in scopes.items():
        hop_groups = sorted({key[0] for key in keys})
        for baseline in BASELINES:
            for metric, field in METRICS.items():
                target_values = np.asarray(
                    [float(by_condition["ra_mmr"][key][field]) for key in keys],
                    dtype=np.float64,
                )
                baseline_values = np.asarray(
                    [float(by_condition[baseline][key][field]) for key in keys],
                    dtype=np.float64,
                )
                differences = target_values - baseline_values
                grouped_differences = [
                    np.asarray(
                        [
                            float(by_condition["ra_mmr"][key][field])
                            - float(by_condition[baseline][key][field])
                            for key in keys
                            if key[0] == hop
                        ],
                        dtype=np.float64,
                    )
                    for hop in hop_groups
                ]
                ci_low, ci_high = stratified_bootstrap_difference(
                    grouped_differences,
                    args.bootstrap_iterations,
                    rng,
                )
                p_value = None
                if metric == "raw_f1":
                    p_value = paired_permutation_p(
                        differences,
                        args.permutation_iterations,
                        rng,
                    )
                comparison = {
                    "scope": scope,
                    "target": "ra_mmr",
                    "baseline": baseline,
                    "metric": metric,
                    "questions": len(keys),
                    "target_mean": round(float(target_values.mean()), 9),
                    "baseline_mean": round(float(baseline_values.mean()), 9),
                    "mean_difference": round(float(differences.mean()), 9),
                    "bootstrap_95_ci_low": round(ci_low, 9),
                    "bootstrap_95_ci_high": round(ci_high, 9),
                    "paired_permutation_p": (
                        round(float(p_value), 9) if p_value is not None else None
                    ),
                    "holm_adjusted_p": None,
                    "target_wins": int(np.count_nonzero(differences > 1e-12)),
                    "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
                    "target_losses": int(np.count_nonzero(differences < -1e-12)),
                }
                comparisons.append(comparison)
                if scope == "all-hops" and metric == "raw_f1":
                    primary_p_values[baseline] = float(p_value)

    adjusted = holm_adjust(primary_p_values)
    for comparison in comparisons:
        if comparison["scope"] == "all-hops" and comparison["metric"] == "raw_f1":
            comparison["holm_adjusted_p"] = round(
                adjusted[str(comparison["baseline"])], 9
            )

    dr_keys = scopes["all-hops"]
    dr_differences = np.asarray(
        [
            float(by_condition["dr_mmr"][key]["f1"])
            - float(by_condition["ra_mmr"][key]["f1"])
            for key in dr_keys
        ],
        dtype=np.float64,
    )
    dr_grouped = [
        np.asarray([dr_differences[index] for index, key in enumerate(dr_keys) if key[0] == hop])
        for hop in (1, 2, 3)
    ]
    dr_ci = stratified_bootstrap_difference(
        dr_grouped, args.bootstrap_iterations, rng
    )
    routing_diagnostic = {
        "comparison": "dr_mmr_minus_ra_mmr",
        "metric": "raw_f1",
        "questions": len(dr_keys),
        "mean_difference": round(float(dr_differences.mean()), 9),
        "bootstrap_95_ci_low": round(dr_ci[0], 9),
        "bootstrap_95_ci_high": round(dr_ci[1], 9),
        "paired_permutation_p": round(
            paired_permutation_p(
                dr_differences, args.permutation_iterations, rng
            ),
            9,
        ),
        "interpretation": "secondary diagnostic; routing policy was frozen on dev",
    }

    payload = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "configuration": {
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap": "paired percentile, stratified by hop for all-hops scope",
            "permutation_iterations": args.permutation_iterations,
            "permutation_test": "two-sided paired sign randomization",
            "multiple_comparison": (
                "Holm correction across the three pre-specified all-hops raw-F1 "
                "comparisons: RA-MMR versus BFS, BM25, and BA-MMR"
            ),
            "primary_metric": "raw macro answer-set F1",
        },
        "question_counts": counts,
        "method_summary": method_summary,
        "comparisons": comparisons,
        "routing_diagnostic": routing_diagnostic,
    }
    atomic_json(args.output_json.resolve(), payload)

    args.output_csv.resolve().parent.mkdir(parents=True, exist_ok=True)
    csv_fields = list(comparisons[0].keys())
    with args.output_csv.resolve().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(comparisons)

    print("status=PASS", flush=True)
    print(f"questions={counts}", flush=True)
    for baseline in BASELINES:
        row = next(
            item
            for item in comparisons
            if item["scope"] == "all-hops"
            and item["metric"] == "raw_f1"
            and item["baseline"] == baseline
        )
        print(
            f"primary=ra_mmr_vs_{baseline} "
            f"target={row['target_mean']:.6f} baseline={row['baseline_mean']:.6f} "
            f"diff={row['mean_difference']:.6f} "
            f"ci=[{row['bootstrap_95_ci_low']:.6f},"
            f"{row['bootstrap_95_ci_high']:.6f}] "
            f"p={row['paired_permutation_p']:.6g} "
            f"holm_p={row['holm_adjusted_p']:.6g} "
            f"wins/ties/losses={row['target_wins']}/{row['ties']}/"
            f"{row['target_losses']}",
            flush=True,
        )
    for scope in ("1-hop", "2-hop", "3-hop"):
        row = next(
            item
            for item in comparisons
            if item["scope"] == scope
            and item["metric"] == "raw_f1"
            and item["baseline"] == "bm25"
        )
        print(
            f"hop_result={scope}/ra_vs_bm25 diff={row['mean_difference']:.6f} "
            f"ci=[{row['bootstrap_95_ci_low']:.6f},"
            f"{row['bootstrap_95_ci_high']:.6f}] "
            f"p={row['paired_permutation_p']:.6g}",
            flush=True,
        )
    print(
        f"routing=dr_minus_ra diff={routing_diagnostic['mean_difference']:.6f} "
        f"ci=[{routing_diagnostic['bootstrap_95_ci_low']:.6f},"
        f"{routing_diagnostic['bootstrap_95_ci_high']:.6f}] "
        f"p={routing_diagnostic['paired_permutation_p']:.6g}",
        flush=True,
    )
    print(f"json={args.output_json.resolve()}", flush=True)
    print(f"csv={args.output_csv.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"status=FAIL error={error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
