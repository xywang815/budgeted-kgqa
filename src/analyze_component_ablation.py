#!/usr/bin/env python3
"""Paired exploratory analysis of the post-hoc Qwen component ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FULL = "ra_mmr"
MECHANISM_ABLATIONS = (
    "racps_no_relation",
    "racps_no_cost",
    "racps_no_novelty",
)
LABEL_SENSITIVITY = (
    "racps_raw_selector_labels",
    "racps_raw_end_to_end",
)
ABLATIONS = MECHANISM_ABLATIONS + LABEL_SENSITIVITY
METRICS = {
    "raw_f1": "f1",
    "constrained_f1": "candidate_constrained_f1",
    "hits_at_1": "hits_at_1",
    "evidence_recall": "evidence_answer_recall",
    "endpoint_f1": "selected_endpoint_f1",
    "input_tokens": "total_input_tokens",
    "selected_paths": "selected_paths",
    "retrieval_seconds": "retrieval_seconds",
    "generation_seconds": "generation_seconds",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def index_condition(
    rows: list[dict[str, object]], condition: str
) -> dict[tuple[int, str], dict[str, object]]:
    indexed: dict[tuple[int, str], dict[str, object]] = {}
    for row in rows:
        if row.get("condition") != condition:
            continue
        key = (int(row["hop"]), str(row["id"]))
        if key in indexed:
            raise ValueError(f"duplicate record for {condition}: {key}")
        indexed[key] = row
    return indexed


def mean_field(
    rows: dict[tuple[int, str], dict[str, object]],
    keys: list[tuple[int, str]],
    field: str,
) -> float:
    return statistics.fmean(float(rows[key][field]) for key in keys)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-results", type=Path, required=True)
    parser.add_argument("--ablation-results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=50_000)
    args = parser.parse_args()

    if args.bootstrap_iterations < 1000 or args.permutation_iterations < 1000:
        raise ValueError("statistical iteration counts must each be at least 1000")

    primary_path = args.primary_results.resolve()
    ablation_path = args.ablation_results.resolve()
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    if primary.get("status") != "PASS" or ablation.get("status") != "PASS":
        raise ValueError("both experiment inputs must have status PASS")

    primary_config = primary.get("configuration", {})
    ablation_config = ablation.get("configuration", {})
    fixed_fields = ("split", "sample_per_hop", "seed", "budget", "max_paths")
    mismatches = {
        field: (primary_config.get(field), ablation_config.get(field))
        for field in fixed_fields
        if primary_config.get(field) != ablation_config.get(field)
    }
    if mismatches:
        raise ValueError(f"configuration mismatch: {mismatches}")
    if primary.get("model", {}).get("id") != ablation.get("model", {}).get("id"):
        raise ValueError("model identifiers do not match")
    if primary.get("model", {}).get("revision") != ablation.get("model", {}).get(
        "revision"
    ):
        raise ValueError("model revisions do not match")
    recorded_reference_hash = ablation_config.get("reference_results_sha256")
    actual_reference_hash = sha256(primary_path)
    if recorded_reference_hash != actual_reference_hash:
        raise ValueError("ablation reference-results hash does not match primary input")

    primary_rows = primary.get("results", [])
    ablation_rows = ablation.get("results", [])
    by_condition = {FULL: index_condition(primary_rows, FULL)}
    by_condition.update(
        {condition: index_condition(ablation_rows, condition) for condition in ABLATIONS}
    )
    reference_keys = set(by_condition[FULL])
    if len(reference_keys) != 900:
        raise ValueError(f"expected 900 frozen questions, found {len(reference_keys)}")
    for condition, indexed in by_condition.items():
        if set(indexed) != reference_keys:
            raise ValueError(f"question mismatch for {condition}")
    for key in reference_keys:
        reference = by_condition[FULL][key]
        for condition in ABLATIONS:
            candidate = by_condition[condition][key]
            for field in ("line_number", "question", "topic_entity", "gold_answers"):
                if candidate.get(field) != reference.get(field):
                    raise ValueError(f"metadata mismatch for {key}, {condition}, {field}")

    all_keys = sorted(reference_keys)
    scopes = {
        "1-hop": [key for key in all_keys if key[0] == 1],
        "2-hop": [key for key in all_keys if key[0] == 2],
        "3-hop": [key for key in all_keys if key[0] == 3],
        "all-hops": all_keys,
    }
    counts = {scope: len(keys) for scope, keys in scopes.items()}
    if counts != {"1-hop": 300, "2-hop": 300, "3-hop": 300, "all-hops": 900}:
        raise ValueError(f"unexpected question counts: {counts}")

    method_summary: dict[str, dict[str, dict[str, float]]] = {}
    for scope, keys in scopes.items():
        method_summary[scope] = {}
        for condition, indexed in by_condition.items():
            method_summary[scope][condition] = {
                metric: round(mean_field(indexed, keys, field), 9)
                for metric, field in METRICS.items()
            }
            method_summary[scope][condition]["output_cap_rate"] = round(
                statistics.fmean(
                    float(bool(indexed[key].get("output_hit_max_new_tokens")))
                    for key in keys
                ),
                9,
            )

    rng = np.random.default_rng(args.seed)
    comparisons: list[dict[str, object]] = []
    all_hops_raw_p: dict[str, float] = {}
    for scope, keys in scopes.items():
        hop_groups = sorted({key[0] for key in keys})
        for ablation_name in ABLATIONS:
            for metric, field in METRICS.items():
                full_values = np.asarray(
                    [float(by_condition[FULL][key][field]) for key in keys]
                )
                ablation_values = np.asarray(
                    [float(by_condition[ablation_name][key][field]) for key in keys]
                )
                differences = full_values - ablation_values
                grouped = [
                    np.asarray(
                        [
                            float(by_condition[FULL][key][field])
                            - float(by_condition[ablation_name][key][field])
                            for key in keys
                            if key[0] == hop
                        ]
                    )
                    for hop in hop_groups
                ]
                ci_low, ci_high = stratified_bootstrap_difference(
                    grouped, args.bootstrap_iterations, rng
                )
                p_value = None
                if metric == "raw_f1":
                    p_value = paired_permutation_p(
                        differences, args.permutation_iterations, rng
                    )
                row = {
                    "scope": scope,
                    "target": FULL,
                    "ablation": ablation_name,
                    "metric": metric,
                    "questions": len(keys),
                    "target_mean": round(float(full_values.mean()), 9),
                    "ablation_mean": round(float(ablation_values.mean()), 9),
                    "mean_difference": round(float(differences.mean()), 9),
                    "bootstrap_95_ci_low": round(ci_low, 9),
                    "bootstrap_95_ci_high": round(ci_high, 9),
                    "paired_permutation_p": (
                        round(float(p_value), 9) if p_value is not None else None
                    ),
                    "holm_adjusted_p": None,
                    "comparison_family": (
                        "mechanism_components"
                        if ablation_name in MECHANISM_ABLATIONS
                        else "relation_label_sensitivity"
                    ),
                    "target_wins": int(np.count_nonzero(differences > 1e-12)),
                    "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
                    "target_losses": int(np.count_nonzero(differences < -1e-12)),
                }
                comparisons.append(row)
                if scope == "all-hops" and metric == "raw_f1":
                    all_hops_raw_p[ablation_name] = float(p_value)

    adjusted = {}
    for family in (MECHANISM_ABLATIONS, LABEL_SENSITIVITY):
        adjusted.update(
            holm_adjust({name: all_hops_raw_p[name] for name in family})
        )
    for row in comparisons:
        if row["scope"] == "all-hops" and row["metric"] == "raw_f1":
            row["holm_adjusted_p"] = round(adjusted[str(row["ablation"])], 9)

    complete_case_sensitivity: list[dict[str, object]] = []
    for ablation_name in ABLATIONS:
        keys = [
            key
            for key in all_keys
            if not bool(
                by_condition[FULL][key].get("output_hit_max_new_tokens")
                or by_condition[ablation_name][key].get("output_hit_max_new_tokens")
            )
        ]
        differences = np.asarray(
            [
                float(by_condition[FULL][key]["f1"])
                - float(by_condition[ablation_name][key]["f1"])
                for key in keys
            ]
        )
        grouped = [
            np.asarray(
                [
                    float(by_condition[FULL][key]["f1"])
                    - float(by_condition[ablation_name][key]["f1"])
                    for key in keys
                    if key[0] == hop
                ]
            )
            for hop in (1, 2, 3)
        ]
        if any(len(values) == 0 for values in grouped):
            raise ValueError(f"empty complete-case hop group for {ablation_name}")
        ci_low, ci_high = stratified_bootstrap_difference(
            grouped, args.bootstrap_iterations, rng
        )
        complete_case_sensitivity.append(
            {
                "target": FULL,
                "ablation": ablation_name,
                "metric": "raw_f1",
                "included_questions": len(keys),
                "excluded_questions": len(all_keys) - len(keys),
                "mean_difference": round(float(differences.mean()), 9),
                "bootstrap_95_ci_low": round(ci_low, 9),
                "bootstrap_95_ci_high": round(ci_high, 9),
                "paired_permutation_p": round(
                    paired_permutation_p(
                        differences, args.permutation_iterations, rng
                    ),
                    9,
                ),
                "interpretation": "post-hoc output-cap complete-case sensitivity",
            }
        )

    payload = {
        "status": "PASS",
        "analysis_status": "post_hoc_exploratory",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "primary": str(primary_path),
            "primary_sha256": actual_reference_hash,
            "ablation": str(ablation_path),
            "ablation_sha256": sha256(ablation_path),
        },
        "configuration": {
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap": "paired percentile; stratified by hop for all-hops",
            "permutation_iterations": args.permutation_iterations,
            "permutation_test": "two-sided paired sign randomization; exploratory",
            "multiple_comparison": (
                "Two prespecified exploratory families: Holm across the three "
                "all-hops raw-F1 mechanism-component comparisons, and Holm "
                "across the two all-hops relation-label sensitivity comparisons"
            ),
        },
        "question_counts": counts,
        "method_summary": method_summary,
        "comparisons": comparisons,
        "output_cap_complete_case_sensitivity": complete_case_sensitivity,
    }
    output_json = args.output_json.resolve()
    output_csv = args.output_csv.resolve()
    atomic_json(output_json, payload)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(comparisons)

    print("status=PASS", flush=True)
    print(f"questions={counts}", flush=True)
    for ablation_name in ABLATIONS:
        row = next(
            item
            for item in comparisons
            if item["scope"] == "all-hops"
            and item["metric"] == "raw_f1"
            and item["ablation"] == ablation_name
        )
        print(
            f"full_vs={ablation_name} diff={row['mean_difference']} "
            f"ci=[{row['bootstrap_95_ci_low']},{row['bootstrap_95_ci_high']}] "
            f"exploratory_holm_p={row['holm_adjusted_p']}",
            flush=True,
        )
    print(f"json={output_json}", flush=True)
    print(f"csv={output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
