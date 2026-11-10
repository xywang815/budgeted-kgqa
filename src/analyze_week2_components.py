#!/usr/bin/env python3
"""Paired analysis for the prospectively frozen week-2 component study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
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
    "evidence_tokens": "evidence_tokens",
    "selected_paths": "selected_paths",
    "retrieval_seconds": "retrieval_seconds",
    "generation_seconds": "generation_seconds",
}
CONTRASTS = (
    ("cost_added", "relation_plus_cost", "relation_only"),
    ("diversity_added", "ra_mmr", "relation_plus_cost"),
    ("ra_vs_standard_mmr_no_cost", "ra_mmr", "standard_mmr_no_cost"),
    ("ra_vs_shortest_first", "ra_mmr", "shortest_first"),
    ("ra_vs_unit_endpoint", "ra_mmr", "unit_endpoint"),
    ("ra_vs_unique_relation_chain", "ra_mmr", "unique_relation_chain"),
)
PRIMARY_FAMILY = {
    "cost_added",
    "diversity_added",
    "ra_vs_standard_mmr_no_cost",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (len(ordered) - rank) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def permutation_p(values: np.ndarray, iterations: int, rng) -> float:
    observed = abs(float(values.mean()))
    if np.allclose(values, 0.0):
        return 1.0
    exceed = 0
    completed = 0
    while completed < iterations:
        count = min(1000, iterations - completed)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8)
        signs = signs * 2 - 1
        exceed += int(
            np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15)
        )
        completed += count
    return (exceed + 1.0) / (iterations + 1.0)


def bootstrap_ci(groups: list[np.ndarray], iterations: int, rng):
    total = sum(len(group) for group in groups)
    values = np.zeros(iterations, dtype=np.float64)
    for group in groups:
        indices = rng.integers(0, len(group), size=(iterations, len(group)))
        values += group[indices].mean(axis=1) * (len(group) / total)
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def index_rows(rows, source_condition: str):
    indexed = {}
    for row in rows:
        if row.get("condition") != source_condition:
            continue
        key = (int(row["hop"]), str(row["id"]))
        if key in indexed:
            raise ValueError(f"Duplicate {source_condition} row: {key}")
        indexed[key] = row
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-results", type=Path, required=True)
    parser.add_argument("--legacy-ablation-results", type=Path, required=True)
    parser.add_argument("--week2-results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=50_000)
    args = parser.parse_args()

    if args.output_json.resolve().exists() or args.output_csv.resolve().exists():
        raise FileExistsError("Refusing to overwrite component-analysis output")
    paths = {
        "primary": args.primary_results.resolve(),
        "legacy": args.legacy_ablation_results.resolve(),
        "week2": args.week2_results.resolve(),
    }
    payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    if any(payload.get("status") != "PASS" for payload in payloads.values()):
        raise ValueError("All component-analysis inputs must have status PASS")
    primary_sha = sha256(paths["primary"])
    if payloads["legacy"].get("configuration", {}).get("reference_results_sha256") != primary_sha:
        raise ValueError("Legacy ablation reference SHA mismatch")
    if payloads["week2"].get("configuration", {}).get("reference_results_sha256") != primary_sha:
        raise ValueError("Week-2 component reference SHA mismatch")

    primary_config = payloads["primary"].get("configuration", {})
    for name in ("legacy", "week2"):
        config = payloads[name].get("configuration", {})
        for key in ("split", "sample_per_hop", "seed", "budget", "max_paths"):
            if config.get(key) != primary_config.get(key):
                raise ValueError(f"{name} configuration mismatch for {key}")
        if payloads[name].get("model", {}).get("id") != payloads["primary"].get("model", {}).get("id"):
            raise ValueError(f"{name} model ID mismatch")
        if payloads[name].get("model", {}).get("revision") != payloads["primary"].get("model", {}).get("revision"):
            raise ValueError(f"{name} model revision mismatch")

    methods = {
        "ra_mmr": index_rows(payloads["primary"]["results"], "ra_mmr"),
        "relation_plus_cost": index_rows(payloads["legacy"]["results"], "ra_no_diversity"),
    }
    for condition in (
        "relation_only",
        "standard_mmr_no_cost",
        "shortest_first",
        "unit_endpoint",
        "unique_relation_chain",
    ):
        methods[condition] = index_rows(payloads["week2"]["results"], condition)

    reference_keys = set(methods["ra_mmr"])
    if len(reference_keys) != 900:
        raise ValueError("Expected 900 reference questions")
    for method, indexed in methods.items():
        if set(indexed) != reference_keys:
            raise ValueError(f"Question-set mismatch for {method}")
    for key in reference_keys:
        reference = methods["ra_mmr"][key]
        for method, indexed in methods.items():
            candidate = indexed[key]
            for field in ("line_number", "question", "topic_entity", "gold_answers"):
                if candidate.get(field) != reference.get(field):
                    raise ValueError(f"Metadata mismatch {key}/{method}/{field}")

    all_keys = sorted(reference_keys)
    scopes = {
        "1-hop": [key for key in all_keys if key[0] == 1],
        "2-hop": [key for key in all_keys if key[0] == 2],
        "3-hop": [key for key in all_keys if key[0] == 3],
        "all-hops": all_keys,
    }
    method_summary = {}
    for scope, keys in scopes.items():
        method_summary[scope] = {}
        for method, indexed in methods.items():
            method_summary[scope][method] = {
                metric: round(
                    statistics.fmean(float(indexed[key][field]) for key in keys), 9
                )
                for metric, field in METRICS.items()
            }
            method_summary[scope][method]["output_cap_rate"] = round(
                statistics.fmean(
                    bool(indexed[key].get("output_hit_max_new_tokens")) for key in keys
                ),
                9,
            )

    rng = np.random.default_rng(args.seed)
    comparisons = []
    primary_raw_p = {}
    for scope, keys in scopes.items():
        for contrast, target, baseline in CONTRASTS:
            for metric, field in METRICS.items():
                differences = np.asarray(
                    [float(methods[target][key][field]) - float(methods[baseline][key][field]) for key in keys]
                )
                groups = [
                    np.asarray([float(methods[target][key][field]) - float(methods[baseline][key][field]) for key in keys if key[0] == hop])
                    for hop in sorted({key[0] for key in keys})
                ]
                ci_low, ci_high = bootstrap_ci(groups, args.bootstrap_iterations, rng)
                p_value = permutation_p(differences, args.permutation_iterations, rng) if metric == "raw_f1" else None
                row = {
                    "scope": scope,
                    "contrast": contrast,
                    "target": target,
                    "baseline": baseline,
                    "metric": metric,
                    "questions": len(keys),
                    "target_mean": round(statistics.fmean(float(methods[target][key][field]) for key in keys), 9),
                    "baseline_mean": round(statistics.fmean(float(methods[baseline][key][field]) for key in keys), 9),
                    "mean_difference": round(float(differences.mean()), 9),
                    "bootstrap_95_ci_low": round(ci_low, 9),
                    "bootstrap_95_ci_high": round(ci_high, 9),
                    "paired_permutation_p": round(float(p_value), 9) if p_value is not None else None,
                    "holm_adjusted_p": None,
                    "target_wins": int(np.count_nonzero(differences > 1e-12)),
                    "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
                    "target_losses": int(np.count_nonzero(differences < -1e-12)),
                    "primary_family": contrast in PRIMARY_FAMILY and scope == "all-hops" and metric == "raw_f1",
                }
                comparisons.append(row)
                if row["primary_family"]:
                    primary_raw_p[contrast] = float(p_value)

    adjusted = holm(primary_raw_p)
    for row in comparisons:
        if row["primary_family"]:
            row["holm_adjusted_p"] = round(adjusted[str(row["contrast"])], 9)

    output = {
        "status": "PASS",
        "analysis_status": "prospectively_frozen_internal_protocol_20260821",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "configuration": {
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap": "paired percentile; stratified by hop for all-hops",
            "permutation_iterations": args.permutation_iterations,
            "permutation": "two-sided paired sign randomization",
            "primary_holm_family": sorted(PRIMARY_FAMILY),
        },
        "question_counts": {scope: len(keys) for scope, keys in scopes.items()},
        "method_summary": method_summary,
        "comparisons": comparisons,
    }
    atomic_json(args.output_json.resolve(), output)
    args.output_csv.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.resolve().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print("status=PASS")
    for row in comparisons:
        if row["primary_family"]:
            print(
                f"primary={row['contrast']} diff={row['mean_difference']} "
                f"ci=[{row['bootstrap_95_ci_low']},{row['bootstrap_95_ci_high']}] "
                f"holm_p={row['holm_adjusted_p']}"
            )
    print(f"json={args.output_json.resolve()}")
    print(f"csv={args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
