#!/usr/bin/env python3
"""Paired analysis for frozen hop-count robustness experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CONDITIONS = ("bfs", "bm25", "ra_mmr")
METRICS = {
    "raw_f1": "f1",
    "constrained_f1": "candidate_constrained_f1",
    "hits_at_1": "hits_at_1",
    "evidence_recall": "evidence_answer_recall",
    "endpoint_f1": "selected_endpoint_f1",
    "evidence_tokens": "evidence_tokens",
    "input_tokens": "total_input_tokens",
    "candidate_paths": "candidate_paths",
    "retrieval_seconds": "retrieval_seconds",
    "generation_seconds": "generation_seconds",
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


def permutation_p(values: np.ndarray, iterations: int, rng) -> float:
    observed = abs(float(values.mean()))
    if np.allclose(values, 0.0):
        return 1.0
    exceed = 0
    completed = 0
    while completed < iterations:
        count = min(1000, iterations - completed)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        exceed += int(
            np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15)
        )
        completed += count
    return (exceed + 1.0) / (iterations + 1.0)


def bootstrap_ci(groups: list[np.ndarray], iterations: int, rng) -> tuple[float, float]:
    total = sum(len(group) for group in groups)
    values = np.zeros(iterations, dtype=np.float64)
    for group in groups:
        indices = rng.integers(0, len(group), size=(iterations, len(group)))
        values += group[indices].mean(axis=1) * (len(group) / total)
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (len(ordered) - rank) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--robustness-results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=50_000)
    args = parser.parse_args()

    if args.output_json.resolve().exists() or args.output_csv.resolve().exists():
        raise FileExistsError("Refusing to overwrite hop-robustness analysis output")
    reference_path = args.reference_results.resolve()
    robustness_path = args.robustness_results.resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    if reference.get("status") != "PASS" or robustness.get("status") != "PASS":
        raise ValueError("Both inputs must have status PASS")
    if robustness.get("configuration", {}).get("reference_results_sha256") != sha256(reference_path):
        raise ValueError("Frozen reference SHA mismatch")

    known: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in reference["results"]:
        condition = str(row["condition"])
        if condition in CONDITIONS:
            key = (int(row["hop"]), str(row["id"]), condition)
            if key in known:
                raise ValueError(f"Duplicate known-hop row {key}")
            known[key] = row
    rows: dict[tuple[int, str, str, str], dict[str, object]] = {}
    for row in robustness["results"]:
        key = (
            int(row["hop"]),
            str(row["id"]),
            str(row["scenario"]),
            str(row["condition"]),
        )
        if key in rows:
            raise ValueError(f"Duplicate robustness row {key}")
        rows[key] = row
    question_keys = sorted({(hop, identifier) for hop, identifier, _, _ in rows})
    if len(question_keys) != 900:
        raise ValueError(f"Expected 900 frozen questions, found {len(question_keys)}")
    for hop, identifier in question_keys:
        for condition in CONDITIONS:
            if (hop, identifier, condition) not in known:
                raise ValueError(f"Missing known-hop row {(hop, identifier, condition)}")

    scenarios_by_hop = {
        hop: sorted({scenario for row_hop, _, scenario, _ in rows if row_hop == hop})
        for hop in (1, 2, 3)
    }
    summary: list[dict[str, object]] = []
    for hop in (1, 2, 3):
        ids = [identifier for row_hop, identifier in question_keys if row_hop == hop]
        for scenario in ("known_hop", *scenarios_by_hop[hop]):
            for condition in CONDITIONS:
                subset = (
                    [known[(hop, identifier, condition)] for identifier in ids]
                    if scenario == "known_hop"
                    else [rows[(hop, identifier, scenario, condition)] for identifier in ids]
                )
                item: dict[str, object] = {
                    "hop": hop,
                    "scenario": scenario,
                    "condition": condition,
                    "questions": len(subset),
                }
                for label, field in METRICS.items():
                    if field in subset[0]:
                        item[label] = round(
                            statistics.fmean(float(row[field]) for row in subset), 9
                        )
                item["candidate_truncation_rate"] = round(
                    statistics.fmean(bool(row.get("candidate_truncated")) for row in subset),
                    9,
                )
                item["output_cap_rate"] = round(
                    statistics.fmean(bool(row.get("output_hit_max_new_tokens")) for row in subset),
                    9,
                )
                summary.append(item)

    rng = np.random.default_rng(args.seed)
    comparisons: list[dict[str, object]] = []
    primary_p: dict[str, float] = {}
    scopes: list[tuple[str, int | None, str]] = []
    for hop in (1, 2, 3):
        scopes.extend((f"{hop}-hop", hop, scenario) for scenario in scenarios_by_hop[hop])
    scopes.append(("all-hops", None, "unknown_hop_1to3"))
    for scope, hop_filter, scenario in scopes:
        keys = [
            (hop, identifier)
            for hop, identifier in question_keys
            if hop_filter is None or hop == hop_filter
        ]
        for baseline in ("bfs", "bm25"):
            for label, field in METRICS.items():
                differences = np.asarray(
                    [
                        float(rows[(hop, identifier, scenario, "ra_mmr")][field])
                        - float(rows[(hop, identifier, scenario, baseline)][field])
                        for hop, identifier in keys
                    ],
                    dtype=np.float64,
                )
                groups = [
                    differences[
                        np.asarray([key[0] == group_hop for key in keys], dtype=bool)
                    ]
                    for group_hop in sorted({key[0] for key in keys})
                ]
                ci_low, ci_high = bootstrap_ci(groups, args.bootstrap_iterations, rng)
                p_value = (
                    permutation_p(differences, args.permutation_iterations, rng)
                    if label == "raw_f1"
                    else None
                )
                primary = scope == "all-hops" and label == "raw_f1"
                name = f"ra_vs_{baseline}"
                comparison = {
                    "comparison_type": "method",
                    "scope": scope,
                    "scenario": scenario,
                    "contrast": name,
                    "metric": label,
                    "questions": len(keys),
                    "mean_difference": round(float(differences.mean()), 9),
                    "bootstrap_95_ci_low": round(ci_low, 9),
                    "bootstrap_95_ci_high": round(ci_high, 9),
                    "paired_permutation_p": round(float(p_value), 9) if p_value is not None else None,
                    "holm_adjusted_p": None,
                    "target_wins": int(np.count_nonzero(differences > 1e-12)),
                    "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
                    "target_losses": int(np.count_nonzero(differences < -1e-12)),
                    "primary_family": primary,
                }
                comparisons.append(comparison)
                if primary:
                    primary_p[name] = float(p_value)

    adjusted = holm(primary_p)
    for row in comparisons:
        if row["primary_family"]:
            row["holm_adjusted_p"] = round(adjusted[str(row["contrast"])], 9)

    for hop in (1, 2, 3):
        ids = [identifier for row_hop, identifier in question_keys if row_hop == hop]
        for scenario in scenarios_by_hop[hop]:
            for condition in CONDITIONS:
                differences = np.asarray(
                    [
                        float(rows[(hop, identifier, scenario, condition)]["f1"])
                        - float(known[(hop, identifier, condition)]["f1"])
                        for identifier in ids
                    ],
                    dtype=np.float64,
                )
                ci_low, ci_high = bootstrap_ci([differences], args.bootstrap_iterations, rng)
                comparisons.append(
                    {
                        "comparison_type": "scenario_vs_known",
                        "scope": f"{hop}-hop",
                        "scenario": scenario,
                        "contrast": f"{condition}_scenario_minus_known",
                        "metric": "raw_f1",
                        "questions": len(ids),
                        "mean_difference": round(float(differences.mean()), 9),
                        "bootstrap_95_ci_low": round(ci_low, 9),
                        "bootstrap_95_ci_high": round(ci_high, 9),
                        "paired_permutation_p": round(
                            permutation_p(differences, args.permutation_iterations, rng), 9
                        ),
                        "holm_adjusted_p": None,
                        "target_wins": int(np.count_nonzero(differences > 1e-12)),
                        "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
                        "target_losses": int(np.count_nonzero(differences < -1e-12)),
                        "primary_family": False,
                    }
                )

    output = {
        "status": "PASS",
        "analysis_status": "prospectively_frozen_internal_protocol_20260821",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
            "robustness": {"path": str(robustness_path), "sha256": sha256(robustness_path)},
        },
        "configuration": {
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap": "paired percentile; stratified by true hop for all-hops",
            "permutation_iterations": args.permutation_iterations,
            "permutation": "two-sided paired sign randomization",
            "primary_holm_family": sorted(primary_p),
        },
        "scenarios_by_hop": scenarios_by_hop,
        "summary": summary,
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
