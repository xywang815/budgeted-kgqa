#!/usr/bin/env python3
"""Template-cluster-aware paired analysis for frozen MetaQA test samples.

The inferential unit is a normalized question template rather than an
individual question.  This corrects the anti-conservative question-level
bootstrap used when templated questions are treated as independent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ENTITY_RE = re.compile(r"\[[^\]]+\]")
SPACE_RE = re.compile(r"\s+")
TARGET = "ra_mmr"
BASELINES = ("bfs", "bm25", "ba_mmr")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def question_template(question: str) -> str:
    """Lowercase, replace every bracketed topic entity, collapse whitespace."""
    return SPACE_RE.sub(" ", ENTITY_RE.sub("[ENTITY]", question).casefold()).strip()


def index_rows(rows: list[dict[str, object]], condition: str):
    indexed = {}
    for row in rows:
        if row.get("condition") != condition:
            continue
        key = (int(row["hop"]), str(row["id"]))
        if key in indexed:
            raise ValueError(f"duplicate {condition} row: {key}")
        indexed[key] = row
    return indexed


def cluster_bootstrap(
    clusters: list[np.ndarray], iterations: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Resample template clusters and concatenate their member questions."""
    draws = np.empty(iterations, dtype=np.float64)
    cluster_count = len(clusters)
    sums = np.asarray([values.sum() for values in clusters], dtype=np.float64)
    sizes = np.asarray([len(values) for values in clusters], dtype=np.float64)
    for start in range(0, iterations, 1000):
        count = min(1000, iterations - start)
        sampled = rng.integers(0, cluster_count, size=(count, cluster_count))
        draws[start : start + count] = sums[sampled].sum(axis=1) / sizes[
            sampled
        ].sum(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def cluster_sign_flip_p(
    clusters: list[np.ndarray], iterations: int, rng: np.random.Generator
) -> float:
    """Two-sided sign-flip test with one sign per normalized template."""
    sums = np.asarray([values.sum() for values in clusters], dtype=np.float64)
    observed = abs(float(sums.sum()))
    if observed == 0.0 and np.allclose(sums, 0.0):
        return 1.0
    exceedances = 0
    completed = 0
    while completed < iterations:
        count = min(1000, iterations - completed)
        signs = rng.integers(0, 2, size=(count, len(sums)), dtype=np.int8) * 2 - 1
        permuted = np.abs(signs @ sums)
        exceedances += int(np.count_nonzero(permuted >= observed - 1e-15))
        completed += count
    return (exceedances + 1.0) / (iterations + 1.0)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        value = min(1.0, (len(ordered) - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--permutation-iterations", type=int, default=100_000)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1000 or args.permutation_iterations < 1000:
        raise ValueError("iteration counts must be at least 1000")

    rng = np.random.default_rng(args.seed)
    outputs = []
    family_p_values: dict[str, float] = {}
    inputs = []

    for path in args.results:
        resolved = path.resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise ValueError(f"input is not PASS: {resolved}")
        configuration = payload.get("configuration", {})
        budget = int(configuration["budget"])
        if int(configuration.get("sample_per_hop", 0)) != 300:
            raise ValueError(f"expected 300 questions per hop: {resolved}")
        rows = payload.get("results", [])
        by_condition = {
            name: index_rows(rows, name) for name in (TARGET,) + BASELINES
        }
        keys = set(by_condition[TARGET])
        if len(keys) != 900:
            raise ValueError(f"expected 900 target questions: {resolved}")
        if any(set(indexed) != keys for indexed in by_condition.values()):
            raise ValueError(f"condition question mismatch: {resolved}")
        inputs.append(
            {
                "path": str(resolved),
                "sha256": sha256(resolved),
                "budget": budget,
                "seed": configuration.get("seed"),
            }
        )

        scopes = {
            "1-hop": sorted(key for key in keys if key[0] == 1),
            "2-hop": sorted(key for key in keys if key[0] == 2),
            "3-hop": sorted(key for key in keys if key[0] == 3),
            "all-hops": sorted(keys),
        }
        for scope, scope_keys in scopes.items():
            for baseline in BASELINES:
                grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
                for key in scope_keys:
                    target_row = by_condition[TARGET][key]
                    baseline_row = by_condition[baseline][key]
                    if target_row["question"] != baseline_row["question"]:
                        raise ValueError(f"question text mismatch: {key}")
                    cluster_key = (
                        key[0],
                        question_template(str(target_row["question"])),
                    )
                    grouped[cluster_key].append(
                        float(target_row["f1"]) - float(baseline_row["f1"])
                    )
                clusters = [np.asarray(values) for values in grouped.values()]
                all_values = np.concatenate(clusters)
                cluster_means = np.asarray([values.mean() for values in clusters])
                ci_low, ci_high = cluster_bootstrap(
                    clusters, args.bootstrap_iterations, rng
                )
                p_value = cluster_sign_flip_p(
                    clusters, args.permutation_iterations, rng
                )
                row = {
                    "budget": budget,
                    "scope": scope,
                    "target": TARGET,
                    "baseline": baseline,
                    "questions": len(all_values),
                    "template_clusters": len(clusters),
                    "largest_cluster": max(len(values) for values in clusters),
                    "mean_cluster_size": round(len(all_values) / len(clusters), 9),
                    "question_weighted_difference": round(float(all_values.mean()), 9),
                    "equal_template_difference": round(float(cluster_means.mean()), 9),
                    "cluster_bootstrap_95_ci_low": round(ci_low, 9),
                    "cluster_bootstrap_95_ci_high": round(ci_high, 9),
                    "cluster_sign_flip_p": round(p_value, 9),
                    "holm_adjusted_p": None,
                }
                outputs.append(row)
                if scope == "all-hops":
                    family_p_values[f"B{budget}:{baseline}"] = p_value

    adjusted = holm_adjust(family_p_values)
    for row in outputs:
        if row["scope"] == "all-hops":
            row["holm_adjusted_p"] = round(
                adjusted[f"B{row['budget']}:{row['baseline']}"], 9
            )

    report = {
        "status": "PASS",
        "analysis": {
            "template_definition": "lowercase + every bracketed entity -> [ENTITY] + whitespace collapse",
            "cluster_key": "(official hop, normalized question template)",
            "estimand": "question-weighted paired macro-F1 difference",
            "bootstrap": "resample template clusters with replacement and concatenate members",
            "randomization_test": "one Rademacher sign per template cluster",
            "multiplicity_family": "nine all-hop primary comparisons: 3 budgets x 3 baselines",
            "holm_adjustment": "single Holm family over the nine all-hop primary comparisons",
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "permutation_iterations": args.permutation_iterations,
        },
        "inputs": inputs,
        "comparisons": outputs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outputs[0]))
        writer.writeheader()
        writer.writerows(outputs)

    print("status=PASS")
    for row in outputs:
        if row["baseline"] == "bm25":
            print(
                f"B{row['budget']} {row['scope']} clusters={row['template_clusters']} "
                f"diff={row['question_weighted_difference']:.6f} "
                f"CI=[{row['cluster_bootstrap_95_ci_low']:.6f},"
                f"{row['cluster_bootstrap_95_ci_high']:.6f}] "
                f"p={row['cluster_sign_flip_p']:.6g}"
            )
    print(f"json={args.output_json.resolve()}")
    print(f"csv={args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
