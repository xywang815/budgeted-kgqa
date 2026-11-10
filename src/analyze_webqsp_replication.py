#!/usr/bin/env python3
"""Paired statistics for WebQSP, stratified by official parse-chain length."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CONDITIONS = ("direct", "bfs", "bm25", "ra_mmr", "oracle")
METRICS = {
    "macro_f1": "f1",
    "answer_recall": "recall",
    "hits_at_1": "hits_at_1",
    "exact_set_match": "exact_set_match",
    "candidate_answer_recall": "candidate_answer_recall",
    "evidence_answer_recall": "evidence_answer_recall",
    "evidence_tokens": "evidence_tokens",
    "total_input_tokens": "total_input_tokens",
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


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    adjusted = {}
    running = 0.0
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * values[name]))
        adjusted[name] = running
    return adjusted


def paired_permutation_p(
    differences: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
    chunk_size: int = 1000,
) -> float:
    if np.allclose(differences, 0.0):
        return 1.0
    observed = abs(float(differences.mean()))
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


def bootstrap_ci(
    differences: np.ndarray, iterations: int, rng: np.random.Generator
) -> tuple[float, float]:
    indices = rng.integers(0, len(differences), size=(iterations, len(differences)))
    samples = differences[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=50_000)
    args = parser.parse_args()

    source_path = args.input.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "PASS":
        raise ValueError("input experiment is not PASS")
    rows = source.get("results", [])
    configuration = source.get("configuration", {})
    if configuration.get("path_direction") != "outgoing":
        raise ValueError("WebQSP analysis expects the preregistered outgoing-only graph")
    if int(configuration.get("rerank_cap", 0)) != 20_000:
        raise ValueError("WebQSP analysis expects the preregistered 20k rerank cap")
    expected_questions = int(configuration.get("sample_size", 0))
    if expected_questions < 1 or len(rows) != expected_questions * len(CONDITIONS):
        raise ValueError("unexpected result count")

    by_condition: dict[str, dict[str, dict[str, object]]] = {
        condition: {} for condition in CONDITIONS
    }
    for row in rows:
        condition = str(row["condition"])
        identifier = str(row["id"])
        if condition not in by_condition or identifier in by_condition[condition]:
            raise ValueError(f"duplicate or unknown condition: {condition} {identifier}")
        if int(row["evidence_tokens"]) > int(configuration["budget"]):
            raise ValueError(f"budget violation: {identifier} {condition}")
        by_condition[condition][identifier] = row
    reference_ids = set(by_condition["ra_mmr"])
    if len(reference_ids) != expected_questions:
        raise ValueError("unexpected unique question count")
    for condition in CONDITIONS:
        if set(by_condition[condition]) != reference_ids:
            raise ValueError(f"question mismatch: {condition}")
    if list(configuration.get("sample_ids", [])) != list(source["configuration"]["sample_ids"]):
        raise ValueError("sample identifier metadata mismatch")

    # `annotated_max_hops` is written by parse_question() from the maximum length
    # of the official WebQSP InferentialChain across valid parses.  It is not a
    # graph-resource shortest-path estimate.
    scopes: dict[str, list[str]] = {
        "official-1-chain": sorted(
            identifier
            for identifier in reference_ids
            if int(by_condition["ra_mmr"][identifier]["annotated_max_hops"]) == 1
        ),
        "official-2+-chain": sorted(
            identifier
            for identifier in reference_ids
            if int(by_condition["ra_mmr"][identifier]["annotated_max_hops"]) >= 2
        ),
        "all": sorted(reference_ids),
    }
    if not scopes["official-1-chain"] or not scopes["official-2+-chain"]:
        raise ValueError("empty official chain-length stratum")

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for scope, identifiers in scopes.items():
        summary[scope] = {}
        for condition in CONDITIONS:
            summary[scope][condition] = {
                metric: round(
                    statistics.fmean(
                        float(by_condition[condition][identifier][field])
                        for identifier in identifiers
                    ),
                    9,
                )
                for metric, field in METRICS.items()
            }

    rng = np.random.default_rng(args.seed)
    comparisons = []
    primary_p_values = {}
    for scope, identifiers in scopes.items():
        for baseline in ("bfs", "bm25"):
            target = np.asarray(
                [float(by_condition["ra_mmr"][identifier]["f1"]) for identifier in identifiers]
            )
            control = np.asarray(
                [float(by_condition[baseline][identifier]["f1"]) for identifier in identifiers]
            )
            differences = target - control
            low, high = bootstrap_ci(differences, args.bootstrap_iterations, rng)
            p_value = paired_permutation_p(
                differences, args.permutation_iterations, rng
            )
            comparison = {
                "scope": scope,
                "target": "ra_mmr",
                "baseline": baseline,
                "metric": "macro_f1",
                "questions": len(identifiers),
                "target_mean": round(float(target.mean()), 9),
                "baseline_mean": round(float(control.mean()), 9),
                "mean_difference": round(float(differences.mean()), 9),
                "bootstrap_95_ci_low": round(low, 9),
                "bootstrap_95_ci_high": round(high, 9),
                "paired_permutation_p": round(p_value, 9),
                "holm_adjusted_p": None,
                "wins": int(np.count_nonzero(differences > 1e-12)),
                "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
                "losses": int(np.count_nonzero(differences < -1e-12)),
            }
            comparisons.append(comparison)
            if scope == "all":
                primary_p_values[baseline] = p_value
    adjusted = holm_adjust(primary_p_values)
    for row in comparisons:
        if row["scope"] == "all":
            row["holm_adjusted_p"] = round(adjusted[str(row["baseline"])], 9)

    candidate_recall = statistics.fmean(
        float(by_condition["ra_mmr"][identifier]["candidate_answer_recall"])
        for identifier in scopes["all"]
    )
    truncated_rate = statistics.fmean(
        float(bool(by_condition["ra_mmr"][identifier]["candidate_truncated"]))
        for identifier in scopes["all"]
    )
    reader_gap = {
        scope: round(
            summary[scope]["oracle"]["evidence_answer_recall"]
            - summary[scope]["oracle"]["macro_f1"],
            9,
        )
        for scope in scopes
    }
    payload = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(source_path),
        "input_sha256": sha256(source_path),
        "configuration": {
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "permutation_iterations": args.permutation_iterations,
            "primary_metric": "macro answer-set F1",
            "multiple_comparison": "Holm across all-scope RA-MMR vs BFS and BM25",
            "depth_definition": (
                "maximum official WebQSP InferentialChain length across valid parses; "
                "strata are exactly 1 versus at least 2"
            ),
            "eligibility": (
                "valid official parse with topic, non-empty InferentialChain, and answers; "
                "local subgraph exists; at least one official topic MID occurs in that subgraph"
            ),
            "graph_direction": "outgoing-only",
            "rerank_cap": 20_000,
        },
        "integrity": {
            "questions": expected_questions,
            "records": len(rows),
            "conditions": list(CONDITIONS),
            "budget": configuration["budget"],
            "candidate_answer_recall": round(candidate_recall, 9),
            "candidate_truncation_rate": round(truncated_rate, 9),
        },
        "question_counts": {scope: len(values) for scope, values in scopes.items()},
        "method_summary": summary,
        "comparisons": comparisons,
        "oracle_reader_gap_evidence_recall_minus_f1": reader_gap,
    }
    atomic_json(args.output_json.resolve(), payload)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(comparisons[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparisons)
    print("status=PASS")
    print(f"questions={payload['question_counts']}")
    for row in comparisons:
        print(
            f"comparison={row['scope']}/ra_mmr_vs_{row['baseline']} "
            f"diff={row['mean_difference']} ci=[{row['bootstrap_95_ci_low']},"
            f"{row['bootstrap_95_ci_high']}] p={row['paired_permutation_p']} "
            f"holm={row['holm_adjusted_p']}"
        )
    print(f"candidate_answer_recall={candidate_recall:.6f} truncated_rate={truncated_rate:.6f}")
    print(f"json={args.output_json.resolve()}")
    print(f"csv={args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
