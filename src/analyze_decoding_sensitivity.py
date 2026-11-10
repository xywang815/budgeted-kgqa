#!/usr/bin/env python3
"""Paired analysis for the frozen greedy-versus-sampling sensitivity study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CONDITIONS = ("bm25", "ra_mmr", "oracle")
SEEDS = (20260818, 20260819, 20260820)
METRICS = ("f1", "candidate_constrained_f1", "hits_at_1", "exact_set_match")


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


def paired_row(name, scope, contrast, differences, hop_labels, iterations, rng):
    groups = [differences[hop_labels == hop] for hop in sorted(set(hop_labels))]
    low, high = bootstrap_ci(groups, iterations[0], rng)
    return {
        "decoder": name,
        "scope": scope,
        "contrast": contrast,
        "metric": "raw_f1",
        "questions": len(differences),
        "mean_difference": round(float(differences.mean()), 9),
        "bootstrap_95_ci_low": round(low, 9),
        "bootstrap_95_ci_high": round(high, 9),
        "paired_permutation_p": round(
            permutation_p(differences, iterations[1], rng), 9
        ),
        "target_wins": int(np.count_nonzero(differences > 1e-12)),
        "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
        "target_losses": int(np.count_nonzero(differences < -1e-12)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=50_000)
    args = parser.parse_args()

    if args.output_json.resolve().exists() or args.output_csv.resolve().exists():
        raise FileExistsError("Refusing to overwrite decoding-analysis output")
    input_path = args.results.resolve()
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if data.get("status") != "PASS":
        raise ValueError("Decoding-sensitivity input must have status PASS")
    config = data.get("configuration", {})
    if tuple(config.get("stochastic_seeds", ())) != SEEDS:
        raise ValueError("Sampling seeds do not match the frozen protocol")
    rows = data["results"]
    expected = int(config["sample_per_hop"]) * 3 * len(CONDITIONS) * (1 + len(SEEDS))
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(rows)}")

    indexed: dict[tuple[int, str, str, str, int | None], dict[str, object]] = {}
    for row in rows:
        seed = row.get("sampling_seed")
        key = (
            int(row["hop"]),
            str(row["id"]),
            str(row["condition"]),
            str(row["decoding"]),
            int(seed) if seed is not None else None,
        )
        if key in indexed:
            raise ValueError(f"Duplicate row {key}")
        indexed[key] = row
    question_keys = sorted({(int(row["hop"]), str(row["id"])) for row in rows})
    if len(question_keys) != int(config["sample_per_hop"]) * 3:
        raise ValueError("Unique question count mismatch")
    for hop, identifier in question_keys:
        for condition in CONDITIONS:
            if (hop, identifier, condition, "greedy", None) not in indexed:
                raise ValueError("Missing greedy row")
            for seed in SEEDS:
                if (
                    hop,
                    identifier,
                    condition,
                    "temperature_0.7_top_p_0.9",
                    seed,
                ) not in indexed:
                    raise ValueError("Missing stochastic row")

    scopes = {
        "1-hop": [key for key in question_keys if key[0] == 1],
        "2-hop": [key for key in question_keys if key[0] == 2],
        "3-hop": [key for key in question_keys if key[0] == 3],
        "all-hops": question_keys,
    }
    run_summary: list[dict[str, object]] = []
    stability: list[dict[str, object]] = []
    for scope, scope_keys in scopes.items():
        for condition in CONDITIONS:
            decoder_runs = [("greedy", None)] + [
                ("temperature_0.7_top_p_0.9", seed) for seed in SEEDS
            ]
            for decoder, seed in decoder_runs:
                subset = [indexed[(hop, identifier, condition, decoder, seed)] for hop, identifier in scope_keys]
                item: dict[str, object] = {
                    "scope": scope,
                    "condition": condition,
                    "decoder": decoder,
                    "sampling_seed": seed,
                    "questions": len(subset),
                }
                for metric in METRICS:
                    item[metric] = round(
                        statistics.fmean(float(row[metric]) for row in subset), 9
                    )
                item["output_cap_rate"] = round(
                    statistics.fmean(bool(row["output_hit_max_new_tokens"]) for row in subset), 9
                )
                item["generation_seconds"] = round(
                    statistics.fmean(float(row["generation_seconds"]) for row in subset), 9
                )
                run_summary.append(item)

            seed_macro_f1 = [
                statistics.fmean(
                    float(
                        indexed[
                            (
                                hop,
                                identifier,
                                condition,
                                "temperature_0.7_top_p_0.9",
                                seed,
                            )
                        ]["f1"]
                    )
                    for hop, identifier in scope_keys
                )
                for seed in SEEDS
            ]
            per_question_sd = []
            exact_prediction_agreement = []
            for hop, identifier in scope_keys:
                sample_rows = [
                    indexed[
                        (
                            hop,
                            identifier,
                            condition,
                            "temperature_0.7_top_p_0.9",
                            seed,
                        )
                    ]
                    for seed in SEEDS
                ]
                per_question_sd.append(
                    statistics.pstdev(float(row["f1"]) for row in sample_rows)
                )
                predictions = [
                    tuple(str(value) for value in row["parsed_predictions"])
                    for row in sample_rows
                ]
                exact_prediction_agreement.append(len(set(predictions)) == 1)
            stability.append(
                {
                    "scope": scope,
                    "condition": condition,
                    "sampling_macro_f1_mean": round(statistics.fmean(seed_macro_f1), 9),
                    "sampling_macro_f1_sd_across_seeds": round(
                        statistics.pstdev(seed_macro_f1), 9
                    ),
                    "mean_per_question_f1_sd": round(statistics.fmean(per_question_sd), 9),
                    "exact_prediction_agreement_rate": round(
                        statistics.fmean(exact_prediction_agreement), 9
                    ),
                }
            )

    rng = np.random.default_rng(args.seed)
    iterations = (args.bootstrap_iterations, args.permutation_iterations)
    comparisons: list[dict[str, object]] = []
    for scope, scope_keys in scopes.items():
        hop_labels = np.asarray([hop for hop, _ in scope_keys], dtype=np.int8)
        greedy_differences = np.asarray(
            [
                float(indexed[(hop, identifier, "ra_mmr", "greedy", None)]["f1"])
                - float(indexed[(hop, identifier, "bm25", "greedy", None)]["f1"])
                for hop, identifier in scope_keys
            ],
            dtype=np.float64,
        )
        comparisons.append(
            paired_row(
                "greedy",
                scope,
                "ra_mmr_minus_bm25",
                greedy_differences,
                hop_labels,
                iterations,
                rng,
            )
        )
        for seed in SEEDS:
            differences = np.asarray(
                [
                    float(
                        indexed[
                            (
                                hop,
                                identifier,
                                "ra_mmr",
                                "temperature_0.7_top_p_0.9",
                                seed,
                            )
                        ]["f1"]
                    )
                    - float(
                        indexed[
                            (
                                hop,
                                identifier,
                                "bm25",
                                "temperature_0.7_top_p_0.9",
                                seed,
                            )
                        ]["f1"]
                    )
                    for hop, identifier in scope_keys
                ],
                dtype=np.float64,
            )
            comparisons.append(
                paired_row(
                    f"sampling_seed_{seed}",
                    scope,
                    "ra_mmr_minus_bm25",
                    differences,
                    hop_labels,
                    iterations,
                    rng,
                )
            )

        averaged_differences = np.asarray(
            [
                statistics.fmean(
                    float(
                        indexed[
                            (
                                hop,
                                identifier,
                                "ra_mmr",
                                "temperature_0.7_top_p_0.9",
                                seed,
                            )
                        ]["f1"]
                    )
                    - float(
                        indexed[
                            (
                                hop,
                                identifier,
                                "bm25",
                                "temperature_0.7_top_p_0.9",
                                seed,
                            )
                        ]["f1"]
                    )
                    for seed in SEEDS
                )
                for hop, identifier in scope_keys
            ],
            dtype=np.float64,
        )
        comparisons.append(
            paired_row(
                "sampling_mean_across_3_seeds",
                scope,
                "ra_mmr_minus_bm25",
                averaged_differences,
                hop_labels,
                iterations,
                rng,
            )
        )

        for condition in CONDITIONS:
            differences = np.asarray(
                [
                    statistics.fmean(
                        float(
                            indexed[
                                (
                                    hop,
                                    identifier,
                                    condition,
                                    "temperature_0.7_top_p_0.9",
                                    seed,
                                )
                            ]["f1"]
                        )
                        for seed in SEEDS
                    )
                    - float(indexed[(hop, identifier, condition, "greedy", None)]["f1"])
                    for hop, identifier in scope_keys
                ],
                dtype=np.float64,
            )
            comparisons.append(
                paired_row(
                    "sampling_mean_minus_greedy",
                    scope,
                    f"sampling_minus_greedy/{condition}",
                    differences,
                    hop_labels,
                    iterations,
                    rng,
                )
            )

    output = {
        "status": "PASS",
        "analysis_status": "prospectively_frozen_internal_protocol_20260821",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(input_path), "sha256": sha256(input_path)},
        "configuration": {
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap": "paired percentile; stratified by hop for all-hops",
            "permutation_iterations": args.permutation_iterations,
            "permutation": "two-sided paired sign randomization",
            "sampling_seeds": SEEDS,
            "sampling_runs_are_not_pooled_with_main_greedy_estimates": True,
        },
        "question_counts": {scope: len(scope_keys) for scope, scope_keys in scopes.items()},
        "run_summary": run_summary,
        "stability": stability,
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
        if row["scope"] == "all-hops" and row["contrast"] == "ra_mmr_minus_bm25":
            print(
                f"decoder={row['decoder']} diff={row['mean_difference']} "
                f"ci=[{row['bootstrap_95_ci_low']},{row['bootstrap_95_ci_high']}] "
                f"p={row['paired_permutation_p']}"
            )
    print(f"json={args.output_json.resolve()}")
    print(f"csv={args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
