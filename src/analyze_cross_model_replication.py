#!/usr/bin/env python3
"""Cross-model replication and output-limit sensitivity analysis.

This script does not alter or regenerate model outputs. It verifies two frozen
inference files and their paired-statistics reports, summarizes the within-model
RA-MMR effects, tests cross-model effect heterogeneity on the shared questions,
and repeats the primary comparisons after excluding output-limit pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BASELINES = ("bfs", "bm25", "ba_mmr")
REQUIRED = {"direct", "bfs", "bm25", "ba_mmr", "ra_mmr", "oracle"}


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


def stratified_bootstrap(
    differences: np.ndarray,
    hops: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    bootstrap = np.zeros(iterations, dtype=np.float64)
    total = len(differences)
    for hop in sorted(set(int(value) for value in hops)):
        values = differences[hops == hop]
        indices = rng.integers(0, len(values), size=(iterations, len(values)))
        bootstrap += values[indices].mean(axis=1) * (len(values) / total)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return float(low), float(high)


def effect_summary(
    differences: np.ndarray,
    hops: np.ndarray,
    iterations_bootstrap: int,
    iterations_permutation: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    if len(differences) == 0:
        raise ValueError("cannot summarize an empty paired sample")
    low, high = stratified_bootstrap(
        differences, hops, iterations_bootstrap, rng
    )
    return {
        "questions": int(len(differences)),
        "mean_difference": round(float(differences.mean()), 9),
        "bootstrap_95_ci_low": round(low, 9),
        "bootstrap_95_ci_high": round(high, 9),
        "paired_permutation_p": round(
            paired_permutation_p(differences, iterations_permutation, rng), 9
        ),
        "target_wins": int(np.count_nonzero(differences > 1e-12)),
        "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
        "target_losses": int(np.count_nonzero(differences < -1e-12)),
    }


def load_run(path: Path) -> tuple[dict[str, object], dict[tuple[int, str, str], dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError(f"run is not PASS: {path}")
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"run has no result rows: {path}")
    index: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in rows:
        key = (int(row["hop"]), str(row["id"]), str(row["condition"]))
        if key in index:
            raise ValueError(f"duplicate record {key} in {path}")
        index[key] = row
    conditions = {key[2] for key in index}
    if conditions != REQUIRED:
        raise ValueError(f"unexpected conditions in {path}: {conditions}")
    questions = {(key[0], key[1]) for key in index}
    if len(questions) * len(REQUIRED) != len(index):
        raise ValueError(f"incomplete condition grid in {path}")
    return payload, index


def load_stats(path: Path, run_path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError(f"statistics report is not PASS: {path}")
    if payload.get("input_sha256") != sha256(run_path):
        raise ValueError(f"statistics input hash mismatch: {path}")
    expected_counts = {"1-hop": 300, "2-hop": 300, "3-hop": 300, "all-hops": 900}
    if payload.get("question_counts") != expected_counts:
        raise ValueError(f"unexpected statistics question counts: {path}")
    for baseline in BASELINES:
        all_hops = primary_row(payload, "all-hops", baseline)
        weighted = statistics.fmean(
            float(primary_row(payload, f"{hop}-hop", baseline)["mean_difference"])
            for hop in (1, 2, 3)
        )
        if not math.isclose(
            float(all_hops["mean_difference"]), weighted, abs_tol=2e-9
        ):
            raise ValueError(
                f"all-hops versus hop-weighted effect mismatch for {baseline}: {path}"
            )
    return payload


def primary_row(stats: dict[str, object], scope: str, baseline: str) -> dict[str, object]:
    matches = [
        row
        for row in stats["comparisons"]
        if row["scope"] == scope
        and row["metric"] == "raw_f1"
        and row["target"] == "ra_mmr"
        and row["baseline"] == baseline
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one statistics row for {scope}/{baseline}")
    return matches[0]


def paired_values(
    index: dict[tuple[int, str, str], dict[str, object]],
    baseline: str,
    questions: list[tuple[int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    differences = np.asarray(
        [
            float(index[(hop, question_id, "ra_mmr")]["f1"])
            - float(index[(hop, question_id, baseline)]["f1"])
            for hop, question_id in questions
        ],
        dtype=np.float64,
    )
    hops = np.asarray([hop for hop, _ in questions], dtype=np.int16)
    return differences, hops


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-results", type=Path, required=True)
    parser.add_argument("--qwen-stats", type=Path, required=True)
    parser.add_argument("--phi-results", type=Path, required=True)
    parser.add_argument("--phi-stats", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=50_000)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1000 or args.permutation_iterations < 1000:
        raise ValueError("iteration counts must each be at least 1000")

    run_paths = {
        "qwen3_8b_awq": args.qwen_results.resolve(),
        "phi4_mini_3_8b_bf16": args.phi_results.resolve(),
    }
    stats_paths = {
        "qwen3_8b_awq": args.qwen_stats.resolve(),
        "phi4_mini_3_8b_bf16": args.phi_stats.resolve(),
    }
    runs: dict[str, dict[str, object]] = {}
    indices: dict[str, dict[tuple[int, str, str], dict[str, object]]] = {}
    stats: dict[str, dict[str, object]] = {}
    for model_name in run_paths:
        runs[model_name], indices[model_name] = load_run(run_paths[model_name])
        stats[model_name] = load_stats(stats_paths[model_name], run_paths[model_name])

    questions_by_model = {
        model_name: {(hop, question_id) for hop, question_id, _ in index}
        for model_name, index in indices.items()
    }
    if questions_by_model["qwen3_8b_awq"] != questions_by_model["phi4_mini_3_8b_bf16"]:
        raise ValueError("cross-model question identifiers do not match")
    questions = sorted(questions_by_model["qwen3_8b_awq"])
    if len(questions) != 900:
        raise ValueError(f"expected 900 shared questions, found {len(questions)}")

    rng = np.random.default_rng(args.seed)
    primary_replication: list[dict[str, object]] = []
    interaction_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []

    for baseline in BASELINES:
        qwen_row = primary_row(stats["qwen3_8b_awq"], "all-hops", baseline)
        phi_row = primary_row(stats["phi4_mini_3_8b_bf16"], "all-hops", baseline)
        primary_replication.append(
            {
                "comparison": f"ra_mmr_vs_{baseline}",
                "qwen_effect": qwen_row["mean_difference"],
                "qwen_ci": [qwen_row["bootstrap_95_ci_low"], qwen_row["bootstrap_95_ci_high"]],
                "qwen_holm_p": qwen_row["holm_adjusted_p"],
                "phi_effect": phi_row["mean_difference"],
                "phi_ci": [phi_row["bootstrap_95_ci_low"], phi_row["bootstrap_95_ci_high"]],
                "phi_holm_p": phi_row["holm_adjusted_p"],
                "same_positive_direction": bool(
                    float(qwen_row["mean_difference"]) > 0
                    and float(phi_row["mean_difference"]) > 0
                ),
                "significant_in_both_models": bool(
                    float(qwen_row["holm_adjusted_p"]) < 0.05
                    and float(phi_row["holm_adjusted_p"]) < 0.05
                ),
            }
        )

        qwen_diff, hops = paired_values(indices["qwen3_8b_awq"], baseline, questions)
        phi_diff, _ = paired_values(indices["phi4_mini_3_8b_bf16"], baseline, questions)
        interaction = effect_summary(
            phi_diff - qwen_diff,
            hops,
            args.bootstrap_iterations,
            args.permutation_iterations,
            rng,
        )
        interaction.update(
            {
                "scope": "all-hops",
                "comparison": f"phi_minus_qwen_of_ra_mmr_vs_{baseline}",
            }
        )
        interaction_rows.append(interaction)

    for hop in (1, 2, 3):
        scoped = [question for question in questions if question[0] == hop]
        qwen_diff, hops = paired_values(indices["qwen3_8b_awq"], "bm25", scoped)
        phi_diff, _ = paired_values(indices["phi4_mini_3_8b_bf16"], "bm25", scoped)
        interaction = effect_summary(
            phi_diff - qwen_diff,
            hops,
            args.bootstrap_iterations,
            args.permutation_iterations,
            rng,
        )
        interaction.update(
            {
                "scope": f"{hop}-hop",
                "comparison": "phi_minus_qwen_of_ra_mmr_vs_bm25",
                "qwen_effect": round(float(qwen_diff.mean()), 9),
                "phi_effect": round(float(phi_diff.mean()), 9),
            }
        )
        interaction_rows.append(interaction)

    for model_name, index in indices.items():
        any_limit_questions = {
            (hop, question_id)
            for (hop, question_id, _), row in index.items()
            if bool(row["output_hit_max_new_tokens"])
        }
        for baseline in BASELINES:
            pair_clean = [
                (hop, question_id)
                for hop, question_id in questions
                if not bool(index[(hop, question_id, "ra_mmr")]["output_hit_max_new_tokens"])
                and not bool(index[(hop, question_id, baseline)]["output_hit_max_new_tokens"])
            ]
            all_clean = [question for question in questions if question not in any_limit_questions]
            for subset_name, subset in (
                ("pairwise_output_complete", pair_clean),
                ("all_conditions_output_complete", all_clean),
            ):
                differences, hops = paired_values(index, baseline, subset)
                row = effect_summary(
                    differences,
                    hops,
                    args.bootstrap_iterations,
                    args.permutation_iterations,
                    rng,
                )
                row.update(
                    {
                        "model": model_name,
                        "scope": "all-hops",
                        "subset": subset_name,
                        "comparison": f"ra_mmr_vs_{baseline}",
                        "excluded_questions": len(questions) - len(subset),
                        "post_hoc_sensitivity": True,
                    }
                )
                sensitivity_rows.append(row)

    hop_replication = []
    for hop in (1, 2, 3):
        qwen = primary_row(stats["qwen3_8b_awq"], f"{hop}-hop", "bm25")
        phi = primary_row(stats["phi4_mini_3_8b_bf16"], f"{hop}-hop", "bm25")
        hop_replication.append(
            {
                "scope": f"{hop}-hop",
                "qwen_ra_minus_bm25": qwen["mean_difference"],
                "qwen_ci": [qwen["bootstrap_95_ci_low"], qwen["bootstrap_95_ci_high"]],
                "phi_ra_minus_bm25": phi["mean_difference"],
                "phi_ci": [phi["bootstrap_95_ci_low"], phi["bootstrap_95_ci_high"]],
            }
        )

    payload = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "permutation_iterations": args.permutation_iterations,
            "interaction_definition": "(RA-Baseline)_Phi minus (RA-Baseline)_Qwen on shared questions",
            "sensitivity_definition": "post-hoc exclusion of output-limit pairs; primary analyses remain unchanged",
        },
        "inputs": {
            name: {
                "results": str(run_paths[name]),
                "results_sha256": sha256(run_paths[name]),
                "stats": str(stats_paths[name]),
                "stats_sha256": sha256(stats_paths[name]),
                "model": runs[name].get("model"),
            }
            for name in run_paths
        },
        "shared_questions": len(questions),
        "primary_replication": primary_replication,
        "hop_replication_ra_vs_bm25": hop_replication,
        "cross_model_interactions": interaction_rows,
        "output_limit_sensitivity": sensitivity_rows,
    }
    atomic_json(args.output_json.resolve(), payload)

    csv_rows: list[dict[str, object]] = []
    for row in interaction_rows:
        csv_rows.append({"analysis": "interaction", **row})
    for row in sensitivity_rows:
        csv_rows.append({"analysis": "output_limit_sensitivity", **row})
    fieldnames = sorted({key for row in csv_rows for key in row})
    args.output_csv.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.resolve().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("status=PASS", flush=True)
    print(f"shared_questions={len(questions)}", flush=True)
    for row in primary_replication:
        print(
            f"replication={row['comparison']} "
            f"qwen_effect={float(row['qwen_effect']):.6f} "
            f"phi_effect={float(row['phi_effect']):.6f} "
            f"significant_both={row['significant_in_both_models']}",
            flush=True,
        )
    for row in hop_replication:
        print(
            f"hop={row['scope']} ra_vs_bm25 "
            f"qwen={float(row['qwen_ra_minus_bm25']):.6f} "
            f"phi={float(row['phi_ra_minus_bm25']):.6f}",
            flush=True,
        )
    for row in sensitivity_rows:
        if row["comparison"] == "ra_mmr_vs_bm25" and row["subset"] == "pairwise_output_complete":
            print(
                f"sensitivity={row['model']}/ra_vs_bm25 "
                f"n={row['questions']} excluded={row['excluded_questions']} "
                f"diff={float(row['mean_difference']):.6f} "
                f"ci=[{float(row['bootstrap_95_ci_low']):.6f},"
                f"{float(row['bootstrap_95_ci_high']):.6f}]",
                flush=True,
            )
    print(f"json={args.output_json.resolve()}", flush=True)
    print(f"csv={args.output_csv.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"status=FAIL error={error}", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
