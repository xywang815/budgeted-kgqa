#!/usr/bin/env python3
"""Paired reader-specific and shared-evidence analysis for two frozen readers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CONDITIONS = ("direct", "bfs", "bm25", "ba_mmr", "ra_mmr", "oracle")
METRICS = {
    "raw_f1": "f1",
    "constrained_f1": "candidate_constrained_f1",
    "hits_at_1": "hits_at_1",
    "evidence_recall": "evidence_answer_recall",
    "input_tokens": "total_input_tokens",
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


def index_rows(payload: dict[str, object]) -> dict[tuple[int, str, str], dict[str, object]]:
    indexed: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in payload["results"]:
        key = (int(row["hop"]), str(row["id"]), str(row["condition"]))
        if key in indexed:
            raise ValueError(f"Duplicate row {key}")
        indexed[key] = row
    return indexed


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


def comparison(
    *,
    contrast_type: str,
    view: str,
    scope: str,
    contrast: str,
    metric: str,
    differences: np.ndarray,
    hop_labels: np.ndarray,
    iterations: tuple[int, int],
    rng,
) -> dict[str, object]:
    groups = [differences[hop_labels == hop] for hop in sorted(set(hop_labels))]
    low, high = bootstrap_ci(groups, iterations[0], rng)
    p_value = permutation_p(differences, iterations[1], rng)
    return {
        "comparison_type": contrast_type,
        "view": view,
        "scope": scope,
        "contrast": contrast,
        "metric": metric,
        "questions": len(differences),
        "mean_difference": round(float(differences.mean()), 9),
        "bootstrap_95_ci_low": round(low, 9),
        "bootstrap_95_ci_high": round(high, 9),
        "paired_permutation_p": round(p_value, 9),
        "target_wins": int(np.count_nonzero(differences > 1e-12)),
        "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
        "target_losses": int(np.count_nonzero(differences < -1e-12)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-reader-specific", type=Path, required=True)
    parser.add_argument("--phi-reader-specific", type=Path, required=True)
    parser.add_argument("--phi-shared-evidence", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=50_000)
    args = parser.parse_args()

    if args.output_json.resolve().exists() or args.output_csv.resolve().exists():
        raise FileExistsError("Refusing to overwrite shared-evidence analysis output")
    paths = {
        "qwen_reader_specific": args.qwen_reader_specific.resolve(),
        "phi_reader_specific": args.phi_reader_specific.resolve(),
        "phi_shared_evidence": args.phi_shared_evidence.resolve(),
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()
    }
    if any(payload.get("status") != "PASS" for payload in payloads.values()):
        raise ValueError("All inputs must have status PASS")
    if (
        payloads["phi_shared_evidence"].get("selection_source", {}).get(
            "reference_results_sha256"
        )
        != sha256(paths["qwen_reader_specific"])
    ):
        raise ValueError("Shared-evidence selection source SHA mismatch")

    views = {name: index_rows(payload) for name, payload in payloads.items()}
    keys = set(views["qwen_reader_specific"])
    if len(keys) != 5_400:
        raise ValueError(f"Expected 5,400 rows, found {len(keys)}")
    if any(set(indexed) != keys for indexed in views.values()):
        raise ValueError("Cross-view question-condition identifiers do not match")
    for key in keys:
        reference = views["qwen_reader_specific"][key]
        for view, indexed in views.items():
            row = indexed[key]
            for field in ("line_number", "question", "topic_entity", "gold_answers"):
                if row.get(field) != reference.get(field):
                    raise ValueError(f"Metadata mismatch {key}/{view}/{field}")
        shared = views["phi_shared_evidence"][key]
        if shared.get("selected_evidence") != reference.get("selected_evidence"):
            raise ValueError(f"Shared evidence mismatch {key}")
        if shared.get("selected_endpoints") != reference.get("selected_endpoints"):
            raise ValueError(f"Shared endpoints mismatch {key}")

    question_keys = sorted({(hop, identifier) for hop, identifier, _ in keys})
    scopes = {
        "1-hop": [key for key in question_keys if key[0] == 1],
        "2-hop": [key for key in question_keys if key[0] == 2],
        "3-hop": [key for key in question_keys if key[0] == 3],
        "all-hops": question_keys,
    }
    summary: list[dict[str, object]] = []
    for scope, scope_keys in scopes.items():
        for view, indexed in views.items():
            for condition in CONDITIONS:
                subset = [indexed[(hop, identifier, condition)] for hop, identifier in scope_keys]
                item: dict[str, object] = {
                    "scope": scope,
                    "view": view,
                    "condition": condition,
                    "questions": len(subset),
                }
                for label, field in METRICS.items():
                    item[label] = round(
                        statistics.fmean(float(row[field]) for row in subset), 9
                    )
                if view == "phi_shared_evidence":
                    item["qwen_evidence_tokens"] = round(
                        statistics.fmean(
                            float(row["selector_tokenizer_evidence_tokens"]) for row in subset
                        ),
                        9,
                    )
                    item["phi_evidence_tokens"] = round(
                        statistics.fmean(
                            float(row["reader_tokenizer_evidence_tokens"]) for row in subset
                        ),
                        9,
                    )
                    item["phi_over_256_rate"] = round(
                        statistics.fmean(bool(row["reader_tokenizer_over_256"]) for row in subset),
                        9,
                    )
                summary.append(item)

    rng = np.random.default_rng(args.seed)
    comparisons: list[dict[str, object]] = []
    iterations = (args.bootstrap_iterations, args.permutation_iterations)
    # The frozen replication criterion: sign and interval of RA-MMR minus BM25.
    for scope, scope_keys in scopes.items():
        hop_labels = np.asarray([hop for hop, _ in scope_keys], dtype=np.int8)
        for view, indexed in views.items():
            differences = np.asarray(
                [
                    float(indexed[(hop, identifier, "ra_mmr")]["f1"])
                    - float(indexed[(hop, identifier, "bm25")]["f1"])
                    for hop, identifier in scope_keys
                ],
                dtype=np.float64,
            )
            comparisons.append(
                comparison(
                    contrast_type="within_view_method",
                    view=view,
                    scope=scope,
                    contrast="ra_mmr_minus_bm25",
                    metric="raw_f1",
                    differences=differences,
                    hop_labels=hop_labels,
                    iterations=iterations,
                    rng=rng,
                )
            )

    # Fixed-evidence reader effect: Phi and Qwen answer the identical Qwen evidence.
    for scope, scope_keys in scopes.items():
        hop_labels = np.asarray([hop for hop, _ in scope_keys], dtype=np.int8)
        for condition in CONDITIONS:
            differences = np.asarray(
                [
                    float(views["phi_shared_evidence"][(hop, identifier, condition)]["f1"])
                    - float(views["qwen_reader_specific"][(hop, identifier, condition)]["f1"])
                    for hop, identifier in scope_keys
                ],
                dtype=np.float64,
            )
            comparisons.append(
                comparison(
                    contrast_type="fixed_evidence_reader",
                    view="shared_qwen_evidence",
                    scope=scope,
                    contrast=f"phi_minus_qwen/{condition}",
                    metric="raw_f1",
                    differences=differences,
                    hop_labels=hop_labels,
                    iterations=iterations,
                    rng=rng,
                )
            )

    # Same Phi reader, alternative evidence packing: deployment view minus shared view.
    for scope, scope_keys in scopes.items():
        hop_labels = np.asarray([hop for hop, _ in scope_keys], dtype=np.int8)
        for condition in CONDITIONS:
            differences = np.asarray(
                [
                    float(views["phi_reader_specific"][(hop, identifier, condition)]["f1"])
                    - float(views["phi_shared_evidence"][(hop, identifier, condition)]["f1"])
                    for hop, identifier in scope_keys
                ],
                dtype=np.float64,
            )
            comparisons.append(
                comparison(
                    contrast_type="phi_evidence_packing",
                    view="phi_reader_specific_minus_shared",
                    scope=scope,
                    contrast=f"reader_specific_minus_shared/{condition}",
                    metric="raw_f1",
                    differences=differences,
                    hop_labels=hop_labels,
                    iterations=iterations,
                    rng=rng,
                )
            )

    output = {
        "status": "PASS",
        "analysis_status": "prospectively_frozen_internal_protocol_20260821",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()
        },
        "configuration": {
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap": "paired percentile; stratified by hop for all-hops",
            "permutation_iterations": args.permutation_iterations,
            "permutation": "two-sided paired sign randomization",
            "primary_replication_criterion": "sign of all-hops RA-MMR minus BM25 raw F1",
        },
        "question_counts": {scope: len(scope_keys) for scope, scope_keys in scopes.items()},
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
        if row["scope"] == "all-hops" and row["contrast"] == "ra_mmr_minus_bm25":
            print(
                f"replication_view={row['view']} diff={row['mean_difference']} "
                f"ci=[{row['bootstrap_95_ci_low']},{row['bootstrap_95_ci_high']}] "
                f"p={row['paired_permutation_p']}"
            )
    print(f"json={args.output_json.resolve()}")
    print(f"csv={args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
