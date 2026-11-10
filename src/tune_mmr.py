#!/usr/bin/env python3
"""Development-only grid search for global BA-MMR and RA-MMR settings."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

from transformers import AutoTokenizer

from kgqa_core import (
    bm25_scores,
    budget_aware_mmr,
    enumerate_simple_paths,
    evidence_tokens,
    load_jsonl,
    load_kg,
    pack_in_order,
    rank_by_scores,
    relation_bm25_scores,
    standalone_evidence_tokens,
)


GLOBAL_ADJACENCY = None
GLOBAL_TOKENIZER = None


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def endpoint_metrics(paths, gold_answers: list[str]) -> dict[str, float]:
    ordered_endpoints = []
    seen = set()
    for path in paths:
        if path.endpoint not in seen:
            seen.add(path.endpoint)
            ordered_endpoints.append(path.endpoint)
    predicted = set(ordered_endpoints)
    gold = set(gold_answers)
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "endpoint_precision": precision,
        "endpoint_recall": recall,
        "endpoint_f1": f1,
        "endpoint_hits_at_1": float(
            bool(ordered_endpoints) and ordered_endpoints[0] in gold
        ),
        "unique_endpoints": len(predicted),
    }


def evaluate_question(task: dict[str, object]) -> dict[str, object]:
    if GLOBAL_ADJACENCY is None or GLOBAL_TOKENIZER is None:
        raise RuntimeError("Worker globals were not initialized")
    row = task["row"]
    hop = int(task["hop"])
    budgets = [int(value) for value in task["budgets"]]
    configurations = [tuple(value) for value in task["configurations"]]
    max_paths = int(task["max_paths"])

    started = time.perf_counter()
    paths, truncated = enumerate_simple_paths(
        GLOBAL_ADJACENCY,
        str(row["topic_entity"]),
        hop,
        max_paths,
        exact_hops_only=True,
    )
    scores = bm25_scores(str(row["question"]), paths)
    relation_scores = relation_bm25_scores(str(row["question"]), paths)
    ranked = rank_by_scores(paths, scores)
    relation_ranked = rank_by_scores(paths, relation_scores)
    standalone_costs = standalone_evidence_tokens(paths, GLOBAL_TOKENIZER)
    signatures = [path.signature() for path in paths]
    directional_signatures = [path.directional_signature() for path in paths]
    cost_by_path = dict(zip(paths, standalone_costs))
    ranked_costs = [cost_by_path[path] for path in ranked]
    relation_ranked_costs = [cost_by_path[path] for path in relation_ranked]
    gold_answers = [str(value) for value in row["gold_answers"]]
    conditions: list[dict[str, object]] = []

    for budget in budgets:
        baselines = {
            "bfs": pack_in_order(
                paths,
                budget,
                GLOBAL_TOKENIZER,
                standalone_costs=standalone_costs,
            ),
            "bm25": pack_in_order(
                ranked,
                budget,
                GLOBAL_TOKENIZER,
                standalone_costs=ranked_costs,
            ),
            "relation_bm25": pack_in_order(
                relation_ranked,
                budget,
                GLOBAL_TOKENIZER,
                standalone_costs=relation_ranked_costs,
            ),
        }
        for method, selected in baselines.items():
            conditions.append(
                {
                    "method": method,
                    "budget": budget,
                    "mmr_lambda": None,
                    "cost_exponent": None,
                    **endpoint_metrics(selected, gold_answers),
                    "selected_paths": len(selected),
                    "evidence_tokens": evidence_tokens(
                        selected, GLOBAL_TOKENIZER
                    ),
                }
            )

        selectors = (
            ("ba_mmr", scores, signatures),
            ("ra_mmr", relation_scores, directional_signatures),
        )
        for method, method_scores, method_signatures in selectors:
            for mmr_lambda, cost_exponent in configurations:
                selected = budget_aware_mmr(
                    paths,
                    method_scores,
                    budget,
                    GLOBAL_TOKENIZER,
                    relevance_weight=float(mmr_lambda),
                    cost_exponent=float(cost_exponent),
                    standalone_costs=standalone_costs,
                    signatures=method_signatures,
                )
                conditions.append(
                    {
                        "method": method,
                        "budget": budget,
                        "mmr_lambda": float(mmr_lambda),
                        "cost_exponent": float(cost_exponent),
                        **endpoint_metrics(selected, gold_answers),
                        "selected_paths": len(selected),
                        "evidence_tokens": evidence_tokens(
                            selected, GLOBAL_TOKENIZER
                        ),
                    }
                )

    return {
        "id": row["id"],
        "hop": hop,
        "line_number": row["line_number"],
        "candidate_paths": len(paths),
        "candidate_truncated": truncated,
        "conditions": conditions,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    def mean(name: str) -> float:
        return round(statistics.fmean(float(row[name]) for row in rows), 6)

    return {
        "observations": len(rows),
        "endpoint_precision": mean("endpoint_precision"),
        "endpoint_recall": mean("endpoint_recall"),
        "endpoint_f1": mean("endpoint_f1"),
        "endpoint_hits_at_1": mean("endpoint_hits_at_1"),
        "mean_unique_endpoints": mean("unique_endpoints"),
        "mean_selected_paths": mean("selected_paths"),
        "mean_evidence_tokens": mean("evidence_tokens"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--sample-per-hop", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument(
        "--mmr-lambdas", type=float, nargs="+", default=[0.30, 0.50, 0.70, 0.90]
    )
    parser.add_argument(
        "--cost-exponents", type=float, nargs="+", default=[0.00, 0.50, 1.00]
    )
    parser.add_argument("--max-paths", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.sample_per_hop < 1 or args.max_paths < 1 or args.workers < 1:
        raise ValueError("sample-per-hop, max-paths, and workers must be positive")
    budgets = sorted(set(args.budgets))
    if not budgets or any(value < 1 for value in budgets):
        raise ValueError("budgets must contain positive integers")
    lambdas = sorted(set(args.mmr_lambdas))
    exponents = sorted(set(args.cost_exponents))
    if any(not 0.0 <= value <= 1.0 for value in lambdas):
        raise ValueError("mmr-lambdas must be in [0, 1]")
    if any(value < 0.0 for value in exponents):
        raise ValueError("cost-exponents must be non-negative")
    configurations = [
        (mmr_lambda, cost_exponent)
        for mmr_lambda in lambdas
        for cost_exponent in exponents
    ]

    started = time.perf_counter()
    data_dir = args.data_dir.resolve()
    print("loading_processed_kg...", flush=True)
    global GLOBAL_ADJACENCY, GLOBAL_TOKENIZER
    GLOBAL_ADJACENCY = load_kg(data_dir / "kg.tsv")
    GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
        args.model_dir.resolve(),
        local_files_only=True,
        trust_remote_code=False,
    )
    print(
        f"kg_entities={len(GLOBAL_ADJACENCY)} configurations={len(configurations)}",
        flush=True,
    )

    tasks: list[dict[str, object]] = []
    for hop in (1, 2, 3):
        population = load_jsonl(data_dir / f"{hop}-hop" / "dev.jsonl")
        generator = random.Random(args.seed + hop * 1000)
        sampled = generator.sample(population, min(args.sample_per_hop, len(population)))
        sampled.sort(key=lambda row: int(row["line_number"]))
        for row in sampled:
            tasks.append(
                {
                    "row": row,
                    "hop": hop,
                    "budgets": budgets,
                    "configurations": configurations,
                    "max_paths": args.max_paths,
                }
            )

    details: list[dict[str, object]] = []
    if args.workers == 1:
        for index, task in enumerate(tasks, start=1):
            details.append(evaluate_question(task))
            print(f"completed={index}/{len(tasks)}", flush=True)
    else:
        context = get_context("fork")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
        ) as executor:
            futures = [executor.submit(evaluate_question, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                detail = future.result()
                details.append(detail)
                print(
                    f"completed={index}/{len(tasks)} id={detail['id']} "
                    f"paths={detail['candidate_paths']}",
                    flush=True,
                )

    details.sort(key=lambda row: (int(row["hop"]), int(row["line_number"])))
    flat_rows = []
    for detail in details:
        for condition in detail["conditions"]:
            flat_rows.append(
                {
                    "id": detail["id"],
                    "hop": detail["hop"],
                    "line_number": detail["line_number"],
                    **condition,
                }
            )

    ranked_configs_by_method: dict[
        str, list[tuple[str, dict[str, float]]]
    ] = {}
    selected_configs: dict[str, tuple[str, dict[str, float]]] = {}
    for method in ("ba_mmr", "ra_mmr"):
        config_summary: dict[str, dict[str, float]] = {}
        for mmr_lambda, cost_exponent in configurations:
            rows = [
                row
                for row in flat_rows
                if row["method"] == method
                and row["mmr_lambda"] == mmr_lambda
                and row["cost_exponent"] == cost_exponent
            ]
            name = f"lambda_{mmr_lambda:.2f}_cost_{cost_exponent:.2f}"
            config_summary[name] = {
                "mmr_lambda": mmr_lambda,
                "cost_exponent": cost_exponent,
                **summarize(rows),
            }

        ranked_configs = sorted(
            config_summary.items(),
            key=lambda item: (
                -item[1]["endpoint_f1"],
                -item[1]["endpoint_recall"],
                -item[1]["endpoint_precision"],
                item[1]["mmr_lambda"],
                item[1]["cost_exponent"],
            ),
        )
        ranked_configs_by_method[method] = ranked_configs
        selected_configs[method] = ranked_configs[0]

    cell_summary = {}
    for hop in (1, 2, 3):
        for budget in budgets:
            for method in (
                "bfs",
                "bm25",
                "relation_bm25",
                "ba_mmr",
                "ra_mmr",
            ):
                selected_values = (
                    selected_configs[method][1]
                    if method in selected_configs
                    else None
                )
                rows = [
                    row
                    for row in flat_rows
                    if int(row["hop"]) == hop
                    and int(row["budget"]) == budget
                    and row["method"] == method
                    and (
                        selected_values is None
                        or (
                            row["mmr_lambda"]
                            == selected_values["mmr_lambda"]
                            and row["cost_exponent"]
                            == selected_values["cost_exponent"]
                        )
                    )
                ]
                cell_summary[f"{hop}-hop/B{budget}/{method}"] = summarize(rows)

    payload = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "split": "dev",
            "sample_per_hop": args.sample_per_hop,
            "seed": args.seed,
            "budgets": budgets,
            "mmr_lambdas": lambdas,
            "cost_exponents": exponents,
            "max_paths": args.max_paths,
            "path_mode": "exact",
            "workers": args.workers,
            "selection_rule": (
                "within each MMR family, maximize macro endpoint F1 over all "
                "dev questions and budgets; tie-break by recall, precision, "
                "smaller lambda, smaller cost exponent"
            ),
        },
        "ranked_configuration_summary": {
            method: [
                {"name": name, **values}
                for name, values in ranked_configs_by_method[method]
            ]
            for method in ranked_configs_by_method
        },
        "selected_configurations": {
            method: {"name": name, **values}
            for method, (name, values) in selected_configs.items()
        },
        "selected_configuration_cell_summary": cell_summary,
        "details": details,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(args.output.resolve(), payload)

    print("status=PASS", flush=True)
    for method, ranked_configs in ranked_configs_by_method.items():
        for rank, (name, values) in enumerate(ranked_configs, start=1):
            print(
                f"method={method} rank={rank} config={name} "
                f"f1={values['endpoint_f1']} "
                f"precision={values['endpoint_precision']} "
                f"recall={values['endpoint_recall']} "
                f"tokens={values['mean_evidence_tokens']}",
                flush=True,
            )
        best_name, best_values = selected_configs[method]
        print(
            f"selected_method={method} config={best_name} "
            f"lambda={best_values['mmr_lambda']} "
            f"cost_exponent={best_values['cost_exponent']}",
            flush=True,
        )
    for name, values in sorted(cell_summary.items()):
        print(
            f"cell={name} f1={values['endpoint_f1']} "
            f"precision={values['endpoint_precision']} "
            f"recall={values['endpoint_recall']}",
            flush=True,
        )
    print(f"report={args.output.resolve()}", flush=True)
    print(f"elapsed_seconds={payload['elapsed_seconds']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"status=FAIL error={error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
