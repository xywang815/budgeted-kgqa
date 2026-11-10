#!/usr/bin/env python3
"""CPU-only pilot audit for MetaQA graph-path selection strategies."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from kgqa_core import (
    answer_recall,
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


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return float(ordered[index])


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    recalls = [float(row["answer_recall"]) for row in rows]
    tokens = [float(row["evidence_tokens"]) for row in rows]
    counts = [float(row["selected_paths"]) for row in rows]
    return {
        "questions": len(rows),
        "mean_answer_recall": round(statistics.fmean(recalls), 6),
        "all_answer_reachable_rate": round(
            statistics.fmean(value == 1.0 for value in recalls), 6
        ),
        "mean_evidence_tokens": round(statistics.fmean(tokens), 3),
        "p50_evidence_tokens": round(percentile(tokens, 0.50), 3),
        "p95_evidence_tokens": round(percentile(tokens, 0.95), 3),
        "mean_selected_paths": round(statistics.fmean(counts), 3),
        "mean_endpoint_precision": round(
            statistics.fmean(float(row["endpoint_precision"]) for row in rows), 6
        ),
        "mean_endpoint_f1": round(
            statistics.fmean(float(row["endpoint_f1"]) for row in rows), 6
        ),
        "endpoint_hits_at_1": round(
            statistics.fmean(float(row["endpoint_hits_at_1"]) for row in rows), 6
        ),
    }


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
        "endpoint_f1": f1,
        "endpoint_hits_at_1": float(
            bool(ordered_endpoints) and ordered_endpoints[0] in gold
        ),
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--sample-per-hop", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--max-paths", type=int, default=100_000)
    parser.add_argument(
        "--path-mode", choices=("up-to", "exact"), default="exact"
    )
    parser.add_argument("--ba-mmr-lambda", type=float, default=0.90)
    parser.add_argument("--ba-cost-exponent", type=float, default=0.50)
    parser.add_argument("--ra-mmr-lambda", type=float, default=0.90)
    parser.add_argument("--ra-cost-exponent", type=float, default=1.00)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.sample_per_hop < 1:
        raise ValueError("sample-per-hop must be positive")
    if any(budget < 1 for budget in args.budgets):
        raise ValueError("budgets must be positive")
    if not 0.0 <= args.ba_mmr_lambda <= 1.0:
        raise ValueError("ba-mmr-lambda must be in [0, 1]")
    if not 0.0 <= args.ra_mmr_lambda <= 1.0:
        raise ValueError("ra-mmr-lambda must be in [0, 1]")
    if args.ba_cost_exponent < 0.0 or args.ra_cost_exponent < 0.0:
        raise ValueError("cost exponents must be non-negative")

    started = time.perf_counter()
    data_dir = args.data_dir.resolve()
    print("loading_processed_kg...", flush=True)
    adjacency = load_kg(data_dir / "kg.tsv")
    print(f"kg_entities={len(adjacency)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir.resolve(),
        local_files_only=True,
        trust_remote_code=False,
    )

    details: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {}
    candidate_reports: dict[str, list[dict[str, object]]] = {}

    for hop in (1, 2, 3):
        source = data_dir / f"{hop}-hop" / f"{args.split}.jsonl"
        population = load_jsonl(source)
        sample_size = min(args.sample_per_hop, len(population))
        generator = random.Random(args.seed + hop * 1000)
        sampled = generator.sample(population, sample_size)
        sampled.sort(key=lambda row: int(row["line_number"]))
        print(f"auditing_{hop}hop_questions={sample_size}", flush=True)

        for position, row in enumerate(sampled, start=1):
            question_started = time.perf_counter()
            paths, truncated = enumerate_simple_paths(
                adjacency,
                str(row["topic_entity"]),
                hop,
                args.max_paths,
                exact_hops_only=args.path_mode == "exact",
            )
            gold_answers = [str(value) for value in row["gold_answers"]]
            scores = bm25_scores(str(row["question"]), paths)
            ranked = rank_by_scores(paths, scores)
            relation_scores = relation_bm25_scores(str(row["question"]), paths)
            relation_ranked = rank_by_scores(paths, relation_scores)
            candidate_recall = answer_recall(paths, gold_answers)
            standalone_costs = standalone_evidence_tokens(paths, tokenizer)
            signatures = [path.signature() for path in paths]
            directional_signatures = [
                path.directional_signature() for path in paths
            ]
            cost_by_path = dict(zip(paths, standalone_costs))

            candidate_row = {
                "id": row["id"],
                "hop": hop,
                "candidate_paths": len(paths),
                "candidate_answer_recall": candidate_recall,
                "truncated": truncated,
            }
            candidate_reports.setdefault(f"{hop}-hop", []).append(candidate_row)
            conditions: dict[str, dict[str, object]] = {}

            for k in (1, 4, 8):
                selected = ranked[:k]
                name = f"bm25_top{k}"
                conditions[name] = {
                    "selected_paths": len(selected),
                    "evidence_tokens": evidence_tokens(selected, tokenizer),
                    "answer_recall": answer_recall(selected, gold_answers),
                    **endpoint_metrics(selected, gold_answers),
                }
                relation_selected = relation_ranked[:k]
                relation_name = f"relation_bm25_top{k}"
                conditions[relation_name] = {
                    "selected_paths": len(relation_selected),
                    "evidence_tokens": evidence_tokens(
                        relation_selected, tokenizer
                    ),
                    "answer_recall": answer_recall(
                        relation_selected, gold_answers
                    ),
                    **endpoint_metrics(relation_selected, gold_answers),
                }

            gold_set = set(gold_answers)
            cheapest_oracle_path = {}
            for path_index, path in enumerate(paths):
                if path.endpoint not in gold_set:
                    continue
                path_cost = standalone_costs[path_index]
                key = (path_cost, path.hops, path.text())
                previous = cheapest_oracle_path.get(path.endpoint)
                if previous is None or key < previous[0]:
                    cheapest_oracle_path[path.endpoint] = (key, path)
            oracle_order = [
                value[1]
                for value in sorted(
                    cheapest_oracle_path.values(),
                    key=lambda value: value[0],
                )
            ]
            ranked_costs = [cost_by_path[path] for path in ranked]
            relation_ranked_costs = [
                cost_by_path[path] for path in relation_ranked
            ]
            oracle_costs = [cost_by_path[path] for path in oracle_order]
            for budget in sorted(set(args.budgets)):
                selections = {
                    f"bfs_prefix_B{budget}": pack_in_order(
                        paths,
                        budget,
                        tokenizer,
                        standalone_costs=standalone_costs,
                    ),
                    f"bm25_budget_B{budget}": pack_in_order(
                        ranked,
                        budget,
                        tokenizer,
                        standalone_costs=ranked_costs,
                    ),
                    f"relation_bm25_budget_B{budget}": pack_in_order(
                        relation_ranked,
                        budget,
                        tokenizer,
                        standalone_costs=relation_ranked_costs,
                    ),
                    f"ba_mmr_B{budget}": budget_aware_mmr(
                        paths,
                        scores,
                        budget,
                        tokenizer,
                        relevance_weight=args.ba_mmr_lambda,
                        cost_exponent=args.ba_cost_exponent,
                        standalone_costs=standalone_costs,
                        signatures=signatures,
                    ),
                    f"ra_mmr_B{budget}": budget_aware_mmr(
                        paths,
                        relation_scores,
                        budget,
                        tokenizer,
                        relevance_weight=args.ra_mmr_lambda,
                        cost_exponent=args.ra_cost_exponent,
                        standalone_costs=standalone_costs,
                        signatures=directional_signatures,
                    ),
                    f"oracle_B{budget}": pack_in_order(
                        oracle_order,
                        budget,
                        tokenizer,
                        standalone_costs=oracle_costs,
                    ),
                }
                for name, selected in selections.items():
                    token_count = evidence_tokens(selected, tokenizer)
                    if token_count > budget:
                        raise AssertionError(
                            f"Budget violation for {row['id']} {name}: {token_count}"
                        )
                    conditions[name] = {
                        "selected_paths": len(selected),
                        "evidence_tokens": token_count,
                        "answer_recall": answer_recall(selected, gold_answers),
                        **endpoint_metrics(selected, gold_answers),
                    }

            detail = {
                "id": row["id"],
                "hop": hop,
                "line_number": row["line_number"],
                "question": row["question"],
                "topic_entity": row["topic_entity"],
                "gold_answers": gold_answers,
                "candidate_paths": len(paths),
                "candidate_answer_recall": candidate_recall,
                "candidate_truncated": truncated,
                "conditions": conditions,
                "elapsed_seconds": round(time.perf_counter() - question_started, 6),
            }
            details.append(detail)
            for name, values in conditions.items():
                grouped.setdefault(f"{hop}-hop/{name}", []).append(values)

            print(
                f"{hop}-hop {position}/{sample_size} id={row['id']} "
                f"paths={len(paths)} recall={candidate_recall:.3f} "
                f"truncated={truncated}",
                flush=True,
            )

    condition_summary = {
        name: summarize(rows)
        for name, rows in sorted(grouped.items())
    }
    candidate_summary = {}
    for name, rows in sorted(candidate_reports.items()):
        path_counts = [float(row["candidate_paths"]) for row in rows]
        recalls = [float(row["candidate_answer_recall"]) for row in rows]
        candidate_summary[name] = {
            "questions": len(rows),
            "mean_candidate_paths": round(statistics.fmean(path_counts), 3),
            "p50_candidate_paths": round(percentile(path_counts, 0.50), 3),
            "p95_candidate_paths": round(percentile(path_counts, 0.95), 3),
            "mean_candidate_answer_recall": round(statistics.fmean(recalls), 6),
            "all_answer_reachable_rate": round(
                statistics.fmean(value == 1.0 for value in recalls), 6
            ),
            "truncated_questions": sum(bool(row["truncated"]) for row in rows),
        }

    payload = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "split": args.split,
            "sample_per_hop": args.sample_per_hop,
            "seed": args.seed,
            "budgets": sorted(set(args.budgets)),
            "max_paths": args.max_paths,
            "path_mode": args.path_mode,
            "ba_mmr": {
                "mmr_lambda": args.ba_mmr_lambda,
                "cost_exponent": args.ba_cost_exponent,
            },
            "ra_mmr": {
                "mmr_lambda": args.ra_mmr_lambda,
                "cost_exponent": args.ra_cost_exponent,
            },
            "dr_mmr": (
                "development-frozen depth routing: BA-MMR for 1/2-hop and "
                "RA-MMR for 3-hop"
            ),
            "budget_definition": "tokenized evidence body only",
            "tokenizer_path": str(args.model_dir.resolve()),
            "sampling": "random without replacement within hop, then line-order sort",
            "relation_scoring": (
                "BM25 over frozen direction-aware schema verbalizations; entity names excluded"
            ),
        },
        "candidate_summary": candidate_summary,
        "condition_summary": condition_summary,
        "details": details,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(args.output.resolve(), payload)

    print("status=PASS", flush=True)
    for name, values in candidate_summary.items():
        print(
            f"{name} mean_paths={values['mean_candidate_paths']} "
            f"p95_paths={values['p95_candidate_paths']} "
            f"mean_recall={values['mean_candidate_answer_recall']} "
            f"truncated={values['truncated_questions']}",
            flush=True,
        )
    middle_budget = sorted(set(args.budgets))[len(set(args.budgets)) // 2]
    displayed_suffixes = (
        "bm25_top4",
        "relation_bm25_top4",
        f"bfs_prefix_B{middle_budget}",
        f"bm25_budget_B{middle_budget}",
        f"relation_bm25_budget_B{middle_budget}",
        f"ba_mmr_B{middle_budget}",
        f"ra_mmr_B{middle_budget}",
        f"oracle_B{middle_budget}",
    )
    for name, values in condition_summary.items():
        if name.endswith(displayed_suffixes):
            print(
                f"selection={name} recall={values['mean_answer_recall']} "
                f"endpoint_f1={values['mean_endpoint_f1']} "
                f"precision={values['mean_endpoint_precision']} "
                f"tokens={values['mean_evidence_tokens']} "
                f"paths={values['mean_selected_paths']}",
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
