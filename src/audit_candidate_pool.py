#!/usr/bin/env python3
"""CPU-only audit of deterministic MetaQA candidate-pool truncation."""

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

from kgqa_core import load_jsonl, load_kg


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return float(ordered[index])


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def stream_prefix_audit(
    adjacency,
    topic: str,
    max_hops: int,
    gold_answers: list[str],
    caps: list[int],
    exact_hops_only: bool = False,
) -> dict[str, object]:
    """Follow the production BFS ordering without materializing GraphPath objects."""
    gold = set(gold_answers)
    found: set[str] = set()
    recall_at_cap: dict[int, float] = {}
    cap_set = set(caps)
    ceiling = max(caps)
    path_count = 0
    frontier: list[tuple[str, ...]] = [(topic,)]
    reached_ceiling = False

    for depth in range(1, max_hops + 1):
        next_frontier: list[tuple[str, ...]] = []
        for prefix in frontier:
            for traversal in adjacency.get(prefix[-1], ()):
                endpoint = traversal.next_entity
                if endpoint in prefix:
                    continue
                next_prefix = prefix + (endpoint,)
                is_candidate = not exact_hops_only or depth == max_hops
                if is_candidate:
                    path_count += 1
                    if endpoint in gold:
                        found.add(endpoint)
                    if path_count in cap_set:
                        recall_at_cap[path_count] = len(found) / len(gold)
                    if path_count >= ceiling:
                        reached_ceiling = True
                        break
                if depth < max_hops:
                    next_frontier.append(next_prefix)
            if reached_ceiling:
                break
        if reached_ceiling:
            break
        frontier = next_frontier
        if not frontier:
            break

    final_recall = len(found) / len(gold)
    for cap in caps:
        if cap not in recall_at_cap:
            recall_at_cap[cap] = final_recall

    return {
        "observed_paths": path_count,
        "right_censored_at_ceiling": reached_ceiling,
        "prefix_answer_recall": {
            str(cap): round(recall_at_cap[cap], 6) for cap in caps
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--sample-per-hop", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--caps", type=int, nargs="+", default=[20_000, 50_000, 100_000]
    )
    parser.add_argument(
        "--path-mode", choices=("up-to", "exact"), default="up-to"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.sample_per_hop < 1:
        raise ValueError("sample-per-hop must be positive")
    caps = sorted(set(args.caps))
    if not caps or any(cap < 1 for cap in caps):
        raise ValueError("caps must contain positive integers")

    started = time.perf_counter()
    data_dir = args.data_dir.resolve()
    print("loading_processed_kg...", flush=True)
    adjacency = load_kg(data_dir / "kg.tsv")
    print(f"kg_entities={len(adjacency)}", flush=True)

    details: list[dict[str, object]] = []
    for hop in (1, 2, 3):
        population = load_jsonl(data_dir / f"{hop}-hop" / f"{args.split}.jsonl")
        sample_size = min(args.sample_per_hop, len(population))
        generator = random.Random(args.seed + hop * 1000)
        sampled = generator.sample(population, sample_size)
        sampled.sort(key=lambda row: int(row["line_number"]))
        print(f"auditing_{hop}hop_questions={sample_size}", flush=True)

        for position, row in enumerate(sampled, start=1):
            question_started = time.perf_counter()
            audit = stream_prefix_audit(
                adjacency=adjacency,
                topic=str(row["topic_entity"]),
                max_hops=hop,
                gold_answers=[str(value) for value in row["gold_answers"]],
                caps=caps,
                exact_hops_only=args.path_mode == "exact",
            )
            detail = {
                "id": row["id"],
                "hop": hop,
                "line_number": row["line_number"],
                "topic_entity": row["topic_entity"],
                **audit,
                "elapsed_seconds": round(time.perf_counter() - question_started, 6),
            }
            details.append(detail)
            if position % 10 == 0 or position == sample_size:
                print(
                    f"{hop}-hop {position}/{sample_size} "
                    f"last_paths={audit['observed_paths']} "
                    f"censored={audit['right_censored_at_ceiling']}",
                    flush=True,
                )

    summary: dict[str, dict[str, object]] = {}
    for hop in (1, 2, 3):
        rows = [row for row in details if int(row["hop"]) == hop]
        counts = [float(row["observed_paths"]) for row in rows]
        hop_summary: dict[str, object] = {
            "questions": len(rows),
            "mean_observed_paths": round(statistics.fmean(counts), 3),
            "p50_observed_paths": round(percentile(counts, 0.50), 3),
            "p95_observed_paths": round(percentile(counts, 0.95), 3),
            "right_censored_questions": sum(
                bool(row["right_censored_at_ceiling"]) for row in rows
            ),
        }
        for cap in caps:
            recalls = [
                float(row["prefix_answer_recall"][str(cap)]) for row in rows
            ]
            hop_summary[f"cap_{cap}"] = {
                "mean_answer_recall": round(statistics.fmean(recalls), 6),
                "all_answers_reachable_rate": round(
                    statistics.fmean(recall == 1.0 for recall in recalls), 6
                ),
            }
        summary[f"{hop}-hop"] = hop_summary

    payload = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "split": args.split,
            "sample_per_hop": args.sample_per_hop,
            "seed": args.seed,
            "caps": caps,
            "path_mode": args.path_mode,
            "sampling": "random without replacement within hop, then line-order sort",
            "candidate_order": (
                "deterministic simple paths of exactly hop length by BFS"
                if args.path_mode == "exact"
                else "deterministic simple paths of lengths 1..hop by BFS"
            ),
            "count_note": "observed path counts are right-censored at the largest cap",
        },
        "summary": summary,
        "details": details,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    atomic_json(args.output.resolve(), payload)

    print("status=PASS", flush=True)
    for name, values in summary.items():
        print(
            f"{name} p50_paths={values['p50_observed_paths']} "
            f"p95_paths={values['p95_observed_paths']} "
            f"censored={values['right_censored_questions']}",
            flush=True,
        )
        for cap in caps:
            cap_values = values[f"cap_{cap}"]
            print(
                f"{name} cap={cap} recall={cap_values['mean_answer_recall']} "
                f"full_recall_rate={cap_values['all_answers_reachable_rate']}",
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
