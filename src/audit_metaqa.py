#!/usr/bin/env python3
"""Audit the MetaQA KB and vanilla development/test question files.

This script uses only the Python standard library.  It validates file identity,
row structure, entity membership, and a graph-reachability sanity check.  The
reachability check treats KG edges as undirected because MetaQA questions may
traverse a stored relation in either semantic direction; it is a structural
diagnostic, not the paper's evidence-recall metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


EXPECTED_ROWS = {
    (1, "dev"): 9_992,
    (1, "test"): 9_947,
    (2, "dev"): 14_872,
    (2, "test"): 14_872,
    (3, "dev"): 14_274,
    (3, "test"): 14_274,
}

PINNED_REVISION = "f83854092893610686ab19f2038c039afa9bc643"
TOPIC_PATTERN = re.compile(r"\[([^\[\]]+)\]")
MAX_ISSUE_EXAMPLES = 20


def file_fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    git_header = f"blob {len(data)}\0".encode("ascii")
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob_sha1": hashlib.sha1(git_header + data).hexdigest(),
    }


def remember_issue(examples: list[dict[str, Any]], line_number: int, issue: str) -> None:
    if len(examples) < MAX_ISSUE_EXAMPLES:
        examples.append({"line": line_number, "issue": issue})


def load_kb(path: Path) -> tuple[dict[str, Any], set[str], dict[str, set[str]]]:
    entities: set[str] = set()
    relations: Counter[str] = Counter()
    adjacency: dict[str, set[str]] = defaultdict(set)
    unique_triples: set[tuple[str, str, str]] = set()
    issue_examples: list[dict[str, Any]] = []
    invalid_rows = 0
    empty_field_rows = 0
    self_loops = 0
    rows = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            rows += 1
            line = raw_line.rstrip("\r\n")
            parts = line.split("|")
            if len(parts) != 3:
                invalid_rows += 1
                remember_issue(issue_examples, line_number, "expected exactly 3 pipe-separated fields")
                continue

            subject, relation, obj = parts
            if not subject or not relation or not obj:
                empty_field_rows += 1
                remember_issue(issue_examples, line_number, "empty subject, relation, or object")
                continue

            entities.update((subject, obj))
            relations[relation] += 1
            unique_triples.add((subject, relation, obj))
            adjacency[subject].add(obj)
            adjacency[obj].add(subject)
            if subject == obj:
                self_loops += 1

    degree_values = [len(neighbors) for neighbors in adjacency.values()]
    stats = {
        "path": str(path.resolve()),
        "fingerprint": file_fingerprint(path),
        "rows": rows,
        "valid_rows": rows - invalid_rows - empty_field_rows,
        "invalid_rows": invalid_rows,
        "empty_field_rows": empty_field_rows,
        "unique_triples": len(unique_triples),
        "duplicate_triples": rows - invalid_rows - empty_field_rows - len(unique_triples),
        "entities": len(entities),
        "relations": len(relations),
        "self_loops": self_loops,
        "undirected_degree_mean": (
            sum(degree_values) / len(degree_values) if degree_values else 0.0
        ),
        "undirected_degree_max": max(degree_values, default=0),
        "top_relations": [
            {"relation": relation, "triples": count}
            for relation, count in relations.most_common(20)
        ],
        "issue_examples": issue_examples,
    }
    return stats, entities, adjacency


def make_reachability_checker(adjacency: dict[str, set[str]]):
    @lru_cache(maxsize=2_048)
    def reachable_within_hops(start: str, max_hops: int) -> frozenset[str]:
        visited = {start}
        frontier = {start}
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for entity in frontier:
                next_frontier.update(adjacency.get(entity, ()))
            next_frontier.difference_update(visited)
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        return frozenset(visited)

    return reachable_within_hops


def audit_qa_file(
    path: Path,
    hop: int,
    split: str,
    entities: set[str],
    reachable_within_hops,
) -> dict[str, Any]:
    rows = 0
    invalid_rows = 0
    empty_question_rows = 0
    empty_answer_rows = 0
    topic_marker_errors = 0
    topic_not_in_kg = 0
    gold_answer_not_in_kg = 0
    questions_with_multiple_answers = 0
    total_gold_answers = 0
    reachable_gold_answers = 0
    valid_questions_for_reachability = 0
    questions_all_gold_reachable = 0
    issue_examples: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            rows += 1
            line = raw_line.rstrip("\r\n")
            parts = line.split("\t")
            if len(parts) != 2:
                invalid_rows += 1
                remember_issue(issue_examples, line_number, "expected one tab between question and answers")
                continue

            question, answer_field = parts
            if not question:
                empty_question_rows += 1
                remember_issue(issue_examples, line_number, "empty question")
            answers = answer_field.split("|") if answer_field else []
            if not answers or any(not answer for answer in answers):
                empty_answer_rows += 1
                remember_issue(issue_examples, line_number, "empty gold answer")
                continue

            topics = TOPIC_PATTERN.findall(question)
            if len(topics) != 1:
                topic_marker_errors += 1
                remember_issue(
                    issue_examples,
                    line_number,
                    f"expected one bracketed topic entity, found {len(topics)}",
                )
                continue

            topic = topics[0]
            if topic not in entities:
                topic_not_in_kg += 1
                remember_issue(issue_examples, line_number, "topic entity absent from KG")

            missing_answers = [answer for answer in answers if answer not in entities]
            gold_answer_not_in_kg += len(missing_answers)
            if missing_answers:
                remember_issue(
                    issue_examples,
                    line_number,
                    f"{len(missing_answers)} gold answer(s) absent from KG",
                )

            if len(answers) > 1:
                questions_with_multiple_answers += 1
            total_gold_answers += len(answers)

            if topic in entities and not missing_answers:
                reachable = reachable_within_hops(topic, hop)
                reachable_count = sum(answer in reachable for answer in answers)
                reachable_gold_answers += reachable_count
                valid_questions_for_reachability += 1
                if reachable_count == len(answers):
                    questions_all_gold_reachable += 1
                else:
                    remember_issue(
                        issue_examples,
                        line_number,
                        f"{len(answers) - reachable_count} gold answer(s) not reachable within {hop} hop(s)",
                    )

    expected_rows = EXPECTED_ROWS[(hop, split)]
    return {
        "path": str(path.resolve()),
        "fingerprint": file_fingerprint(path),
        "hop": hop,
        "split": split,
        "rows": rows,
        "expected_rows": expected_rows,
        "matches_expected_rows": rows == expected_rows,
        "invalid_rows": invalid_rows,
        "empty_question_rows": empty_question_rows,
        "empty_answer_rows": empty_answer_rows,
        "topic_marker_errors": topic_marker_errors,
        "topic_not_in_kg": topic_not_in_kg,
        "gold_answer_not_in_kg": gold_answer_not_in_kg,
        "questions_with_multiple_answers": questions_with_multiple_answers,
        "total_gold_answers": total_gold_answers,
        "reachable_gold_answers": reachable_gold_answers,
        "gold_answer_reachability": (
            reachable_gold_answers / total_gold_answers if total_gold_answers else 0.0
        ),
        "valid_questions_for_reachability": valid_questions_for_reachability,
        "questions_all_gold_reachable": questions_all_gold_reachable,
        "all_gold_reachable_question_rate": (
            questions_all_gold_reachable / valid_questions_for_reachability
            if valid_questions_for_reachability
            else 0.0
        ),
        "issue_examples": issue_examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/metaqa"),
        help="MetaQA root containing kb/ and 1-hop/ through 3-hop/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/data_audit.json"),
        help="JSON report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    data_dir = args.data_dir.resolve()
    kb_path = data_dir / "kb" / "kb.txt"
    qa_paths = {
        (hop, split): data_dir / f"{hop}-hop" / "vanilla" / f"qa_{split}.txt"
        for hop in (1, 2, 3)
        for split in ("dev", "test")
    }
    required_paths = [kb_path, *qa_paths.values()]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        print("ERROR: required files are missing:", file=sys.stderr)
        for path in missing_paths:
            print(path, file=sys.stderr)
        return 2

    print("Loading and auditing MetaQA knowledge graph...", flush=True)
    kb_stats, entities, adjacency = load_kb(kb_path)
    reachable_within_hops = make_reachability_checker(adjacency)

    qa_stats: dict[str, Any] = {}
    for (hop, split), path in qa_paths.items():
        key = f"{hop}-hop/{split}"
        print(f"Auditing {key}...", flush=True)
        qa_stats[key] = audit_qa_file(
            path=path,
            hop=hop,
            split=split,
            entities=entities,
            reachable_within_hops=reachable_within_hops,
        )

    hard_failures: list[str] = []
    if kb_stats["invalid_rows"] or kb_stats["empty_field_rows"]:
        hard_failures.append("KB contains malformed or empty-field rows")
    for key, stats in qa_stats.items():
        if not stats["matches_expected_rows"]:
            hard_failures.append(f"{key} row count differs from the official count")
        for field in (
            "invalid_rows",
            "empty_question_rows",
            "empty_answer_rows",
            "topic_marker_errors",
            "topic_not_in_kg",
            "gold_answer_not_in_kg",
        ):
            if stats[field]:
                hard_failures.append(f"{key} has nonzero {field}: {stats[field]}")

    reachability_warnings = [
        f"{key}: some gold answers are not structurally reachable within {stats['hop']} hop(s)"
        for key, stats in qa_stats.items()
        if stats["reachable_gold_answers"] != stats["total_gold_answers"]
    ]

    elapsed = time.perf_counter() - started
    report = {
        "status": "PASS" if not hard_failures else "FAIL",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "provenance": {
            "dataset": "MetaQA vanilla text",
            "official_repository": "https://github.com/yuyuz/MetaQA",
            "download_repository": "https://huggingface.co/datasets/camazlucas/MetaQA",
            "pinned_revision": PINNED_REVISION,
            "note": "Test files and KB were cross-checked by Git blob SHA-1 against paper code repositories.",
        },
        "reachability_definition": "Undirected KG reachability within the nominal hop count; diagnostic only.",
        "kb": kb_stats,
        "qa": qa_stats,
        "hard_failures": hard_failures,
        "warnings": reachability_warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"status={report['status']}")
    print(
        f"kb_rows={kb_stats['rows']} entities={kb_stats['entities']} "
        f"relations={kb_stats['relations']} duplicate_triples={kb_stats['duplicate_triples']}"
    )
    for key, stats in qa_stats.items():
        print(
            f"{key} rows={stats['rows']} total_gold_answers={stats['total_gold_answers']} "
            f"gold_answer_reachability={stats['gold_answer_reachability']:.6f} "
            f"all_gold_reachable_question_rate={stats['all_gold_reachable_question_rate']:.6f}"
        )
    print(f"hard_failures={len(hard_failures)} warnings={len(reachability_warnings)}")
    print(f"report={args.output.resolve()}")
    print(f"elapsed_seconds={elapsed:.3f}")
    return 0 if not hard_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
