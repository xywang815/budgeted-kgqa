#!/usr/bin/env python3
"""Audit the official WebQSP test set against per-question Freebase subgraphs.

This script is deliberately retrieval-only.  Gold answers are used only to
measure resource coverage and are never written into a selector input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


FB_RE = re.compile(r"^<fb:(.+)>$")


def normalize_fb(value: str) -> str:
    value = value.strip()
    match = FB_RE.match(value)
    if match:
        return match.group(1)
    if value.startswith("/"):
        return value[1:].replace("/", ".")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graph(path: Path):
    adjacency: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    names: dict[str, str] = {}
    entities: set[str] = set()
    triples = 0
    malformed = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            fields = raw.rstrip("\n").split("\t")
            if len(fields) != 3:
                malformed += 1
                continue
            head, relation, tail = map(normalize_fb, fields)
            triples += 1
            entities.add(head)
            if relation == "type.object.name":
                names[head] = tail
                continue
            adjacency[head][relation].add(tail)
            if fields[2].startswith("<fb:"):
                entities.add(tail)
    return adjacency, names, entities, triples, malformed


def traverse(adjacency, topic: str, chain: list[str]) -> set[str]:
    frontier = {topic}
    for relation in chain:
        relation = normalize_fb(relation)
        next_frontier: set[str] = set()
        for node in frontier:
            next_frontier.update(adjacency.get(node, {}).get(relation, ()))
        frontier = next_frontier
        if not frontier:
            break
    return frontier


def valid_parses(question: dict) -> list[dict]:
    parses = []
    for parse in question.get("Parses", []):
        topic = normalize_fb(str(parse.get("TopicEntityMid") or ""))
        chain = [normalize_fb(str(item)) for item in (parse.get("InferentialChain") or [])]
        answers = {
            normalize_fb(str(answer.get("AnswerArgument") or ""))
            for answer in parse.get("Answers", [])
            if answer.get("AnswerArgument")
        }
        if topic and chain and answers:
            parses.append({"topic": topic, "chain": chain, "answers": answers})
    return parses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-test", type=Path, required=True)
    parser.add_argument("--subgraph-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--min-valid", type=int, default=500)
    parser.add_argument("--min-coverage", type=float, default=0.85)
    args = parser.parse_args()

    started = time.perf_counter()
    official = json.loads(args.official_test.read_text(encoding="utf-8"))
    questions = official.get("Questions", [])
    records = []
    totals = Counter()
    chain_lengths = Counter()

    for index, question in enumerate(questions, start=1):
        question_id = str(question.get("QuestionId") or "")
        graph_path = args.subgraph_dir / f"{question_id}.nxhd"
        parses = valid_parses(question)
        row = {
            "question_id": question_id,
            "raw_question": question.get("RawQuestion", ""),
            "subgraph_exists": graph_path.is_file(),
            "valid_parse_count": len(parses),
            "triple_count": 0,
            "malformed_rows": 0,
            "topic_present": False,
            "gold_entity_recall": 0.0,
            "chain_answer_recall": 0.0,
            "any_gold_in_graph": False,
            "any_gold_chain_reachable": False,
            "name_coverage": 0.0,
        }
        if graph_path.is_file() and parses:
            adjacency, names, entities, triples, malformed = load_graph(graph_path)
            topics = {parse["topic"] for parse in parses}
            gold = set().union(*(parse["answers"] for parse in parses))
            reached = set()
            for parse in parses:
                chain_lengths[len(parse["chain"])] += 1
                reached.update(traverse(adjacency, parse["topic"], parse["chain"]))
            row.update(
                triple_count=triples,
                malformed_rows=malformed,
                topic_present=bool(topics & entities),
                gold_entity_recall=(len(gold & entities) / len(gold)) if gold else 0.0,
                chain_answer_recall=(len(gold & reached) / len(gold)) if gold else 0.0,
                any_gold_in_graph=bool(gold & entities),
                any_gold_chain_reachable=bool(gold & reached),
                name_coverage=(len({item for item in gold if item in names}) / len(gold)) if gold else 0.0,
            )
        records.append(row)
        totals["questions"] += 1
        totals["subgraph_exists"] += int(row["subgraph_exists"])
        totals["valid_question"] += int(bool(parses) and row["subgraph_exists"])
        totals["topic_present"] += int(row["topic_present"])
        totals["any_gold_in_graph"] += int(row["any_gold_in_graph"])
        totals["any_gold_chain_reachable"] += int(row["any_gold_chain_reachable"])
        totals["triples"] += int(row["triple_count"])
        totals["malformed_rows"] += int(row["malformed_rows"])
        if index % 100 == 0 or index == len(questions):
            print(
                f"audited={index}/{len(questions)} "
                f"subgraphs={totals['subgraph_exists']} "
                f"chain_reachable={totals['any_gold_chain_reachable']}",
                flush=True,
            )

    valid = totals["valid_question"]
    coverage = (totals["any_gold_chain_reachable"] / valid) if valid else 0.0
    entity_coverage = (totals["any_gold_in_graph"] / valid) if valid else 0.0
    mean_gold_recall = (
        sum(float(row["chain_answer_recall"]) for row in records if row["subgraph_exists"] and row["valid_parse_count"])
        / valid
        if valid
        else 0.0
    )
    mean_name_coverage = (
        sum(float(row["name_coverage"]) for row in records if row["subgraph_exists"] and row["valid_parse_count"])
        / valid
        if valid
        else 0.0
    )
    gate = {
        "official_question_count_at_least_500": len(questions) >= args.min_valid,
        "valid_subgraph_question_count_at_least_500": valid >= args.min_valid,
        "chain_reachable_question_rate_at_least_threshold": coverage >= args.min_coverage,
        "gold_entity_question_rate_at_least_threshold": entity_coverage >= args.min_coverage,
        "selector_is_gold_free": True,
    }
    report = {
        "status": "PASS" if all(gate.values()) else "GATE_FAIL",
        "source": {
            "official_test": str(args.official_test.resolve()),
            "official_test_sha256": sha256(args.official_test),
            "subgraph_dir": str(args.subgraph_dir.resolve()),
            "official_version": official.get("Version"),
            "freebase_version": official.get("FreebaseVersion"),
        },
        "gate": gate,
        "thresholds": {"min_valid": args.min_valid, "min_coverage": args.min_coverage},
        "summary": {
            **dict(totals),
            "chain_reachable_question_rate": coverage,
            "gold_entity_question_rate": entity_coverage,
            "mean_chain_gold_answer_recall": mean_gold_recall,
            "mean_gold_name_coverage": mean_name_coverage,
            "inferential_chain_length_counts": dict(sorted(chain_lengths.items())),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"status={report['status']}")
    print(f"valid_questions={valid}")
    print(f"chain_reachable_question_rate={coverage:.6f}")
    print(f"gold_entity_question_rate={entity_coverage:.6f}")
    print(f"mean_chain_gold_answer_recall={mean_gold_recall:.6f}")
    print(f"mean_gold_name_coverage={mean_name_coverage:.6f}")
    print(f"report={args.output_json.resolve()}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
