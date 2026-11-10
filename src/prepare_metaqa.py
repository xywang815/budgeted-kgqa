#!/usr/bin/env python3
"""Create a reproducible, non-destructive processed view of MetaQA.

The raw files are never modified.  The script:
1. removes only exact duplicate KG triples while preserving first-seen order;
2. converts each QA row to JSONL and retains both raw and corrected answers;
3. repairs a split answer only when two adjacent raw answer fragments, joined by
   a comma, form an exact one-hop KG neighbour of the question topic;
4. writes a manifest containing hashes, counts, and every correction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


QUESTION_TOPIC_RE = re.compile(r"\[(.*?)\]")
DATASETS = tuple(
    (hop, split)
    for hop in (1, 2, 3)
    for split in ("dev", "test")
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def json_line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def load_and_deduplicate_kg(
    kb_path: Path,
) -> tuple[list[tuple[str, str, str]], int, dict[str, dict[str, set[str]]]]:
    unique_triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    raw_rows = 0
    adjacency: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    with kb_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw_rows += 1
            fields = raw.rstrip("\n").split("|")
            if len(fields) != 3 or any(not field for field in fields):
                raise ValueError(
                    f"Malformed KG row at {kb_path}:{line_number}: {raw!r}"
                )
            head, relation, tail = fields
            triple = (head, relation, tail)
            if triple not in seen:
                seen.add(triple)
                unique_triples.append(triple)
                adjacency[head][tail].add(relation)
                adjacency[tail][head].add(relation)

    return unique_triples, raw_rows, adjacency


def repair_one_hop_answers(
    answers: list[str],
    topic_neighbors: dict[str, set[str]],
) -> tuple[list[str], list[dict[str, object]]]:
    """Repair only unambiguous adjacent fragments supported by the KG."""
    corrected: list[str] = []
    corrections: list[dict[str, object]] = []
    index = 0

    while index < len(answers):
        if index + 1 >= len(answers):
            corrected.append(answers[index])
            break

        left = answers[index]
        right = answers[index + 1]
        candidates = []
        if left not in topic_neighbors and right not in topic_neighbors:
            for separator in (", ", ","):
                candidate = left + separator + right
                if candidate in topic_neighbors:
                    candidates.append(candidate)

        candidates = sorted(set(candidates))
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous comma repair for {left!r}|{right!r}: {candidates}"
            )
        if candidates:
            merged = candidates[0]
            corrected.append(merged)
            corrections.append(
                {
                    "from_answers": [left, right],
                    "to_answer": merged,
                    "supporting_relations": sorted(topic_neighbors[merged]),
                    "rule": "adjacent_unreachable_fragments_form_direct_kg_neighbor",
                }
            )
            index += 2
        else:
            corrected.append(left)
            index += 1

    return corrected, corrections


def parse_qa_rows(
    path: Path,
    hop: int,
    split: str,
    adjacency: dict[str, dict[str, set[str]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    all_corrections: list[dict[str, object]] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(
                    f"Malformed QA row at {path}:{line_number}: expected 2 tab fields"
                )
            question, answer_text = fields
            match = QUESTION_TOPIC_RE.search(question)
            if match is None:
                raise ValueError(f"Missing [topic] at {path}:{line_number}")
            topic = match.group(1)
            raw_answers = answer_text.split("|")
            if any(not answer for answer in raw_answers):
                raise ValueError(f"Empty answer at {path}:{line_number}")

            corrected_answers = list(raw_answers)
            corrections: list[dict[str, object]] = []
            if hop == 1:
                corrected_answers, corrections = repair_one_hop_answers(
                    raw_answers,
                    adjacency.get(topic, {}),
                )

            question_id = f"metaqa-{hop}hop-{split}-{line_number:05d}"
            row = {
                "id": question_id,
                "hop": hop,
                "split": split,
                "line_number": line_number,
                "question": question,
                "topic_entity": topic,
                "raw_answers": raw_answers,
                "gold_answers": corrected_answers,
                "label_corrections": corrections,
            }
            rows.append(row)

            for correction in corrections:
                all_corrections.append(
                    {
                        "id": question_id,
                        "source_file": str(path),
                        "line_number": line_number,
                        "question": question,
                        "topic_entity": topic,
                        **correction,
                    }
                )

    return rows, all_corrections


def serialize_tsv(rows: Iterable[tuple[str, str, str]]) -> str:
    return "".join("\t".join(row) + "\n" for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve()
    kb_path = data_dir / "kb" / "kb.txt"
    if not kb_path.is_file():
        raise FileNotFoundError(kb_path)

    unique_triples, raw_kg_rows, adjacency = load_and_deduplicate_kg(kb_path)
    processed_kg_path = output_dir / "kg.tsv"
    write_text_atomic(processed_kg_path, serialize_tsv(unique_triples))

    dataset_reports = []
    corrections: list[dict[str, object]] = []
    total_qa_rows = 0

    for hop, split in DATASETS:
        source_path = data_dir / f"{hop}-hop" / "vanilla" / f"qa_{split}.txt"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        rows, file_corrections = parse_qa_rows(
            source_path,
            hop,
            split,
            adjacency,
        )
        destination = output_dir / f"{hop}-hop" / f"{split}.jsonl"
        write_text_atomic(destination, "".join(json_line(row) for row in rows))
        total_qa_rows += len(rows)
        corrections.extend(file_corrections)
        dataset_reports.append(
            {
                "hop": hop,
                "split": split,
                "rows": len(rows),
                "corrections": len(file_corrections),
                "source_path": str(source_path),
                "source_sha256": sha256_file(source_path),
                "processed_path": str(destination),
                "processed_sha256": sha256_file(destination),
            }
        )

    expected_corrections = {"1-hop/dev": 2, "1-hop/test": 3}
    observed_corrections = {
        f"{report['hop']}-hop/{report['split']}": report["corrections"]
        for report in dataset_reports
        if report["corrections"]
    }
    if observed_corrections != expected_corrections:
        raise RuntimeError(
            "Correction count differs from the audited pinned dataset: "
            f"expected={expected_corrections}, observed={observed_corrections}"
        )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "policy": {
            "raw_files_modified": False,
            "kg": "remove_exact_duplicate_triples_preserving_first_seen_order",
            "labels": (
                "retain raw_answers and additionally repair only adjacent unreachable "
                "one-hop fragments whose comma-joined form is an exact KG neighbour"
            ),
            "evaluation": (
                "report corrected gold_answers as primary and raw_answers as a "
                "documented sensitivity/comparability analysis"
            ),
        },
        "kg": {
            "source_path": str(kb_path),
            "source_sha256": sha256_file(kb_path),
            "raw_rows": raw_kg_rows,
            "unique_rows": len(unique_triples),
            "removed_exact_duplicates": raw_kg_rows - len(unique_triples),
            "processed_path": str(processed_kg_path),
            "processed_sha256": sha256_file(processed_kg_path),
        },
        "qa": {
            "total_rows": total_qa_rows,
            "files": dataset_reports,
        },
        "corrections": corrections,
    }
    write_text_atomic(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    print("status=PASS")
    print(
        f"kg_raw_rows={raw_kg_rows} kg_unique_rows={len(unique_triples)} "
        f"exact_duplicates_removed={raw_kg_rows - len(unique_triples)}"
    )
    for report in dataset_reports:
        print(
            f"{report['hop']}-hop/{report['split']} "
            f"rows={report['rows']} corrections={report['corrections']}"
        )
    print(f"qa_total_rows={total_qa_rows} corrections_total={len(corrections)}")
    for correction in corrections:
        source = "|".join(correction["from_answers"])
        print(
            f"correction={correction['id']} {source} -> "
            f"{correction['to_answer']}"
        )
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"status=FAIL error={error}", file=sys.stderr)
        raise
