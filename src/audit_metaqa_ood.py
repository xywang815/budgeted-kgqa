#!/usr/bin/env python3
"""Pre-inference MetaQA template/relation-shift audit on the frozen test sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from kgqa_core import (
    enumerate_simple_paths,
    load_jsonl,
    load_kg,
    relation_bm25_scores,
)
from run_week2_components import atomic_json


ENTITY_RE = re.compile(r"\[[^\]]+\]")


def question_template(question: str) -> str:
    normalized = ENTITY_RE.sub("[ENTITY]", question)
    return " ".join(normalized.casefold().split())


def directed_chain(path) -> tuple[str, ...]:
    values = []
    for index, edge in enumerate(path.edges):
        source = path.nodes[index]
        target = path.nodes[index + 1]
        if edge.head == source and edge.tail == target:
            values.append(f"F:{edge.relation}")
        elif edge.tail == source and edge.head == target:
            values.append(f"I:{edge.relation}")
        else:
            raise ValueError("Path contains a disconnected edge")
    return tuple(values)


def derive_signature(adjacency, row: dict[str, object], cap: int) -> dict[str, object]:
    paths, truncated = enumerate_simple_paths(
        adjacency,
        str(row["topic_entity"]),
        int(row["hop"]),
        cap,
        exact_hops_only=True,
    )
    gold = {str(value) for value in row["gold_answers"]}
    endpoints_by_chain: dict[tuple[str, ...], set[str]] = defaultdict(set)
    representative_by_chain = {}
    for path in paths:
        chain = directed_chain(path)
        endpoints_by_chain[chain].add(path.endpoint)
        representative_by_chain.setdefault(chain, path)

    ordered_chains = sorted(endpoints_by_chain)
    semantic_scores = relation_bm25_scores(
        str(row["question"]),
        [representative_by_chain[chain] for chain in ordered_chains],
    )
    semantic_by_chain = dict(zip(ordered_chains, semantic_scores))

    ranked = []
    for chain, endpoints in endpoints_by_chain.items():
        overlap = len(endpoints & gold)
        precision = overlap / len(endpoints) if endpoints else 0.0
        recall = overlap / len(gold) if gold else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        ranked.append(
            (f1, recall, precision, semantic_by_chain[chain], chain, endpoints)
        )
    answering = [item for item in ranked if item[1] > 0.0]
    if answering:
        ranked = answering
    ranked.sort(
        key=lambda item: (-item[3], -item[0], -item[1], -item[2], item[4])
    )
    best = ranked[0] if ranked else (0.0, 0.0, 0.0, 0.0, tuple(), set())
    best_f1, best_recall, best_precision, best_semantic, chain, endpoints = best
    answer_fit_ties = sum(
        abs(candidate[0] - best_f1) <= 1e-12
        and abs(candidate[1] - best_recall) <= 1e-12
        and abs(candidate[2] - best_precision) <= 1e-12
        for candidate in ranked
    )
    tied = sum(
        abs(candidate[3] - best_semantic) <= 1e-12
        and abs(candidate[0] - best_f1) <= 1e-12
        and abs(candidate[1] - best_recall) <= 1e-12
        and abs(candidate[2] - best_precision) <= 1e-12
        for candidate in ranked
    )
    reached = endpoints & gold
    return {
        "signature": ">".join(chain),
        "chain": list(chain),
        "canonical_chain_answer_fit_ties": answer_fit_ties,
        "canonical_chain_ties": tied,
        "canonical_chain_relation_bm25": best_semantic,
        "canonical_chain_precision": best_precision,
        "canonical_chain_recall": best_recall,
        "canonical_chain_f1": best_f1,
        "gold_reached": len(reached),
        "gold_total": len(gold),
        "gold_recall": len(reached) / len(gold) if gold else 0.0,
        "candidate_paths": len(paths),
        "candidate_truncated": truncated,
    }


def mean(rows: list[dict[str, object]], key: str) -> float:
    return round(statistics.fmean(float(row[key]) for row in rows), 6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--sample-per-hop", type=int, default=300)
    parser.add_argument("--max-paths", type=int, default=100_000)
    parser.add_argument("--dev-examples-per-template", type=int, default=3)
    parser.add_argument("--minimum-overall-group", type=int, default=100)
    parser.add_argument("--minimum-hop-stratum", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    if not 1 <= args.sample_per_hop <= 300:
        raise ValueError("sample-per-hop must be in [1, 300]")
    if args.max_paths != 100_000:
        raise ValueError("Frozen protocol requires max_paths=100000")
    if args.dev_examples_per_template != 3:
        raise ValueError("Frozen audit requires three dev checks per template")
    if args.minimum_overall_group != 100 or args.minimum_hop_stratum != 30:
        raise ValueError("Frozen group-size gates are 100 overall and 30 per hop")
    if args.output_json.resolve().exists() or args.output_csv.resolve().exists():
        raise FileExistsError("Refusing to overwrite an OOD audit output")

    started = time.perf_counter()
    reference_path = args.reference_results.resolve()
    manifest_path = args.sample_manifest.resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if reference.get("status") != "PASS" or manifest.get("status") != "PASS":
        raise ValueError("Reference or manifest status is not PASS")
    if manifest.get("source_result_sha256") != reference_sha:
        raise ValueError("Manifest/reference SHA mismatch")
    if reference.get("configuration", {}).get("seed") != 20260818:
        raise ValueError("Reference seed mismatch")

    results = reference.get("results", [])
    by_id: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        by_id[str(row["id"])].append(row)
    all_manifest_rows = manifest.get("questions", [])
    all_manifest_ids = [str(row["id"]) for row in all_manifest_rows]
    if len(all_manifest_ids) != 900 or len(set(all_manifest_ids)) != 900:
        raise ValueError("Frozen sample manifest cardinality mismatch")
    manifest_rows = []
    for hop in (1, 2, 3):
        hop_rows = [row for row in all_manifest_rows if int(row["hop"]) == hop]
        if len(hop_rows) != 300:
            raise ValueError(f"Frozen sample hop-{hop} cardinality mismatch")
        manifest_rows.extend(hop_rows[: args.sample_per_hop])
    manifest_ids = [str(row["id"]) for row in manifest_rows]
    if any(len(by_id[identifier]) != 6 for identifier in manifest_ids):
        raise ValueError("Reference does not contain six conditions per frozen ID")

    data_dir = args.data_dir.resolve()
    adjacency = load_kg(data_dir / "kg.tsv")
    dev_rows = []
    for hop in (1, 2, 3):
        dev_rows.extend(load_jsonl(data_dir / f"{hop}-hop" / "dev.jsonl"))
    dev_templates: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in dev_rows:
        dev_templates[question_template(str(row["question"]))].append(row)

    dev_template_audits: dict[str, dict[str, object]] = {}
    observed_relation_signatures: set[str] = set()
    for index, template in enumerate(sorted(dev_templates), start=1):
        checks = []
        for row in dev_templates[template][: args.dev_examples_per_template]:
            derived = derive_signature(adjacency, row, args.max_paths)
            checks.append(
                {
                    "id": row["id"],
                    "hop": row["hop"],
                    **derived,
                }
            )
        signatures = sorted(
            {str(check["signature"]) for check in checks if check["signature"]}
        )
        consistent = len(signatures) <= 1
        dev_template_audits[template] = {
            "dev_count": len(dev_templates[template]),
            "checked": len(checks),
            "signatures": signatures,
            "consistent": consistent,
            "checks": checks,
        }
        # Only signatures from semantically stable surface templates define the
        # in-distribution relation set.  Ambiguous templates remain auditable but
        # are not forced into either the observed or shifted relation stratum.
        if consistent:
            observed_relation_signatures.update(signatures)
        print(
            f"dev_template={index}/{len(dev_templates)} checks={len(checks)} "
            f"consistent={consistent}",
            flush=True,
        )

    audit_rows: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "seed": 20260818,
            "split": "test",
            "sample_per_hop": args.sample_per_hop,
            "max_paths": args.max_paths,
            "question_template": "lowercase + bracketed entity -> [ENTITY] + whitespace collapse",
            "relation_signature": "among exact-hop chains reaching at least one gold endpoint, maximize direction-aware relation BM25 against the question, then endpoint-set F1, then lexicographically",
            "dev_template_signature_validation": "first three ordered dev examples per normalized template",
            "minimum_overall_group": args.minimum_overall_group,
            "minimum_hop_stratum": args.minimum_hop_stratum,
            "analysis_status": "prospectively_frozen_internal_protocol_20260821",
        },
        "sources": {
            "reference_results": str(reference_path),
            "reference_results_sha256": reference_sha,
            "sample_manifest": str(manifest_path),
            "sample_manifest_sha256": manifest_sha,
        },
        "dev_template_audits": dev_template_audits,
        "rows": audit_rows,
    }
    atomic_json(args.output_json.resolve(), payload)

    reference_question = {
        identifier: by_id[identifier][0] for identifier in manifest_ids
    }
    for index, manifest_row in enumerate(manifest_rows, start=1):
        identifier = str(manifest_row["id"])
        row = reference_question[identifier]
        template = question_template(str(row["question"]))
        derived = derive_signature(adjacency, row, args.max_paths)
        signature = str(derived["signature"])
        template_seen = template in dev_templates
        template_consistent = bool(
            template_seen and dev_template_audits[template]["consistent"]
        )
        if not template_seen:
            relation_status = "template_shifted"
        elif not template_consistent:
            relation_status = "ambiguous_dev_template"
        elif bool(signature) and signature in observed_relation_signatures:
            relation_status = "observed"
        else:
            relation_status = "shifted"
        audit_rows.append(
            {
                "id": identifier,
                "hop": int(row["hop"]),
                "line_number": int(row["line_number"]),
                "question_template": template,
                "template_seen_in_dev": template_seen,
                "dev_template_relation_consistent": template_consistent,
                "relation_signature": signature,
                "relation_signature_status": relation_status,
                "relation_signature_seen_in_dev": relation_status == "observed",
                **{key: value for key, value in derived.items() if key != "signature"},
            }
        )
        if index % args.checkpoint_every == 0 or index == len(manifest_rows):
            atomic_json(args.output_json.resolve(), payload)
            print(
                f"test_checkpoint={index}/{len(manifest_rows)}", flush=True
            )

    audit_by_id = {str(row["id"]): row for row in audit_rows}
    enriched_results = []
    for row in results:
        identifier = str(row["id"])
        if identifier not in audit_by_id:
            continue
        audit = audit_by_id[identifier]
        enriched_results.append(
            {
                "id": identifier,
                "hop": int(row["hop"]),
                "condition": row["condition"],
                "f1": float(row["f1"]),
                "hits_at_1": float(row["hits_at_1"]),
                "evidence_answer_recall": float(row["evidence_answer_recall"]),
                "template_seen_in_dev": bool(audit["template_seen_in_dev"]),
                "dev_template_relation_consistent": bool(
                    audit["dev_template_relation_consistent"]
                ),
                "relation_signature_status": audit["relation_signature_status"],
                "relation_signature_seen_in_dev": bool(
                    audit["relation_signature_seen_in_dev"]
                ),
            }
        )

    group_counts: dict[str, int] = {}
    group_results: list[dict[str, object]] = []
    group_specs = {
        "template_observed": ("template_seen_in_dev", True),
        "template_shifted": ("template_seen_in_dev", False),
        "relation_observed": ("relation_signature_status", "observed"),
        "relation_shifted": ("relation_signature_status", "shifted"),
        "relation_ambiguous": (
            "relation_signature_status",
            "ambiguous_dev_template",
        ),
    }
    for group, (field, target) in group_specs.items():
        question_ids = {
            str(row["id"])
            for row in audit_rows
            if (
                bool(row[field]) is target
                if isinstance(target, bool)
                else row[field] == target
            )
        }
        group_counts[group] = len(question_ids)
        for hop_label, hop_value in (("all", None), ("1", 1), ("2", 2), ("3", 3)):
            ids = {
                identifier
                for identifier in question_ids
                if hop_value is None or int(audit_by_id[identifier]["hop"]) == hop_value
            }
            threshold = (
                args.minimum_overall_group
                if hop_value is None
                else args.minimum_hop_stratum
            )
            publishable = len(ids) >= threshold
            for condition in sorted({str(row["condition"]) for row in results}):
                subset = [
                    row
                    for row in enriched_results
                    if str(row["id"]) in ids and row["condition"] == condition
                ]
                group_results.append(
                    {
                        "group": group,
                        "hop": hop_label,
                        "condition": condition,
                        "questions": len(ids),
                        "publishable_comparison": publishable,
                        "macro_f1": mean(subset, "f1") if subset else None,
                        "hits_at_1": mean(subset, "hits_at_1") if subset else None,
                        "mean_evidence_answer_recall": (
                            mean(subset, "evidence_answer_recall") if subset else None
                        ),
                    }
                )

    payload["group_counts"] = group_counts
    payload["group_results"] = group_results
    payload["quality"] = {
        "dev_templates": len(dev_templates),
        "reliable_dev_templates": sum(
            bool(value["consistent"]) for value in dev_template_audits.values()
        ),
        "observed_relation_signatures": len(observed_relation_signatures),
        "inconsistent_checked_dev_templates": sum(
            not bool(value["consistent"])
            for value in dev_template_audits.values()
        ),
        "test_signature_coverage": round(
            statistics.fmean(bool(row["relation_signature"]) for row in audit_rows),
            6,
        ),
        "test_mean_gold_path_recall": round(
            statistics.fmean(float(row["gold_recall"]) for row in audit_rows),
            6,
        ),
        "test_candidate_truncation_rate": round(
            statistics.fmean(bool(row["candidate_truncated"]) for row in audit_rows),
            6,
        ),
        "test_ambiguous_template_rate": round(
            statistics.fmean(
                row["relation_signature_status"] == "ambiguous_dev_template"
                for row in audit_rows
            ),
            6,
        ),
    }
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    payload["status"] = "PASS"
    atomic_json(args.output_json.resolve(), payload)

    args.output_csv.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.resolve().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_results[0]))
        writer.writeheader()
        writer.writerows(group_results)

    print("status=PASS", flush=True)
    print(f"group_counts={group_counts}", flush=True)
    print(f"quality={payload['quality']}", flush=True)
    print(f"json={args.output_json.resolve()}", flush=True)
    print(f"csv={args.output_csv.resolve()}", flush=True)
    print(f"elapsed_seconds={payload['elapsed_seconds']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"status=FAIL error={error}", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
