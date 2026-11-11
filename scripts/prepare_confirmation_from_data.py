#!/usr/bin/env python3
"""Reconstruct the paper's confirmation sample from processed public MetaQA.

No predictions, historical result files, model weights, or supplementary
archives are required. Exclusions are regenerated from historical sampling
rules and the preprocessor's recorded label repairs. Hashes validate the
reconstructed IDs; they are fingerprints, not bundled experimental data.
"""
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

EXCLUSION_IDS_SHA256 = "eea6325cb71d631ce8751c7a90f1cd4905e6dffed6822ec663902f0dbadf22d8"
SAMPLE_IDS_SHA256 = "9644d06715233655c1f40e0b76f733aa1f8fceaf38f7088bc00f573261091487"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ids_sha(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()

def qkey(q):
    return " ".join(q.casefold().replace("[", "").replace("]", "").split())

def sample_rows(rows, n, seed, hop):
    # Identical to the historical inference runner's per-hop sampling.
    return random.Random(seed + hop * 1000).sample(rows, min(n, len(rows)))

def reconstruct(data):
    populations = {}
    excluded = {}
    hashes = {}
    for split in ("dev", "test"):
        for hop in (1, 2, 3):
            f = data / f"{hop}-hop/{split}.jsonl"
            rows = [json.loads(line) for line in f.read_text().splitlines()]
            populations[split, hop] = rows
            hashes[str(f.relative_to(data))] = sha(f)
            configs = [(42, 20), (7, 1)] if split == "dev" else [
                (42, 300), (20260818, 300), (20260819, 300), (20260820, 300)]
            for seed, n in configs:
                for row in sample_rows(rows, n, seed, hop):
                    excluded[row["id"]] = row
            # The original archive scan also saw the preprocessing correction log.
            for row in rows:
                if row.get("label_corrections"):
                    excluded[row["id"]] = row
    ids = sorted(excluded)
    if ids_sha(ids) != EXCLUSION_IDS_SHA256:
        raise ValueError("Exclusion IDs differ from the paper; verify source data and preprocessing.")
    seen_q = {qkey(r["question"]) for r in excluded.values()}
    sampled, eligible = [], {}
    for hop in (1, 2, 3):
        rows = [r for r in populations["test", hop]
                if r["id"] not in excluded and qkey(r["question"]) not in seen_q]
        rows.sort(key=lambda r: hashlib.sha256(f"20261111:{r['id']}".encode()).hexdigest())
        unique, seen = [], set()
        for row in rows:
            key = qkey(row["question"])
            if key not in seen:
                unique.append(row)
                seen.add(key)
        eligible[str(hop)] = len(unique)
        sampled.extend(unique[:100])
    if ids_sha([r["id"] for r in sampled]) != SAMPLE_IDS_SHA256:
        raise ValueError("Confirmation sample differs from the paper; no manifest written.")
    if len({qkey(r["question"]) for r in sampled}) != 300:
        raise ValueError("Duplicate normalized questions in reconstructed sample.")
    hashes["kg.tsv"] = sha(data / "kg.tsv")
    return dict(sample=sampled, excluded_ids=ids,
                excluded_normalized_questions=len(seen_q),
                eligible_unique_by_hop=eligible, processed_data_hashes=hashes,
                overlap_ids=0, overlap_normalized_questions=0,
                selection_rule="lowest SHA256(seed:question_id), 100 unique normalized questions per hop",
                exclusion_source_hashes={},
                provenance="Reconstructed from public processed data and historical sampling rules, not original archive bytes.")

def prepare(project, output):
    manifest = reconstruct(project / "data/processed/metaqa")
    plan = dict(n_per_hop=100, seed=20261111, budget=256, output_cap=96,
                conditions="full/no_cost crossed with original/no_rep_constraints",
                main_estimand="(full-no_cost)_no_rep_constraints - (full-no_cost)_original",
                inference="20000 paired hop-stratified bootstrap; template-cluster sensitivity",
                stopping="all 300 questions and four cells; no outcome-based changes",
                scope="Same benchmark and model; templates may overlap.",
                provenance="Executable reconstruction of the original protocol; not an external preregistration.")
    plan_text = json.dumps(plan, indent=2) + "\n"
    manifest["plan_sha256"] = hashlib.sha256(plan_text.encode()).hexdigest()
    manifest["reconstructed_utc"] = datetime.now(timezone.utc).isoformat()
    # Never replace an existing sample or protocol.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(plan_text)
    (output / "sample_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {"status": "PASS", "sample_ids_sha256": SAMPLE_IDS_SHA256,
            "excluded_ids_sha256": EXCLUSION_IDS_SHA256, "questions": 300,
            "exclusion_ids": len(manifest["excluded_ids"]),
            "eligible_unique_by_hop": manifest["eligible_unique_by_hop"],
            "output": str(output)}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(prepare(a.project, a.output), indent=2))

if __name__ == "__main__":
    main()

