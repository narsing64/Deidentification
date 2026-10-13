"""Create the fixed validation / held-out test split (note ids only).

Notes sharing a patient name are kept together. Names are hashed in memory only; neither the names
nor their hashes are written anywhere.

    python scripts/make_splits.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from deid.config import load_yaml, resolve_path
from deid.data.gold import GoldDoc
from deid.data.loader import load_reference_pairs
from deid.data.splits import group_notes, stratified_group_split
from deid.privacy import unkeyed_hash


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split (never after test use)")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    processed = resolve_path(cfg["data"]["processed_dir"])
    out = processed / "splits.json"
    if out.exists() and not args.force:
        print(f"{out.name} already exists; refusing to overwrite (use --force before any test-set use).")
        return 1

    gold_rows = [json.loads(l) for l in open(processed / "gold.jsonl", encoding="utf-8")]
    texts = {r["note_id"]: r["raw"] for r in load_reference_pairs(resolve_path(cfg["data"]["annotated_csv"]))[0]}
    keys: dict[str, set[str]] = {}
    if cfg["split"].get("group_by_patient_name_hash", True):
        for row in gold_rows:
            doc = GoldDoc.from_dict(row)
            raw = texts[row["note_id"]]
            keys[row["note_id"]] = {
                unkeyed_hash(" ".join(raw[e.start:e.end].lower().split()))
                for e in doc.spans if e.fine_label == "PATIENT_NAME"
            }
    ids = [r["note_id"] for r in gold_rows]
    groups = group_notes(ids, keys)
    sp = cfg["split"]
    split = stratified_group_split(gold_rows, groups, sp["validation_fraction"], sp["seed"], sp["stratify_by"])

    info = {r["note_id"]: r for r in gold_rows}
    def describe(part: list[str]) -> dict:
        return {
            "notes": len(part),
            "by_stratum": dict(Counter(f'{info[i]["source"]}/{info[i]["tier"]}' for i in part)),
            "reference_spans": sum(len(info[i]["spans"]) for i in part),
            "spans_by_label": dict(Counter(s["label"] for i in part for s in info[i]["spans"]).most_common()),
        }
    result = {
        "seed": sp["seed"],
        "validation_fraction": sp["validation_fraction"],
        "stratify_by": sp["stratify_by"],
        "multi_note_groups": sum(1 for g in groups if len(g) > 1),
        "validation": split["validation"],
        "test": split["test"],
        "summary": {"validation": describe(split["validation"]), "test": describe(split["test"])},
    }
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"multi_note_groups": result["multi_note_groups"], **result["summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
