"""Derive reference PHI spans for the annotated notes and validate them.

Writes data/processed/gold.jsonl (offsets + labels only, no text) and a sanitised summary
(counts only) to results/00_gold/gold_summary.json.

    python scripts/build_gold.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from deid.config import load_labels, load_yaml, resolve_path
from deid.data.gold import (ReferenceLabels, coverage_adjustments, derive_reference_spans,
                            reconstruction_mismatch, validate_gold)
from deid.data.loader import load_annotation_meta, load_reference_pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--out", default=None, help="gold JSONL path (default: <processed_dir>/gold.jsonl)")
    ap.add_argument("--summary", default="results/00_gold/gold_summary.json")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    data = cfg["data"]
    pairs, pair_report = load_reference_pairs(resolve_path(data["annotated_csv"]))
    meta, meta_report = load_annotation_meta(resolve_path(data["annotations_jsonl"]))
    labels = ReferenceLabels(load_labels())

    out_path = Path(args.out) if args.out else resolve_path(data["processed_dir"]) / "gold.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    by_label: Counter = Counter()
    by_fine: Counter = Counter()
    problems: Counter = Counter()
    mismatch_tokens: list[int] = []
    missing_meta = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in pairs:
            m = meta.get(rec["note_id"])
            if m is None:
                missing_meta += 1
                m = {"source": "unknown", "tier": "unknown", "possible_leaks": []}
            doc = derive_reference_spans(rec["note_id"], rec["raw"], rec["reference"], labels, m["possible_leaks"])
            coverage_adjustments(rec["raw"], doc)
            for p in validate_gold(rec["raw"], doc):
                problems[p] += 1
            mismatch_tokens.append(reconstruction_mismatch(rec["raw"], rec["reference"], doc))
            stats.update(doc.stats)
            by_label.update(e.label for e in doc.spans)
            by_fine.update(e.fine_label for e in doc.spans)
            row = doc.to_dict()
            row.update({"source": m["source"], "tier": m["tier"], "n_chars": len(rec["raw"])})
            f.write(json.dumps(row) + "\n")

    mm = sorted(mismatch_tokens)
    summary = {
        "notes": len(pairs),
        "notes_missing_metadata": missing_meta,
        "loader_reports": [pair_report.to_dict(), meta_report.to_dict()],
        "derivation_stats": dict(sorted(stats.items())),
        "spans_by_normalised_label": dict(by_label.most_common()),
        "spans_by_fine_label": dict(by_fine.most_common()),
        "structural_problems": dict(problems),
        "reconstruction_unexplained_reference_tokens": {
            "total": sum(mm), "notes_with_zero": sum(1 for x in mm if x == 0),
            "median": mm[len(mm) // 2] if mm else 0, "max": mm[-1] if mm else 0,
        },
        "output": str(out_path.relative_to(resolve_path("."))) if out_path.is_relative_to(resolve_path(".")) else out_path.name,
    }
    summary_path = resolve_path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("notes", "derivation_stats", "spans_by_normalised_label",
                                               "structural_problems", "reconstruction_unexplained_reference_tokens")}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
