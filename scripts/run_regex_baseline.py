"""Experiment 2: regex-only baseline on the validation split.

    python scripts/run_regex_baseline.py --out results/exp2_regex
"""
from __future__ import annotations

import argparse
import sys
import time

from deid.config import load_labels, load_yaml, resolve_path
from deid.detectors.regex_detector import RegexDetector
from deid.evaluation.benchmark import evaluate_system, load_eval_notes, run_detector, save_predictions, write_json
from deid.evaluation.error_analysis import render_markdown
from deid.evaluation.report import confusion_table, error_table, headline_table, per_category_table
from deid.evaluation.resources import PeakMemory, latency_summary, machine_info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation", choices=["validation"])
    ap.add_argument("--detection-config", default="configs/detection.yaml")
    ap.add_argument("--out", default="results/exp2_regex")
    args = ap.parse_args(argv)

    det_cfg = load_yaml(args.detection_config)
    notes = load_eval_notes(args.split)
    with PeakMemory() as pm:
        t0 = time.perf_counter()
        det = RegexDetector.from_config(det_cfg)
        init_s = time.perf_counter() - t0
        preds, secs = run_detector(det.detect, notes)
    out = resolve_path(args.out)
    save_predictions(out / "predictions" / f"regex.{args.split}.jsonl", preds, secs, {"system": "regex"})
    res = evaluate_system(preds, notes)
    resources = {"init_s": init_s, "peak_rss_bytes": pm.peak, "download_bytes": 0, "disk_bytes": 0,
                 "latency": latency_summary([secs[n.note_id] for n in notes], [len(n.text) for n in notes])}
    rule_counts: dict[str, int] = {}
    for ents in preds.values():
        for e in ents:
            rule_counts[e.fine_label] = rule_counts.get(e.fine_label, 0) + 1
    write_json(out / f"metrics.regex.{args.split}.json",
               {"system": "regex", "split": args.split, "metrics": res, "resources": resources,
                "rule_match_counts": rule_counts, "detection_config": det_cfg, "machine": machine_info()})
    cats = load_labels()["categories"]
    rows = [("regex — adjusted ref", res["adjusted"]), ("regex — as-is ref", res["as_is"])]
    md = [f"# Experiment 2 — regex-only baseline ({args.split}, {len(notes)} notes)", "",
          "Reference profiles: *adjusted* applies coverage rules R1/R2, *as-is* does not "
          "(docs/04_reference_and_evaluation_protocol.md).", "",
          "## Headline (micro-averaged; 95% bootstrap CI on recall)", headline_table(rows), "",
          "## Per category (strict)", per_category_table(rows, cats), "",
          "## Error breakdown (strict)", error_table(rows), "",
          "## Confusion, adjusted reference (exact-span matches)", confusion_table(res["adjusted"], cats), "",
          "## Sanitised error analysis (adjusted reference)", render_markdown(res["errors_adjusted"], "regex"), "",
          f"Latency: median {resources['latency']['median_s']*1000:.1f} ms/note, "
          f"p95 {resources['latency']['p95_s']*1000:.1f} ms/note; peak RSS {pm.peak/2**20:.0f} MB.", ""]
    (out / f"report.regex.{args.split}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:14]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
