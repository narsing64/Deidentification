"""Experiment 5 helper: which system fails to detect which PHI category, and how.

For every system (regex, each model alone, each model + regex) and every category:
reference count, strict recall, and the false negatives split into
  missed (no overlapping prediction) | boundary (overlap, wrong span) | wrong_label (exact span),
plus the most frequent sanitised shapes of missed spans. Output: counts and shapes only.

    python scripts/analyze_failures.py --run results/exp3_4_validation
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

from deid.config import load_labels, load_yaml, resolve_path
from deid.evaluation.benchmark import filter_excluded, load_eval_notes, load_predictions, profile_view
from deid.fusion.span_merger import FusionConfig, fuse
from deid.privacy import safe_shape


def failures(notes, preds):
    per = defaultdict(Counter)
    shapes = defaultdict(Counter)
    confused = defaultdict(Counter)
    for n in notes:
        gold, ign = profile_view(n, "adjusted")
        pr = [p for p in filter_excluded(n, preds.get(n.note_id, []), "adjusted")
              if not any(p.start < r.end and r.start < p.end for r in ign)]
        keys = {(p.start, p.end, p.label) for p in pr}
        for g in gold:
            c = per[g.label]
            c["ref"] += 1
            if (g.start, g.end, g.label) in keys:
                c["tp"] += 1
                continue
            ov = [p for p in pr if p.start < g.end and g.start < p.end]
            exact = [p for p in ov if (p.start, p.end) == (g.start, g.end)]
            if exact:
                c["wrong_label"] += 1
                confused[g.label][exact[0].label] += 1
            elif ov:
                c["boundary"] += 1
            else:
                c["missed"] += 1
                shapes[g.label][(g.fine_label, safe_shape(n.text[g.start:g.end], 24))] += 1
    return per, shapes, confused


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="results/exp3_4_validation")
    ap.add_argument("--split", default="validation", choices=["validation"])
    args = ap.parse_args(argv)
    run = resolve_path(args.run)
    cats = load_labels()["categories"]
    keys = [m["key"] for m in load_yaml("configs/models.yaml")["shortlist"]]
    notes = load_eval_notes(args.split)
    fcfg = FusionConfig.from_config(load_yaml("configs/detection.yaml"))
    rp, _, _ = load_predictions(run / "predictions" / f"regex.{args.split}.jsonl")
    systems = {"regex": rp}
    for k in keys:
        mp, _, _ = load_predictions(run / "predictions" / f"{k}.{args.split}.jsonl")
        systems[k] = mp
        systems[f"{k}+regex"] = {n.note_id: fuse(n.text, [rp.get(n.note_id, []), mp.get(n.note_id, [])], fcfg)
                                 for n in notes}
    out, md = {}, [f"# Which system misses which category ({args.split}, adjusted reference, strict)", ""]
    results = {name: failures(notes, p) for name, p in systems.items()}
    present = [c for c in cats if any(results["regex"][0][c]["ref"] for _ in [0])]
    md += ["## Strict recall by category (n = reference spans)", "",
           "| System | " + " | ".join(f"{c} (n={results['regex'][0][c]['ref']})" for c in present) + " |",
           "|---|" + "---|" * len(present)]
    for name, (per, _, _) in results.items():
        md.append(f"| {name} | " + " | ".join(
            f"{per[c]['tp'] / per[c]['ref']:.2f}" if per[c]["ref"] else "n/a" for c in present) + " |")
    md += ["", "## False negatives by kind: missed / boundary / wrong-label", "",
           "| System | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    for name, (per, _, _) in results.items():
        md.append(f"| {name} | " + " | ".join(
            f"{per[c]['missed']}/{per[c]['boundary']}/{per[c]['wrong_label']}" for c in present) + " |")
    md += ["", "## Most frequent shapes of completely MISSED spans", ""]
    for name, (per, shapes, confused) in results.items():
        md.append(f"**{name}**")
        for c in present:
            if shapes[c]:
                top = "; ".join(f"{fine} `{sh}` ×{k}" for (fine, sh), k in shapes[c].most_common(4))
                md.append(f"- {c}: {top}")
            if confused[c]:
                md.append(f"- {c} found with wrong label: " + ", ".join(f"{l} ×{k}" for l, k in confused[c].most_common()))
        md.append("")
        out[name] = {c: dict(per[c]) for c in present}
    (run / f"category_failures.{args.split}.md").write_text("\n".join(md), encoding="utf-8")
    (run / f"category_failures.{args.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
