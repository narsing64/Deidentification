"""Measure a regex / post-processing update against saved predictions (no model re-run).

Baseline = the saved regex predictions of a finished run; update = the CURRENT regex code and
configs/detection.yaml. Model predictions are reused, so the only difference is the shared rule layer.
Reports regex-only and model+regex for every model, both reference profiles, plus per-category recall.

    python scripts/evaluate_rule_update.py --run results/exp6_model_decoding/no_normalization --split validation
"""
from __future__ import annotations

import argparse
import json
import sys

from deid.config import load_yaml, resolve_path
from deid.detectors.regex_detector import RegexDetector
from deid.evaluation.benchmark import load_eval_notes, load_predictions, score_predictions
from deid.evaluation.metrics import fmt
from deid.fusion.span_merger import FusionConfig, fuse
from deid.postprocess import postprocess_final, postprocess_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--out", default="results/exp10_rules_v2")
    args = ap.parse_args(argv)
    run = resolve_path(args.run)
    cfg = load_yaml("configs/detection.yaml")
    notes = load_eval_notes(args.split, allow_test=args.allow_test, purpose="rule update v2 (post-hoc report)")
    keys = args.models or [m["key"] for m in load_yaml("configs/models.yaml")["shortlist"]]
    old_rp, _, _ = load_predictions(run / "predictions" / f"regex.{args.split}.jsonl")
    det = RegexDetector.from_config(cfg)
    new_rp = {n.note_id: det.detect(n.text) for n in notes}
    fcfg = FusionConfig.from_config(cfg)

    def system(rp, mp):
        out = {}
        for n in notes:
            m = postprocess_model(mp.get(n.note_id, []), n.text, cfg) if mp is not None else []
            srcs = [rp.get(n.note_id, [])] + ([m] if mp is not None else [])
            out[n.note_id] = postprocess_final(fuse(n.text, srcs, fcfg), n.text, cfg)
        return out

    rows, per_cat = [], {}
    pairs = [("regex", None)] + [(k, load_predictions(run / "predictions" / f"{k}.{args.split}.jsonl")[0])
                                  for k in keys if (run / "predictions" / f"{k}.{args.split}.jsonl").exists()]
    for name, mp in pairs:
        label = name if mp is None else f"{name}+regex"
        res = {}
        for tag, rp in (("v1", old_rp), ("v2", new_rp)):
            preds = system(rp, mp)
            res[tag] = {p: score_predictions(preds, notes, bootstrap=False, profile=p) for p in ("adjusted", "as_is")}
        rows.append((label, res))
        per_cat[label] = {tag: {c: v["recall"] for c, v in res[tag]["adjusted"]["per_category"].items()
                                if v["tp"] + v["fn"] > 0} for tag in ("v1", "v2")}
        print(label, "done", flush=True)

    md = [f"# Rule update v2 — effect on the {args.split} split (measured; model predictions reused)", "",
          "| System | Strict P v1 → v2 | Strict R v1 → v2 | Strict F1 v1 → v2 | Agnostic R v1 → v2 | As-is strict P v1 → v2 |",
          "|---|---|---|---|---|---|"]
    for label, res in rows:
        a1, a2 = res["v1"]["adjusted"], res["v2"]["adjusted"]
        s1, s2 = res["v1"]["as_is"]["strict"], res["v2"]["as_is"]["strict"]
        md.append(f"| {label} | {fmt(a1['strict']['precision'], 4)} → {fmt(a2['strict']['precision'], 4)} | "
                  f"{fmt(a1['strict']['recall'], 4)} → {fmt(a2['strict']['recall'], 4)} | "
                  f"{fmt(a1['strict']['f1'], 4)} → {fmt(a2['strict']['f1'], 4)} | "
                  f"{fmt(a1['agnostic']['recall'], 4)} → {fmt(a2['agnostic']['recall'], 4)} | "
                  f"{fmt(s1['precision'], 4)} → {fmt(s2['precision'], 4)} |")
    md += ["", "## Per-category strict recall, v1 → v2 (adjusted)", ""]
    cats = sorted({c for v in per_cat.values() for c in v["v1"]})
    md += ["| System | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for label, v in per_cat.items():
        md.append(f"| {label} | " + " | ".join(f"{v['v1'].get(c, 0):.2f} → {v['v2'].get(c, 0):.2f}" for c in cats) + " |")
    rule_counts: dict[str, int] = {}
    for ents in new_rp.values():
        for e in ents:
            rule_counts[e.fine_label] = rule_counts.get(e.fine_label, 0) + 1
    md += ["", "Regex v2 rule firing counts on this split: " + ", ".join(f"{k} {v}" for k, v in sorted(rule_counts.items()))]
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"report.{args.split}.md").write_text("\n".join(md), encoding="utf-8")
    (out / f"summary.{args.split}.json").write_text(json.dumps(
        {label: {t: {p: {k: r[t][p][k] for k in ("strict", "agnostic")} for p in r[t]} for t in r} for label, r in rows},
        indent=2), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
