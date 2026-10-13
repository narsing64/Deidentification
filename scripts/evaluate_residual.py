"""Residual-PHI evaluation (Section 12) for the selected system on one split.

Rebuilds the final entity set from saved predictions with exactly the pipeline's functions
(postprocess_model -> fuse with regex -> postprocess_final), applies the configured policy, then reports:
  * validator view (no reference needed): detected / transformed / potential residual / review required
  * reference view: for every reference span, how much of it is left verbatim in the output,
    by category, using the exact original->output mapping of the transformation records
Counts only; no text is written. An ephemeral key is used (no linkage needed for evaluation).

    python scripts/evaluate_residual.py --run results/exp6_validation_final --model <key> --split validation
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
from collections import Counter

from deid.config import load_yaml, resolve_path
from deid.detectors.regex_detector import RegexDetector
from deid.evaluation.benchmark import load_eval_notes, load_predictions, profile_view
from deid.fusion.span_merger import FusionConfig, fuse
from deid.policies.transform import PolicyEngine
from deid.postprocess import postprocess_final, postprocess_model
from deid.validation.residual import ResidualReport, ResidualValidator, reference_residual


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    run = resolve_path(args.run)
    det_cfg = load_yaml("configs/detection.yaml")
    notes = load_eval_notes(args.split, allow_test=args.allow_test, purpose=f"final evaluation: residual PHI {args.model}")
    mp, _, _ = load_predictions(run / "predictions" / f"{args.model}.{args.split}.jsonl")
    rp, _, _ = load_predictions(run / "predictions" / f"regex.{args.split}.jsonl")
    fcfg = FusionConfig.from_config(det_cfg)
    engine = PolicyEngine(load_yaml("configs/deid_policy.yaml"), key=secrets.token_bytes(32))
    validator = ResidualValidator(RegexDetector.from_config(det_cfg))

    total = ResidualReport()
    by_cat = {"adjusted": {}, "as_is": {}}
    agg = {"adjusted": Counter(), "as_is": Counter()}
    notes_with_exposure = Counter()
    for n in notes:
        m = postprocess_model(mp.get(n.note_id, []), n.text, det_cfg)
        ents = postprocess_final(fuse(n.text, [rp.get(n.note_id, []), m], fcfg), n.text, det_cfg)
        res = engine.transform(n.text, ents, n.note_id)
        total.merge(validator.validate(n.text, ents, res))
        for prof in ("adjusted", "as_is"):
            gold, _ = profile_view(n, prof)
            rr = reference_residual(gold, res, n.text)
            for cat, c in rr["by_category"].items():
                by_cat[prof].setdefault(cat, Counter()).update(c)
            agg[prof].update({"spans": rr["spans"], "fully_exposed": rr["fully_exposed"],
                              "partially_exposed": rr["partially_exposed"]})
            for c in rr["by_category"].values():
                agg[prof].update({k: v for k, v in c.items() if k.startswith("partially_exposed_")})
            if rr["fully_exposed"] or rr["partially_exposed"]:
                notes_with_exposure[prof] += 1
    out = {"split": args.split, "model": args.model, "notes": len(notes),
           "validator": total.to_dict(),
           "reference_residual": {p: {"totals": dict(agg[p]), "notes_with_any_exposed_span": notes_with_exposure[p],
                                      "by_category": {k: dict(v) for k, v in sorted(by_cat[p].items())}}
                                  for p in ("adjusted", "as_is")}}
    (run / f"residual.{args.model}.{args.split}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = [f"# Residual PHI — {args.model} + regex ({args.split}, {len(notes)} notes; measured)", "",
          "Reference view: a span is *fully exposed* if none of its characters were transformed, *partially* if some were.", "",
          "Partial exposures are split into *identifying* (some name/number characters left) and *non-identifying* "
          "(only a title such as 'Dr.', a credential, punctuation or whitespace left).", "",
          "| Profile | Reference spans | Fully protected | Partially exposed (identifying / non-identifying) | Fully exposed | Notes with any exposure |",
          "|---|---|---|---|---|---|"]
    for p in ("adjusted", "as_is"):
        t = agg[p]
        md.append(f"| {p} | {t['spans']} | {t['spans'] - t['fully_exposed'] - t['partially_exposed']} | "
                  f"{t['partially_exposed']} ({t['partially_exposed_identifying']} / {t['partially_exposed_nonidentifying']}) | "
                  f"{t['fully_exposed']} | {notes_with_exposure[p]} |")
    md += ["", "| Category (adjusted) | Spans | Fully protected | Partially exposed (identifying / non-identifying) | Fully exposed |",
           "|---|---|---|---|---|"]
    for cat, c in sorted(by_cat["adjusted"].items()):
        md.append(f"| {cat} | {c['spans']} | {c.get('fully_protected', 0)} | {c.get('partially_exposed', 0)} "
                  f"({c.get('partially_exposed_identifying', 0)} / {c.get('partially_exposed_nonidentifying', 0)}) | {c.get('fully_exposed', 0)} |")
    v = out["validator"]
    md += ["", "Validator view (no reference; shares the detectors' blind spots, so it is not a guarantee):", "",
           f"- detected: {sum(v['detected'].values())} {v['detected']}",
           f"- transformed: {sum(v['transformed'].values())}; fallback (review): {v['fallback_review']}",
           f"- potential residual (regex re-scan of output): {v['potential_residual_regex']}",
           f"- potential residual (detected value repeated elsewhere, verbatim): {v['potential_residual_repeated_value']}",
           f"- review required: {v['review_required']}"]
    (run / f"residual.{args.model}.{args.split}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
