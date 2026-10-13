"""Experiment 6, model side (M1): decision rule for turning word scores into PHI tags.

Re-decodes the dumped per-word scores (no model re-run) with
  argmax (current)  |  o_threshold t: a word is PHI if P(O) < t, labelled with its best non-O label
and evaluates every model alone and fused with the shared regex, through the same post-processing.

Acceptance rule (declared before measuring; same "A" rule as the other recall-oriented candidates):
  relative to the current rule (argmax, with the accepted B2 score filter), accept a threshold if mean
  strict recall over the five model+regex systems does not fall, mean strict F1 does not fall, and no
  system loses more than 0.002 strict recall. Among accepted thresholds the one with the highest mean
  recall is chosen (ties: higher mean F1). Variants with the B2 score filter switched off are also
  reported, because a recall-oriented decision rule lowers word confidences.

    python scripts/sweep_decision.py --run results/exp6_validation_final
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

from deid.config import load_labels, load_yaml, resolve_path
from deid.detectors.hf_detector import WordScores, entities_from_word_scores
from deid.detectors.regex_detector import RegexDetector
from deid.evaluation.benchmark import load_eval_notes, score_predictions, write_json
from deid.fusion.span_merger import FusionConfig, fuse
from deid.postprocess import postprocess_final, postprocess_model

GRID = [("argmax", None), ("o_threshold", 0.6), ("o_threshold", 0.7), ("o_threshold", 0.8), ("o_threshold", 0.9),
        ("o_threshold", 0.95)]


def load_scores(path):
    meta, rows = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "_meta" in r:
                meta = r["_meta"]
            else:
                rows[r["note_id"]] = WordScores.from_dict(r["ws"])
    meta["tags"] = [tuple(t) for t in meta["tags"]]
    return meta, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="results/exp6_validation_final")
    ap.add_argument("--out", default="results/exp6_model_decoding")
    args = ap.parse_args(argv)
    run, out = resolve_path(args.run), resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det_cfg = load_yaml("configs/detection.yaml")
    labels = load_labels()
    models = load_yaml("configs/models.yaml")["shortlist"]
    notes = load_eval_notes("validation")
    regex = RegexDetector.from_config(det_cfg)
    rp = {n.note_id: regex.detect(n.text) for n in notes}
    fcfg = FusionConfig.from_config(det_cfg)
    mc = det_cfg.get("model", {})
    age_min = det_cfg["regex"]["age_min"] if mc.get("age_scope", True) else None

    results = {}
    for decision, t in GRID:
        for b2 in (True, False):
            cfg = copy.deepcopy(det_cfg)
            if not b2:
                cfg["postprocess"]["min_model_score"] = None
            name = f"{decision}{'' if t is None else '@' + str(t)}{'' if b2 else ' (B2 off)'}"
            results[name] = {}
            for m in models:
                meta, scores = load_scores(run / "wordscores" / f"{m['key']}.validation.jsonl")
                only, comb = {}, {}
                for n in notes:
                    ws = scores.get(n.note_id)
                    ents = [] if ws is None else entities_from_word_scores(
                        n.text, ws, tags=meta["tags"], scheme=meta["scheme"],
                        label_map=labels["models"][meta["label_schema"]], o_index=meta["o_index"],
                        decision=decision, o_threshold=t or 0.5, merge_categories=mc.get("merge_adjacent_categories") or (),
                        age_min=age_min, source=m["key"], normalize_text=bool(cfg.get("normalize_for_detection", True)))
                    ents = postprocess_model(ents, n.text, cfg)
                    only[n.note_id] = postprocess_final(ents, n.text, cfg)
                    comb[n.note_id] = postprocess_final(fuse(n.text, [rp[n.note_id], ents], fcfg), n.text, cfg)
                so = score_predictions(only, notes, bootstrap=False)
                sc = score_predictions(comb, notes, bootstrap=False)
                results[name][m["key"]] = {"model_only": {k: so[k] for k in ("strict", "agnostic")},
                                           "combined": {k: sc[k] for k in ("strict", "agnostic")}}
            print(name, "done", flush=True)

    keys = [m["key"] for m in models]
    mean = lambda name, part, met: sum(results[name][k][part]["strict"][met] for k in keys) / len(keys)
    base = "argmax"
    accepted = []
    for name in results:
        if name == base:
            continue
        dr = {k: results[name][k]["combined"]["strict"]["recall"] - results[base][k]["combined"]["strict"]["recall"] for k in keys}
        ok = (mean(name, "combined", "recall") >= mean(base, "combined", "recall")
              and mean(name, "combined", "f1") >= mean(base, "combined", "f1") and min(dr.values()) >= -0.002)
        results[name]["_accepted"] = ok
        if ok:
            accepted.append(name)
    choice = max(accepted, key=lambda n: (mean(n, "combined", "recall"), mean(n, "combined", "f1"))) if accepted else base

    md = ["# Experiment 6 (model side) — decision rule sweep (validation, adjusted reference, strict; measured)", "",
          "Re-decoded from dumped word scores; regex, fusion and post-processing as accepted in Experiment 6.", "",
          "| Rule | Model-only mean R | Model-only mean P | Combined mean R | Combined mean P | Combined mean F1 | Per-model combined R | Decision |",
          "|---|---|---|---|---|---|---|---|"]
    for name in results:
        per = " / ".join(f"{results[name][k]['combined']['strict']['recall']:.3f}" for k in keys)
        dec = "current" if name == base else ("**chosen**" if name == choice else ("meets rule" if results[name].get("_accepted") else "fails rule"))
        md.append(f"| {name} | {mean(name, 'model_only', 'recall'):.4f} | {mean(name, 'model_only', 'precision'):.4f} | "
                  f"{mean(name, 'combined', 'recall'):.4f} | {mean(name, 'combined', 'precision'):.4f} | "
                  f"{mean(name, 'combined', 'f1'):.4f} | {per} | {dec} |")
    md += ["", "Per-model model-only strict recall (argmax → best o_threshold):", ""]
    for k in keys:
        best = max((n for n in results if not n.endswith("(B2 off)")), key=lambda n: results[n][k]["model_only"]["strict"]["recall"])
        md.append(f"- {k}: {results[base][k]['model_only']['strict']['recall']:.3f} → {results[best][k]['model_only']['strict']['recall']:.3f} ({best})")
    md += ["", f"Decision: **{choice}**"]
    (out / "decision_sweep.md").write_text("\n".join(md), encoding="utf-8")
    write_json(out / "decision_sweep.json", {"results": results, "choice": choice})
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
