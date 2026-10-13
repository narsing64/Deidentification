"""Experiment 6: shared improvements, evaluated cumulatively on the VALIDATION split.

Uses the cached corrected-baseline model predictions (results/exp3_4_validation/predictions) and
re-runs only the cheap shared parts: regex, fusion and post-processing. Every candidate is applied
identically to all five model+regex systems.

Acceptance rules (declared before any candidate was measured; strict metrics, adjusted reference,
means over the five model+regex systems):
  recall-oriented candidates (A*): accept if mean dRecall >= 0 and mean dF1 >= 0 and no system loses
                                   more than 0.002 strict recall
  precision-oriented (B*):         accept if mean dF1 > 0 and mean dRecall >= -0.002 and no system
                                   loses more than 0.005 strict recall
  B2 (model score threshold):      grid 0.5..0.9; the largest threshold meeting the B* rule is taken
An accepted candidate stays in the configuration for the following steps; a rejected one is reverted.

    python scripts/run_improvements.py --out results/exp6_improvements
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

from deid.config import load_yaml, resolve_path
from deid.detectors.regex_detector import RegexDetector
from deid.evaluation.benchmark import load_eval_notes, load_predictions, score_predictions, write_json
from deid.evaluation.metrics import fmt
from deid.fusion.span_merger import FusionConfig, fuse
from deid.postprocess import postprocess_final, postprocess_model

REC_MAX_SYSTEM_DROP_A, REC_MAX_MEAN_DROP_B, REC_MAX_SYSTEM_DROP_B = 0.002, 0.002, 0.005

CANDIDATES = [
    ("A6", "time-only DATE -> TIME (same rule as the reference)", "A", lambda c: c["postprocess"].__setitem__("time_only_date_to_time", True)),
    ("A2", "strip leading titles from PERSON spans", "A", lambda c: c["postprocess"].__setitem__("strip_person_titles", True)),
    ("A1", "extend PERSON over adjacent name words", "A", lambda c: c["postprocess"].__setitem__("extend_person_names", True)),
    ("A3", "regex: 4-digit year in an age slot is a DATE", "A", lambda c: c["regex"]["optional_rules"].extend(["year_as_age", "year_as_age_old"])),
    ("A4", "regex: '+' international phone numbers", "A", lambda c: c["regex"]["optional_rules"].append("phone_intl")),
    ("A7", "regex: address line ending in a postal code", "A", lambda c: c["regex"]["optional_rules"].append("address_zip")),
    ("B3", "regex: ORGANIZATION needs >= 3 words", "B", lambda c: c["regex"].__setitem__("org_min_words", 3)),
]
B2_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]


def evaluate(cfg, notes, model_preds):
    regex = RegexDetector.from_config(cfg)
    fcfg = FusionConfig.from_config(cfg)
    rp = {n.note_id: regex.detect(n.text) for n in notes}
    res = {}
    for key, mp in model_preds.items():
        sys_preds = {}
        for n in notes:
            m = postprocess_model(mp.get(n.note_id, []), n.text, cfg)
            sys_preds[n.note_id] = postprocess_final(fuse(n.text, [rp[n.note_id], m], fcfg), n.text, cfg)
        s = score_predictions(sys_preds, notes, bootstrap=False, profile="adjusted")
        a = score_predictions(sys_preds, notes, bootstrap=False, profile="as_is")
        res[key] = {"strict": s["strict"], "agnostic": s["agnostic"], "as_is_strict": a["strict"]}
    return res


def mean(xs):
    return sum(xs) / len(xs)


def deltas(new, old):
    keys = list(new)
    d_r = {k: new[k]["strict"]["recall"] - old[k]["strict"]["recall"] for k in keys}
    d_f = {k: new[k]["strict"]["f1"] - old[k]["strict"]["f1"] for k in keys}
    d_p = {k: new[k]["strict"]["precision"] - old[k]["strict"]["precision"] for k in keys}
    return d_r, d_p, d_f


def accepted(kind, d_r, d_f):
    if kind == "A":
        return mean(d_r.values()) >= 0 and mean(d_f.values()) >= 0 and min(d_r.values()) >= -REC_MAX_SYSTEM_DROP_A
    return mean(d_f.values()) > 0 and mean(d_r.values()) >= -REC_MAX_MEAN_DROP_B and min(d_r.values()) >= -REC_MAX_SYSTEM_DROP_B


def summary_row(step, desc, res, d=None, decision=""):
    r = mean([v["strict"]["recall"] for v in res.values()])
    p = mean([v["strict"]["precision"] for v in res.values()])
    f = mean([v["strict"]["f1"] for v in res.values()])
    ar = mean([v["agnostic"]["recall"] for v in res.values()])
    per = " / ".join(f"{v['strict']['recall']:.3f}" for v in res.values())
    dd = ""
    if d:
        d_r, d_p, d_f = d
        dd = f"{mean(d_r.values()):+.4f} | {mean(d_p.values()):+.4f} | {mean(d_f.values()):+.4f}"
    else:
        dd = " | | "
    return f"| {step} | {desc} | {r:.4f} | {p:.4f} | {f:.4f} | {ar:.4f} | {dd} | {per} | {decision} |"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-run", default="results/exp3_4_validation")
    ap.add_argument("--out", default="results/exp6_improvements")
    args = ap.parse_args(argv)
    keys = [m["key"] for m in load_yaml("configs/models.yaml")["shortlist"]]
    notes = load_eval_notes("validation")
    run = resolve_path(args.baseline_run)
    model_preds = {k: load_predictions(run / "predictions" / f"{k}.validation.jsonl")[0] for k in keys}

    cfg = load_yaml("configs/detection.yaml")
    cfg.setdefault("postprocess", {})
    cfg["regex"].setdefault("optional_rules", [])
    for k in ("min_model_score", "time_only_date_to_time", "strip_person_titles", "extend_person_names"):
        cfg["postprocess"][k] = None if k == "min_model_score" else False
    cfg["regex"]["optional_rules"], cfg["regex"]["org_min_words"] = [], 2

    base = evaluate(cfg, notes, model_preds)
    log = [{"step": "baseline", "config": copy.deepcopy(cfg), "results": base}]
    rows = [summary_row("S0", "corrected baseline", base, decision="—")]
    cur = base
    for step, desc, kind, apply in CANDIDATES:
        trial = copy.deepcopy(cfg)
        apply(trial)
        res = evaluate(trial, notes, model_preds)
        d = deltas(res, cur)
        ok = accepted(kind, d[0], d[2])
        rows.append(summary_row(step, desc, res, d, "**accepted**" if ok else "rejected"))
        log.append({"step": step, "desc": desc, "accepted": ok, "config": trial, "results": res,
                    "delta_recall": d[0], "delta_precision": d[1], "delta_f1": d[2]})
        print(rows[-1], flush=True)
        if ok:
            cfg, cur = trial, res
    best = None
    for thr in B2_GRID:
        trial = copy.deepcopy(cfg)
        trial["postprocess"]["min_model_score"] = thr
        res = evaluate(trial, notes, model_preds)
        d = deltas(res, cur)
        ok = accepted("B", d[0], d[2])
        rows.append(summary_row(f"B2@{thr}", f"drop model spans with score < {thr}", res, d, "meets rule" if ok else "fails rule"))
        log.append({"step": f"B2@{thr}", "accepted": ok, "config": trial, "results": res,
                    "delta_recall": d[0], "delta_precision": d[1], "delta_f1": d[2]})
        print(rows[-1], flush=True)
        if ok:
            best = (thr, trial, res)
    if best:
        cfg, cur = best[1], best[2]
        rows.append(f"| B2 | threshold chosen: {best[0]} | | | | | | | | **accepted** |")
    else:
        rows.append("| B2 | no threshold meets the rule | | | | | | | | rejected |")

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final_rows = ["| System | Strict P | Strict R | Strict F1 | Agnostic R | As-is strict P |", "|---|---|---|---|---|---|"]
    for k in keys:
        b, f = base[k], cur[k]
        final_rows.append(f"| {k}+regex | {fmt(b['strict']['precision'],3)} → {fmt(f['strict']['precision'],3)} | "
                          f"{fmt(b['strict']['recall'],3)} → {fmt(f['strict']['recall'],3)} | {fmt(b['strict']['f1'],3)} → {fmt(f['strict']['f1'],3)} | "
                          f"{fmt(b['agnostic']['recall'],3)} → {fmt(f['agnostic']['recall'],3)} | "
                          f"{fmt(b['as_is_strict']['precision'],3)} → {fmt(f['as_is_strict']['precision'],3)} |")
    md = ["# Experiment 6 — shared improvements (validation, adjusted reference, strict; measured)", "",
          "Means over the five model+regex systems. Per-system recall in model order: " + ", ".join(keys) + ".", "",
          "| Step | Candidate | Mean R | Mean P | Mean F1 | Mean agnostic R | ΔR | ΔP | ΔF1 | Per-system R | Decision |",
          "|---|---|---|---|---|---|---|---|---|---|---|", *rows, "",
          "## Corrected baseline → after accepted improvements", "", *final_rows, "",
          "## Accepted configuration", "", "```json",
          json.dumps({"regex": cfg["regex"], "postprocess": cfg["postprocess"]}, indent=2), "```"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    write_json(out / "steps.json", log)
    write_json(out / "accepted_config.json", {"regex": cfg["regex"], "postprocess": cfg["postprocess"]})
    print("\n".join(md[-(len(final_rows) + 8):]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
