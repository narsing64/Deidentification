"""Evaluate the deployed system on an external / synthetic annotated dataset (Experiment 12).

Dataset format (JSONL, one note per line):
    {"note_id", "text", "spans": [{"start","end","label"}], "ignore": [{"start","end","reason"}] (optional),
     "doc_type" (optional), "split" (optional)}

Systems (detection run once per note, then combined exactly as in the pipeline):
    rules  = regex only            -> postprocess_final
    model  = selected model only   -> postprocess_model -> postprocess_final
    system = rules + model         -> fuse -> postprocess_final       (the deployed configuration)

Metrics:
  * strict / label-agnostic exact-span P, R, F1 (+ document bootstrap 95% CI), per category
  * convention-robust views:
      - span coverage: share of reference spans fully / partly covered by any prediction
      - token-level P/R/F1 (whitespace tokens; a token is PHI if it overlaps a reference span)
  * residual PHI after the configured policy is applied (fully protected / identifying leftover / fully exposed)
Nothing is tuned here: configs are used as they are.

    python scripts/evaluate_external.py --data data/synthetic_in/test.jsonl --name indian_synth_test
"""
from __future__ import annotations

import argparse
import json
import secrets
import time
from collections import Counter, defaultdict
from pathlib import Path

from deid.config import load_labels, load_yaml, resolve_path
from deid.evaluation.metrics import CorpusScore, prf, score_document
from deid.fusion.span_merger import FusionConfig, fuse
from deid.pipeline import DeidPipeline
from deid.policies.transform import PolicyEngine
from deid.postprocess import postprocess_final, postprocess_model
from deid.schema import Entity, Region
from deid.validation.residual import reference_residual
import re

TOKEN = re.compile(r"\S+")


def load(path: Path):
    notes = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        gold = [Entity(s["start"], s["end"], s["label"], source="reference") for s in r["spans"]]
        ign = [Region(g["start"], g["end"], g.get("reason", "ignore")) for g in r.get("ignore", [])]
        notes.append((r["note_id"], r["text"], gold, ign, r.get("doc_type", "")))
    return notes


def token_counts(text: str, gold, pred, ignore) -> Counter:
    c = Counter()
    for m in TOKEN.finditer(text):
        s, e = m.start(), m.end()
        if any(s < r.end and r.start < e for r in ignore):
            continue
        g = any(s < x.end and x.start < e for x in gold)
        p = any(s < x.end and x.start < e for x in pred)
        c["tp"] += g and p
        c["fp"] += p and not g
        c["fn"] += g and not p
    return c


def coverage(gold, pred) -> tuple[list[bool], list[bool]]:
    """Per reference span: (fully covered by the union of predictions, overlapped by any prediction)."""
    merged = []
    for s, e in sorted((p.start, p.end) for p in pred):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    full, part = [], []
    for g in gold:
        cov = sum(max(0, min(g.end, e) - max(g.start, s)) for s, e in merged)
        full.append(cov == g.end - g.start)
        part.append(cov > 0)
    return full, part


def evaluate(name: str, notes, preds: dict, engine: PolicyEngine | None, cats) -> dict:
    cs = CorpusScore()
    tok = Counter()
    cov_cat = defaultdict(Counter)
    cov_type = defaultdict(Counter)
    resid = Counter()
    resid_cat = defaultdict(Counter)
    for nid, text, gold, ign, dtype in notes:
        pred = preds[nid]
        cs.add(nid, score_document(gold, pred, ign))
        tok.update(token_counts(text, gold, pred, ign))
        full, part = coverage(gold, pred)
        for g, f, p in zip(gold, full, part):
            cov_cat[g.label].update({"spans": 1, "full": f, "partial_only": p and not f, "missed": not p})
            cov_type[dtype].update({"spans": 1, "full": f})
        if engine is not None:
            res = engine.transform(text, sorted(pred, key=lambda e: (e.start, e.end)), nid)
            rr = reference_residual(gold, res, text)
            for cat, c in rr["by_category"].items():
                resid_cat[cat].update(c)
                resid.update({k: v for k, v in c.items()})
    summ = cs.summary(bootstrap=1000, seed=7, categories=cats)
    t = prf(tok["tp"], tok["fp"], tok["fn"])
    out = {"system": name, "scores": summ, "token_level": {**dict(tok), **t},
           "coverage_by_category": {k: dict(v) for k, v in sorted(cov_cat.items())},
           "full_coverage_by_doc_type": {k: {"spans": v["spans"], "full": v["full"],
                                             "rate": v["full"] / v["spans"] if v["spans"] else None}
                                         for k, v in sorted(cov_type.items())}}
    if engine is not None:
        out["residual"] = {"totals": dict(resid), "by_category": {k: dict(v) for k, v in sorted(resid_cat.items())}}
    return out


def error_examples(notes, preds, limit: int = 40) -> dict:
    """Synthetic data only: most frequent missed reference strings and false-positive strings."""
    missed, spurious = Counter(), Counter()
    for nid, text, gold, ign, dtype in notes:
        pred = preds[nid]
        for g in gold:
            if not any(p.start < g.end and g.start < p.end for p in pred):
                missed[(g.label, text[g.start:g.end])] += 1
        for p in pred:
            if any(p.start < r.end and r.start < p.end for r in ign):
                continue
            if not any(p.start < g.end and g.start < p.end for g in gold):
                spurious[(p.label, text[p.start:p.end])] += 1
    return {"missed_reference_spans": [{"label": l, "text": t, "n": n} for (l, t), n in missed.most_common(limit)],
            "false_positive_spans": [{"label": l, "text": t, "n": n} for (l, t), n in spurious.most_common(limit)],
            "missed_by_label": dict(Counter(l for (l, _), n in missed.items() for _ in range(n))),
            "false_positive_by_label": dict(Counter(l for (l, _), n in spurious.items() for _ in range(n)))}


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def report(name: str, data_path: str, res: dict, speed: dict, errs: dict, cats) -> str:
    md = [f"# Experiment 12 — external evaluation: {name}", "",
          f"Data: `{data_path}` · {res['system']['scores']['documents']} notes · measured on this machine.",
          "No rule, threshold or model setting was changed for this evaluation.", "",
          "## Headline", "",
          "| System | Strict P | Strict R [95% CI] | Strict F1 | Label-agnostic R | Spans fully covered | Token P | Token R | Token F1 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key in ("rules", "model", "system"):
        r = res[key]
        s = r["scores"]
        ci = s.get("bootstrap_ci", {}).get("strict", {}).get("recall", [None, None])
        md.append(f"| {key} | {pct(s['strict']['precision'])} | {pct(s['strict']['recall'])} [{pct(ci[0])}–{pct(ci[1])}] | "
                  f"{pct(s['strict']['f1'])} | {pct(s['agnostic']['recall'])} | "
                  f"{pct(s['overlap_diagnostics']['gold_fully_covered_rate'])} | {pct(r['token_level']['precision'])} | "
                  f"{pct(r['token_level']['recall'])} | {pct(r['token_level']['f1'])} |")
    sysr = res["system"]
    md += ["", "## Per category — deployed system (rules + model)", "",
           "| Category | Reference spans | Strict R | Strict P | Fully covered | Partly covered | Missed |", "|---|---|---|---|---|---|---|"]
    for c in cats:
        pc = sysr["scores"]["per_category"].get(c)
        cv = sysr["coverage_by_category"].get(c)
        if not cv:
            continue
        md.append(f"| {c} | {cv['spans']} | {pct(pc['recall']) if pc else 'n/a'} | {pct(pc['precision']) if pc else 'n/a'} | "
                  f"{pct(cv['full'] / cv['spans'])} | {pct(cv['partial_only'] / cv['spans'])} | {pct(cv['missed'] / cv['spans'])} |")
    if "residual" in sysr:
        t = sysr["residual"]["totals"]
        n = t.get("spans", 0)
        md += ["", "## Residual PHI after the configured policy (deployed system)", "",
               f"- Reference spans: {n}",
               f"- Fully protected: {t.get('fully_protected', 0)} ({pct(t.get('fully_protected', 0) / n if n else None)})",
               f"- Partly exposed, identifying characters left: {t.get('partially_exposed_identifying', 0)}",
               f"- Partly exposed, only titles/punctuation left: {t.get('partially_exposed_nonidentifying', 0)}",
               f"- Fully exposed (missed): {t.get('fully_exposed', 0)} ({pct(t.get('fully_exposed', 0) / n if n else None)})"]
    md += ["", "## Full coverage by document type (deployed system)", "", "| Document type | Spans | Fully covered |", "|---|---|---|"]
    for k, v in sysr["full_coverage_by_doc_type"].items():
        md.append(f"| {k} | {v['spans']} | {pct(v['rate'])} |")
    md += ["", "## Speed (this machine)", "",
           f"- rules: {speed['rules_ms']:.1f} ms/note · model: {speed['model_s']:.2f} s/note · notes: {speed['notes']} · "
           f"mean length {speed['mean_chars']} chars", "",
           "## Most frequent misses and false alarms (synthetic data, fictional values)", "",
           "Missed reference spans (no overlap with any prediction):", ""]
    md += [f"- {e['label']}: `{e['text']}` ×{e['n']}" for e in errs["missed_reference_spans"][:25]]
    md += ["", "False positives (no overlap with any reference span):", ""]
    md += [f"- {e['label']}: `{e['text']}` ×{e['n']}" for e in errs["false_positive_spans"][:25]]
    return "\n".join(md) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/exp12_external")
    args = ap.parse_args(argv)
    notes = load(resolve_path(args.data))
    cats = load_labels()["categories"]
    det_cfg = load_yaml("configs/detection.yaml")
    pipe = DeidPipeline("selected", deployment=True)
    fcfg = FusionConfig.from_config(det_cfg)
    rules, model, system = {}, {}, {}
    t_rules = t_model = 0.0
    for i, (nid, text, *_r) in enumerate(notes):
        t0 = time.perf_counter()
        r = pipe.regex.detect(text)
        t1 = time.perf_counter()
        m = postprocess_model(pipe.model.detect(text), text, det_cfg)
        t2 = time.perf_counter()
        t_rules += t1 - t0
        t_model += t2 - t1
        rules[nid] = postprocess_final(r, text, det_cfg)
        model[nid] = postprocess_final(m, text, det_cfg)
        system[nid] = postprocess_final(fuse(text, [r, m], fcfg), text, det_cfg)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(notes)} notes", flush=True)
    engine = PolicyEngine(load_yaml("configs/deid_policy.yaml"), key=secrets.token_bytes(32))
    res = {"rules": evaluate("rules", notes, rules, None, cats),
           "model": evaluate("model", notes, model, None, cats),
           "system": evaluate("system", notes, system, engine, cats)}
    speed = {"notes": len(notes), "rules_ms": 1000 * t_rules / len(notes), "model_s": t_model / len(notes),
             "mean_chars": round(sum(len(n[1]) for n in notes) / len(notes))}
    errs = error_examples(notes, system)
    out = resolve_path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"data": args.data, "results": res, "speed": speed,
                                                  "errors_system": errs}, indent=2), encoding="utf-8")
    for key, preds in (("rules", rules), ("model", model), ("system", system)):
        with open(out / f"predictions.{key}.jsonl", "w", encoding="utf-8") as f:
            for nid in sorted(preds):
                f.write(json.dumps({"note_id": nid, "entities": [e.to_dict() for e in preds[nid]]}) + "\n")
    md = report(args.name, args.data, res, speed, errs, cats)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
