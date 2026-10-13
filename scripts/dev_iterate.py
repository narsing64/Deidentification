"""Development loop for rule / post-processing changes (Experiment 13). Uses DEV data only:
  * MIMIC validation split (adjusted reference profile)
  * synthetic Indian dev split (data/synthetic_in/dev.jsonl)

Raw predictions of the deployed model (int8 ONNX) are cached once. Every run then re-applies the CURRENT
regex code, post-processing and fusion, so a change can be measured in seconds. Aggregate numbers only;
no note text is printed.

    python scripts/dev_iterate.py --tag baseline        # writes results/exp13_dev_fixes/<tag>.json
    python scripts/dev_iterate.py --tag v4 --compare baseline
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_external import evaluate  # noqa: E402

from deid.config import load_labels, load_yaml, resolve_path
from deid.detectors.regex_detector import RegexDetector
from deid.evaluation.benchmark import filter_excluded, load_eval_notes, profile_view
from deid.fusion.span_merger import FusionConfig, fuse
from deid.policies.transform import PolicyEngine
from deid.postprocess import postprocess_final, postprocess_model
from deid.schema import Entity, Region

OUT = "results/exp13_dev_fixes"


def dev_sets(final_test: bool = False):
    if final_test:   # ONE logged access to the MIMIC held-out test split for the final v4 report
        notes = load_eval_notes("test", allow_test=True, purpose="Experiment 13: final re-score of the v4 system")
        return {"mimic_test": [(n.note_id, n.text, *profile_view(n, "adjusted"), n.source or "mimic", n) for n in notes]}
    mimic = []
    for n in load_eval_notes("validation"):
        gold, ign = profile_view(n, "adjusted")
        mimic.append((n.note_id, n.text, gold, ign, n.source or "mimic", n))
    def jsonl(path):
        out = []
        for line in open(resolve_path(path), encoding="utf-8"):
            r = json.loads(line)
            out.append((r["note_id"], r["text"], [Entity(s["start"], s["end"], s["label"]) for s in r["spans"]],
                        [Region(g["start"], g["end"], g.get("reason", "")) for g in r.get("ignore", [])], r["doc_type"], None))
        return out
    sets = {"mimic_validation": mimic, "indian_dev": jsonl("data/synthetic_in/dev.jsonl")}
    # Experiment 14: more development data (a second Indian dev split, and Gretel's VALIDATION split)
    for name, path in (("indian_dev2", "data/synthetic_in/dev2.jsonl"),
                       ("gretel_dev", "data/external/gretel_pii_en_v1/healthcare_validation.jsonl")):
        if resolve_path(path).exists():
            sets[name] = jsonl(path)
    return sets


def raw_model_preds(name: str, notes) -> dict:
    path = resolve_path(OUT) / "cache" / f"model_raw.{name}.jsonl"
    if path.exists():
        return {r["note_id"]: [Entity.from_dict(e) for e in r["entities"]]
                for r in map(json.loads, open(path, encoding="utf-8"))}
    from deid.pipeline import DeidPipeline
    pipe = DeidPipeline("selected", deployment=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    with open(path, "w", encoding="utf-8") as f:
        for nid, text, *_ in notes:
            out[nid] = pipe.model.detect(text)
            f.write(json.dumps({"note_id": nid, "entities": [e.to_dict() for e in out[nid]]}) + "\n")
    return out


def summarise(res: dict) -> dict:
    s = res["scores"]
    r = res.get("residual", {}).get("totals", {})
    n = r.get("spans") or 1
    return {"strict_P": s["strict"]["precision"], "strict_R": s["strict"]["recall"], "strict_F1": s["strict"]["f1"],
            "agnostic_R": s["agnostic"]["recall"], "overlap_P": s["overlap_diagnostics"]["pred_overlap_precision"],
            "fully_covered": s["overlap_diagnostics"]["gold_fully_covered_rate"],
            "token_P": res["token_level"]["precision"], "token_R": res["token_level"]["recall"],
            "ident_left": r.get("partially_exposed_identifying", 0) / n, "fully_exposed": r.get("fully_exposed", 0) / n,
            "per_cat_R": {c: v["recall"] for c, v in s["per_category"].items() if v["tp"] + v["fn"]},
            "per_cat_P": {c: v["precision"] for c, v in s["per_category"].items() if v["tp"] + v["fp"]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--compare")
    ap.add_argument("--final-mimic-test", action="store_true", help="score the MIMIC held-out test split (logged)")
    args = ap.parse_args(argv)
    cfg = load_yaml("configs/detection.yaml")
    fcfg = FusionConfig.from_config(cfg)
    regex = RegexDetector.from_config(cfg)
    engine = PolicyEngine(load_yaml("configs/deid_policy.yaml"), key=secrets.token_bytes(32))
    cats = load_labels()["categories"]
    summary = {}
    for name, notes in dev_sets(args.final_mimic_test).items():
        raw = raw_model_preds(name, notes)
        preds = {"rules": {}, "model": {}, "system": {}}
        for nid, text, gold, ign, dtype, orig in notes:
            r = regex.detect(text)
            m = postprocess_model(raw[nid], text, cfg)
            for key, ents in (("rules", r), ("model", m), ("system", fuse(text, [r, m], fcfg))):
                p = postprocess_final(ents, text, cfg)
                preds[key][nid] = filter_excluded(orig, p, "adjusted") if orig is not None else p
        tuples = [(nid, text, gold, ign, dtype) for nid, text, gold, ign, dtype, _ in notes]
        summary[name] = {k: summarise(evaluate(k, tuples, preds[k], engine if k == "system" else None, cats))
                         for k in ("rules", "model", "system")}
    out = resolve_path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    base = json.loads((out / f"{args.compare}.json").read_text()) if args.compare else None
    for name, d in summary.items():
        print(f"== {name}")
        for k in ("rules", "model", "system"):
            row = d[k]
            parts = []
            for m in ("strict_P", "strict_R", "strict_F1", "overlap_P", "fully_covered", "token_P", "token_R", "ident_left", "fully_exposed"):
                v = row[m]
                delta = f"({100 * (v - base[name][k][m]):+.1f})" if base else ""
                parts.append(f"{m} {100 * v:5.1f}{delta}")
            print(f"  {k:6s} " + " | ".join(parts))
        if base:
            pc, bc = d["system"]["per_cat_R"], base[name]["system"]["per_cat_R"]
            pp, bp = d["system"]["per_cat_P"], base[name]["system"]["per_cat_P"]
            ch = [f"{c}: R {100 * bc.get(c, 0):.1f}->{100 * pc[c]:.1f} P {100 * (bp.get(c) or 0):.1f}->{100 * (pp.get(c) or 0):.1f}"
                  for c in pc if abs(pc[c] - bc.get(c, 0)) > 1e-9 or abs((pp.get(c) or 0) - (bp.get(c) or 0)) > 1e-9]
            print("  system per-category changes: " + ("; ".join(ch) if ch else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
