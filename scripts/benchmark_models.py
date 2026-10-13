"""Experiments 3 and 4: model-only and model+regex evaluation of the fixed shortlist.

Stage `predict`: each model runs in its own process (isolated peak-RAM measurement) on the chosen
split and writes predictions (offsets/labels/scores only) plus a resources JSON.
Stage `evaluate`: scores regex-only, model-only and fused model+regex predictions with the shared
metrics under both reference profiles, and writes comparison tables.

    python scripts/benchmark_models.py --stage predict  --out results/exp3_4_validation
    python scripts/benchmark_models.py --stage evaluate --out results/exp3_4_validation
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def predict_single(key: str, split: str, out: Path, allow_test: bool, det_cfg_path: str = "configs/detection.yaml",
                   dump_scores: bool = False, deployment: bool = False) -> int:
    from deid.config import load_labels, load_yaml
    from deid.detectors.hf_detector import ModelLoadError, build_detector, entities_from_word_scores, resolve_cache_dir
    from deid.evaluation.benchmark import load_eval_notes, save_predictions
    from deid.evaluation.resources import PeakMemory, dir_size_bytes, latency_summary, machine_info

    models_cfg, det_cfg = load_yaml("configs/models.yaml"), load_yaml(det_cfg_path)
    notes = load_eval_notes(split, allow_test=allow_test, purpose=f"final evaluation: {key}")
    res: dict = {"key": key, "split": split, "ok": False}
    with PeakMemory() as pm:
        t0 = time.perf_counter()
        try:
            det = build_detector(key, models_cfg, load_labels(), det_cfg, deployment=deployment)
        except ModelLoadError as exc:
            res["error"] = str(exc)
            (out / f"resources.{key}.{split}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            return 1
        res["init_s"] = time.perf_counter() - t0
        det.detect(notes[0].text)  # warm-up (not timed)
        preds, secs, dumps = {}, {}, {}
        for i, n in enumerate(notes):
            t = time.perf_counter()
            ws = det.score_words(n.text)
            preds[n.note_id] = [] if ws is None else entities_from_word_scores(n.text, ws, **det.decode_params())
            secs[n.note_id] = time.perf_counter() - t
            if dump_scores and ws is not None:
                dumps[n.note_id] = ws.to_dict()
            if (i + 1) % 20 == 0:
                print(f"  {key}: {i + 1}/{len(notes)}", flush=True)
    from huggingface_hub import snapshot_download
    snap = snapshot_download(det.spec.hf_id, revision=det.spec.revision, local_files_only=True,
                             cache_dir=resolve_cache_dir(models_cfg.get("cache_dir")))
    disk = dir_size_bytes(snap)
    if det.backend == "onnx":
        from pathlib import Path as _P
        disk = _P(det.describe().get("onnx_path") or "").stat().st_size if det.describe().get("onnx_path") else disk
    res.update({
        "ok": True, "peak_rss_bytes": pm.peak, "describe": det.describe(),
        "download_bytes": dir_size_bytes(snap), "disk_bytes": disk,
        "latency": latency_summary([secs[n.note_id] for n in notes], [len(n.text) for n in notes]),
        "machine": machine_info(), "detection_config": det_cfg,
    })
    save_predictions(out / "predictions" / f"{key}.{split}.jsonl", preds, secs, {"system": key, "split": split})
    if dump_scores:  # per-word scores: offsets and numbers only (git-ignored)
        wd = out / "wordscores"
        wd.mkdir(parents=True, exist_ok=True)
        with open(wd / f"{key}.{split}.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": {"tags": det.tags, "o_index": det.o_index, "scheme": det.spec.tag_scheme,
                                          "label_schema": det.spec.label_schema}}) + "\n")
            for nid, d in dumps.items():
                f.write(json.dumps({"note_id": nid, "ws": d}) + "\n")
    (out / f"resources.{key}.{split}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return 0


def stage_predict(args, keys: list[str], out: Path) -> int:
    rc = 0
    for key in keys:
        print(f"[predict] {key} on {args.split}", flush=True)
        cmd = [sys.executable, __file__, "--single", key, "--split", args.split, "--out", str(out),
               "--detection-config", args.detection_config] + (["--dump-word-scores"] if args.dump_word_scores else [])             + (["--deployment"] if args.deployment else [])
        if args.allow_test:
            cmd.append("--allow-test")
        r = subprocess.run(cmd)
        rc |= r.returncode
    # the regex predictions for the same split (cheap, in-process)
    from deid.config import load_yaml
    from deid.detectors.regex_detector import RegexDetector
    from deid.evaluation.benchmark import load_eval_notes, run_detector, save_predictions
    from deid.evaluation.resources import PeakMemory, latency_summary
    notes = load_eval_notes(args.split, allow_test=args.allow_test, purpose="final evaluation: regex")
    with PeakMemory() as pm:
        t0 = time.perf_counter()
        det = RegexDetector.from_config(load_yaml(args.detection_config))
        init_s = time.perf_counter() - t0
        preds, secs = run_detector(det.detect, notes)
    save_predictions(out / "predictions" / f"regex.{args.split}.jsonl", preds, secs, {"system": "regex"})
    (out / f"resources.regex.{args.split}.json").write_text(json.dumps({
        "key": "regex", "ok": True, "init_s": init_s, "peak_rss_bytes": pm.peak, "download_bytes": 0, "disk_bytes": 0,
        "latency": latency_summary([secs[n.note_id] for n in notes], [len(n.text) for n in notes])}, indent=2),
        encoding="utf-8")
    return rc


def stage_evaluate(args, keys: list[str], out: Path) -> int:
    from deid.config import load_labels, load_yaml
    from deid.evaluation.benchmark import evaluate_system, load_eval_notes, load_predictions, write_json
    from deid.evaluation.error_analysis import render_markdown
    from deid.evaluation.report import (confusion_table, error_table, headline_table, per_category_table,
                                        resource_table)
    from deid.fusion.span_merger import FusionConfig, fuse
    from deid.postprocess import postprocess_final, postprocess_model

    cats = load_labels()["categories"]
    det_cfg = load_yaml(args.detection_config)
    fcfg = FusionConfig.from_config(det_cfg)
    notes = load_eval_notes(args.split, allow_test=args.allow_test, purpose="final evaluation: scoring")
    texts = {n.note_id: n.text for n in notes}
    regex_preds, _, _ = load_predictions(out / "predictions" / f"regex.{args.split}.jsonl")
    final = lambda preds: {nid: postprocess_final(preds.get(nid, []), texts[nid], det_cfg) for nid in texts}
    systems: dict[str, dict] = {"regex": evaluate_system(final(regex_preds), notes)}
    resources: dict[str, dict] = {"regex": json.loads((out / f"resources.regex.{args.split}.json").read_text())}
    fusion_stats: dict[str, dict] = {}
    for key in keys:
        p = out / "predictions" / f"{key}.{args.split}.jsonl"
        if not p.exists():
            print(f"[evaluate] missing predictions for {key}; skipped")
            continue
        mp, _, _ = load_predictions(p)
        mp = {nid: postprocess_model(mp.get(nid, []), texts[nid], det_cfg) for nid in texts}
        systems[f"{key}"] = evaluate_system(final(mp), notes)
        fused, stats = {}, {}
        for nid, text in texts.items():
            fused[nid], st = fuse(text, [regex_preds.get(nid, []), mp.get(nid, [])], fcfg, return_stats=True)
            for k, v in st.items():
                stats[k] = stats.get(k, 0) + v
        systems[f"{key}+regex"] = evaluate_system(final(fused), notes)
        fusion_stats[key] = stats
        resources[key] = json.loads((out / f"resources.{key}.{args.split}.json").read_text())
    write_json(out / f"metrics.{args.split}.json", {"split": args.split, "systems": systems,
                                                     "resources": resources, "fusion_stats": fusion_stats})
    order = ["regex"] + [k for k in keys if k in systems] + [f"{k}+regex" for k in keys if f"{k}+regex" in systems]
    md = [f"# Experiments 3–4 — model-only and model+regex ({args.split}, {len(notes)} notes)", "",
          "All numbers are MEASURED on this machine. Reference profiles: *adjusted* (coverage rules R1/R2, "
          "used for decisions) and *as-is* (docs/04_reference_and_evaluation_protocol.md).", ""]
    for prof in ("adjusted", "as_is"):
        rows = [(name, systems[name][prof]) for name in order]
        md += [f"## Headline — {prof} reference (micro; 95% bootstrap CI on recall)", headline_table(rows), "",
               f"## Error breakdown — {prof} reference (strict)", error_table(rows), ""]
    rows = [(name, systems[name]["adjusted"]) for name in order]
    md += ["## Per category, strict, adjusted reference", per_category_table(rows, cats), "",
           "## Resources (model-only inference on this split)",
           resource_table([(k, resources[k]) for k in ["regex"] + [k for k in keys if k in resources]]), ""]
    for name in order:
        md += [f"## Confusion (exact-span), adjusted — {name}", confusion_table(systems[name]["adjusted"], cats), ""]
    md += ["## Sanitised error analysis (adjusted reference)", ""]
    for name in order:
        md += [render_markdown(systems[name]["errors_adjusted"], name), ""]
    (out / f"report.{args.split}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:2 + 2 * (len(order) + 6)]))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["predict", "evaluate", "all"], default="all")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--allow-test", action="store_true", help="required for --split test (Experiment 8 only)")
    ap.add_argument("--models", nargs="*", help="subset of shortlist keys (default: all)")
    ap.add_argument("--out", default="results/exp3_4_validation")
    ap.add_argument("--detection-config", default="configs/detection.yaml")
    ap.add_argument("--dump-word-scores", action="store_true", help="also save per-word model scores")
    ap.add_argument("--deployment", action="store_true", help="use the accepted deployment settings (models.yaml)")
    ap.add_argument("--single", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    from deid.config import load_yaml, resolve_path
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.split == "test" and not args.allow_test:
        print("Refusing to touch the held-out test split without --allow-test.")
        return 2
    if args.single:
        return predict_single(args.single, args.split, out, args.allow_test, args.detection_config, args.dump_word_scores,
                              args.deployment)
    keys = [m["key"] for m in load_yaml("configs/models.yaml")["shortlist"]]
    if args.models:
        keys = [k for k in keys if k in args.models]
    rc = 0
    if args.stage in ("predict", "all"):
        rc |= stage_predict(args, keys, out)
    if args.stage in ("evaluate", "all"):
        rc |= stage_evaluate(args, keys, out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
