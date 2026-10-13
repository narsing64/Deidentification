"""Section 11: CPU optimisation of ONE model (normally the selected one), measured, not assumed.

Variants: PyTorch fp32 (baseline) | ONNX Runtime fp32 | ONNX Runtime dynamic int8.
Also a small sweep over torch threads and batch size on unlabelled notes.

Acceptance criteria, declared BEFORE measuring:
  an optimisation is accepted only if, on the validation split, combined (model+regex) strict
  recall and label-agnostic recall each drop by at most 0.002 absolute (adjusted reference), AND
  median latency improves by >= 15%.

    python scripts/optimize_model.py --model <key> --out results/exp9_optimization
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ACCEPT_MAX_RECALL_DROP = 0.002
ACCEPT_MIN_SPEEDUP = 1.15


def export_onnx(key: str, onnx_dir: Path) -> tuple[Path, Path]:
    import torch
    from transformers import AutoModelForTokenClassification

    from deid.config import load_yaml
    from deid.detectors.hf_detector import resolve_cache_dir
    models_cfg = load_yaml("configs/models.yaml")
    spec = next(m for m in models_cfg["shortlist"] if m["key"] == key)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    fp32, int8 = onnx_dir / "model.onnx", onnx_dir / "model.int8.onnx"
    if not fp32.exists():
        model = AutoModelForTokenClassification.from_pretrained(
            spec["hf_id"], revision=spec["revision"], cache_dir=resolve_cache_dir(models_cfg.get("cache_dir")),
            local_files_only=True).eval()
        ids = torch.ones((2, 16), dtype=torch.long)
        mask = torch.ones((2, 16), dtype=torch.long)

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask):
                return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

        kwargs = dict(input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "b", 1: "n"}, "attention_mask": {0: "b", 1: "n"},
                                    "logits": {0: "b", 1: "n"}}, opset_version=17)
        try:
            torch.onnx.export(Wrap(model), (ids, mask), str(fp32), dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(Wrap(model), (ids, mask), str(fp32), **kwargs)
    if not int8.exists():
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    return fp32, int8


def run_variant(key: str, variant: str, onnx_path: str | None, out: Path, threads: int, batch: int) -> int:
    from deid.config import load_labels, load_yaml
    from deid.detectors.hf_detector import build_detector
    from deid.evaluation.benchmark import load_eval_notes, save_predictions
    from deid.evaluation.resources import PeakMemory, latency_summary

    models_cfg = load_yaml("configs/models.yaml")
    models_cfg["torch_threads"], models_cfg["batch_size"] = threads, batch
    notes = load_eval_notes("validation")
    over = {} if variant == "torch" else {"backend": "onnx", "onnx_path": onnx_path}
    with PeakMemory() as pm:
        t0 = time.perf_counter()
        det = build_detector(key, models_cfg, load_labels(), load_yaml("configs/detection.yaml"), **over)
        init_s = time.perf_counter() - t0
        det.detect(notes[0].text)
        preds, secs = {}, {}
        for n in notes:
            t = time.perf_counter()
            preds[n.note_id] = det.detect(n.text)
            secs[n.note_id] = time.perf_counter() - t
    tag = f"{key}.{variant}.t{threads}.b{batch}"
    save_predictions(out / "predictions" / f"{tag}.jsonl", preds, secs, {"variant": variant})
    (out / f"resources.{tag}.json").write_text(json.dumps({
        "variant": variant, "threads": threads, "batch": batch, "init_s": init_s, "peak_rss_bytes": pm.peak,
        "onnx_bytes": Path(onnx_path).stat().st_size if onnx_path else None,
        "latency": latency_summary([secs[n.note_id] for n in notes], [len(n.text) for n in notes])}, indent=2),
        encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/exp9_optimization")
    ap.add_argument("--configs", nargs="*",
                    default=["torch:4:8", "torch:8:8", "torch:4:16", "onnx_fp32:4:8", "onnx_int8:4:8"],
                    help="variant:threads:batch; the first one is the baseline")
    ap.add_argument("--baseline-run", default="results/exp6_validation_final",
                    help="run folder holding the regex predictions for the validation split")
    ap.add_argument("--single", nargs=4, help=argparse.SUPPRESS)  # variant onnx_path threads batch
    args = ap.parse_args(argv)
    from deid.config import resolve_path
    out = resolve_path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    if args.single:
        v, p, t, b = args.single
        return run_variant(args.model, v, None if p == "-" else p, out, int(t), int(b))

    onnx_dir = resolve_path(".cache/onnx") / args.model
    fp32, int8 = export_onnx(args.model, onnx_dir)
    paths = {"torch": "-", "onnx_fp32": str(fp32), "onnx_int8": str(int8)}
    for spec in args.configs:
        v, t, b = spec.split(":")
        print(f"[opt] {args.model} {v} threads={t} batch={b}", flush=True)
        subprocess.run([sys.executable, __file__, "--model", args.model, "--out", str(out),
                        "--single", v, paths[v], t, b], check=False)

    # evaluate every variant (model-only and +regex) with the shared metrics
    from deid.config import load_yaml
    from deid.evaluation.benchmark import load_eval_notes, load_predictions, score_predictions
    from deid.evaluation.metrics import fmt
    from deid.fusion.span_merger import FusionConfig, fuse
    from deid.postprocess import postprocess_final, postprocess_model
    notes = load_eval_notes("validation")
    det_cfg = load_yaml("configs/detection.yaml")
    fcfg = FusionConfig.from_config(det_cfg)
    rp, _, _ = load_predictions(resolve_path(args.baseline_run) / "predictions" / "regex.validation.jsonl")
    rows = []
    for res_file in sorted(out.glob(f"resources.{args.model}.*.json")):
        tag = res_file.name[len("resources."):-len(".json")]
        r = json.loads(res_file.read_text())
        mp, _, _ = load_predictions(out / "predictions" / f"{tag}.jsonl")
        mp = {n.note_id: postprocess_model(mp.get(n.note_id, []), n.text, det_cfg) for n in notes}
        fused = {n.note_id: postprocess_final(fuse(n.text, [rp.get(n.note_id, []), mp[n.note_id]], fcfg), n.text, det_cfg)
                 for n in notes}
        mp = {n.note_id: postprocess_final(mp[n.note_id], n.text, det_cfg) for n in notes}
        m_only = score_predictions(mp, notes, bootstrap=False)
        comb = score_predictions(fused, notes, bootstrap=False)
        rows.append({"tag": tag, **r, "model_only": {k: m_only[k] for k in ("strict", "agnostic")},
                     "combined": {k: comb[k] for k in ("strict", "agnostic")}})
    bv, bt, bb = args.configs[0].split(":")
    base = next((x for x in rows if x["variant"] == bv and x["threads"] == int(bt) and x["batch"] == int(bb)), rows[0])
    md = [f"# Section 11 — CPU optimisation of `{args.model}` (validation, measured)", "",
          f"Acceptance (declared before measuring): combined strict and agnostic recall drop <= {ACCEPT_MAX_RECALL_DROP} "
          f"and median-latency speed-up >= {ACCEPT_MIN_SPEEDUP}x vs `{base['tag']}`.", "",
          "| Variant | Threads | Batch | Weights (MB) | Peak RSS (MB) | Init (s) | Median s/note | p95 s/note | Speed-up | "
          "Model-only strict R | Combined strict P / R | Combined agnostic R | Accepted |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        sp = base["latency"]["median_s"] / x["latency"]["median_s"]
        dr_s = base["combined"]["strict"]["recall"] - x["combined"]["strict"]["recall"]
        dr_a = base["combined"]["agnostic"]["recall"] - x["combined"]["agnostic"]["recall"]
        x["speedup"], x["accepted"] = sp, (x is base) or (dr_s <= ACCEPT_MAX_RECALL_DROP and dr_a <= ACCEPT_MAX_RECALL_DROP and sp >= ACCEPT_MIN_SPEEDUP)
        wmb = f"{x['onnx_bytes']/2**20:.0f}" if x.get("onnx_bytes") else "(torch)"
        md.append(f"| {x['variant']} | {x['threads']} | {x['batch']} | {wmb} | {x['peak_rss_bytes']/2**20:.0f} | {x['init_s']:.1f} | "
                  f"{x['latency']['median_s']:.3f} | {x['latency']['p95_s']:.3f} | {sp:.2f}x | {fmt(x['model_only']['strict']['recall'], 4)} | "
                  f"{fmt(x['combined']['strict']['precision'], 4)} / {fmt(x['combined']['strict']['recall'], 4)} | "
                  f"{fmt(x['combined']['agnostic']['recall'], 4)} | {'baseline' if x is base else ('yes' if x['accepted'] else 'no')} |")
    (out / f"report.{args.model}.md").write_text("\n".join(md), encoding="utf-8")
    (out / f"summary.{args.model}.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
