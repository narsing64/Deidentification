"""Experiment 1: feasibility checks for every shortlisted model (each in its own process).

Checks: the pinned revision loads offline, the classification head is present, every model label is
mapped, the fast tokenizer gives offsets, synthetic sanity sentences produce sensible entities,
sizes on disk, initialisation time, peak RAM and latency on a seeded sample of UNLABELLED notes
(train.tsv; text is never printed).

    python scripts/check_models.py                 # all models, writes results/exp1_feasibility/
    python scripts/check_models.py --only obi_deid_bert_i2b2
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

SANITY = [  # synthetic sentences only
    "Patient Jane Roe was seen by Dr. Max Mustermann on 03/04/2101 at 10:30 AM.",
    "MRN 00123456. Call (555) 123-4567 or email jane.roe@example.org. Lives at 12 Oak Tree Lane, Springfield.",
    "Admitted to Lakeside General Hospital. Metoprolol 25 mg PO BID; BP 120/80; WBC 8.5.",
]


def run_single(key: str, out_path: Path, n_speed: int, warmup: int) -> int:
    from deid.config import load_labels, load_yaml, resolve_path
    from deid.data.loader import load_unlabelled_notes
    from deid.detectors.hf_detector import ModelLoadError, build_detector, resolve_cache_dir
    from deid.evaluation.resources import PeakMemory, dir_size_bytes, latency_summary, machine_info

    models_cfg, labels_cfg = load_yaml("configs/models.yaml"), load_labels()
    det_cfg, bench_cfg = load_yaml("configs/detection.yaml"), load_yaml("configs/benchmark.yaml")
    res: dict = {"key": key, "ok": False}
    with PeakMemory() as pm:
        t0 = time.perf_counter()
        try:
            det = build_detector(key, models_cfg, labels_cfg, det_cfg)
        except ModelLoadError as exc:
            res["error"] = str(exc)
            out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
            return 1
        res["init_s"] = time.perf_counter() - t0
        res["rss_after_load_bytes"] = pm.proc.memory_info().rss
        res["describe"] = det.describe()
        res["sanity"] = [[(s[e.start:e.end], e.label, e.fine_label, e.score) for e in det.detect(s)] for s in SANITY]
        notes, _ = load_unlabelled_notes(resolve_path(bench_cfg["data"]["unlabelled_tsv"]))
        rng = random.Random(bench_cfg["split"]["seed"])
        sample = rng.sample(notes, min(len(notes), n_speed + warmup))
        lat, chars, n_ents = [], [], 0
        for i, note in enumerate(sample):
            t = time.perf_counter()
            ents = det.detect(note.text)
            dt = time.perf_counter() - t
            if i >= warmup:
                lat.append(dt)
                chars.append(len(note.text))
                n_ents += len(ents)
        res["speed_unlabelled"] = latency_summary(lat, chars)
        res["speed_unlabelled"]["entities_per_note"] = n_ents / max(1, len(lat))
    res["peak_rss_bytes"] = pm.peak
    from huggingface_hub import snapshot_download
    snap = snapshot_download(det.spec.hf_id, revision=det.spec.revision, local_files_only=True,
                             cache_dir=resolve_cache_dir(models_cfg.get("cache_dir")))
    res["disk_bytes"] = dir_size_bytes(snap)
    res["weights_bytes"] = dir_size_bytes(snap, exts=(".safetensors", ".bin"))
    res["machine"] = machine_info()
    res["ok"] = True
    out_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--single", help=argparse.SUPPRESS)
    ap.add_argument("--json-out", help=argparse.SUPPRESS)
    ap.add_argument("--speed-notes", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="results/exp1_feasibility")
    args = ap.parse_args(argv)
    if args.single:
        return run_single(args.single, Path(args.json_out), args.speed_notes, args.warmup)

    from deid.config import load_yaml, resolve_path
    from deid.evaluation.report import resource_table
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = [m["key"] for m in load_yaml("configs/models.yaml")["shortlist"]]
    if args.only:
        keys = [k for k in keys if k in args.only]
    results = []
    for key in keys:
        jp = out / f"{key}.json"
        print(f"[exp1] {key} ...", flush=True)
        proc = subprocess.run([sys.executable, __file__, "--single", key, "--json-out", str(jp),
                               "--speed-notes", str(args.speed_notes), "--warmup", str(args.warmup)],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        if not jp.exists():
            results.append({"key": key, "ok": False, "error": (proc.stderr or "")[-2000:]})
        else:
            results.append(json.loads(jp.read_text(encoding="utf-8")))
        r = results[-1]
        print(f"   ok={r['ok']} " + (f"init={r['init_s']:.1f}s peak={r['peak_rss_bytes']/2**20:.0f}MB "
                                     f"median={r['speed_unlabelled']['median_s']:.2f}s/note" if r["ok"] else r.get("error", "")[:300]), flush=True)
    rows = []
    for r in results:
        if r["ok"]:
            rows.append((r["key"], {"download_bytes": r["disk_bytes"], "disk_bytes": r["disk_bytes"],
                                    "peak_rss_bytes": r["peak_rss_bytes"], "init_s": r["init_s"],
                                    "latency": r["speed_unlabelled"]}))
    md = ["# Experiment 1 — feasibility (measured)", "",
          f"Latency sample: {args.speed_notes} seeded notes from the UNLABELLED train.tsv (after {args.warmup} warm-up notes), "
          "one process per model, torch threads fixed in configs/models.yaml.", "",
          resource_table(rows), "", "## Load checks", "",
          "| Model | Loaded | Params | Tokenizer | Labels | Scheme | Window/context | Missing/unexpected keys |",
          "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if not r["ok"]:
            md.append(f"| {r['key']} | **FAILED** | | | | | | {r.get('error', '')[:200]} |")
            continue
        d = r["describe"]
        md.append(f"| {r['key']} | yes | {d['params']/1e6:.1f} M | {d['tokenizer']} | {d['n_labels']} | {d['tag_scheme']} | "
                  f"{d['window_tokens']}/{d['context_tokens']} | {d['loading_info']} |")
    md += ["", "## Synthetic sanity sentences", ""]
    for i, s in enumerate(SANITY):
        md.append(f"**S{i+1}:** {s}")
        for r in results:
            if r["ok"]:
                md.append(f"- {r['key']}: " + "; ".join(f"`{t}`→{lab}" for t, lab, _f, _s in r["sanity"][i]))
        md.append("")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
