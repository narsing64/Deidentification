"""Reproducible, privacy-safe dataset inspection (Experiment 1, data part).

Prints and saves COUNTS ONLY: schema, sizes, encodings, duplicates, lengths, label vocabulary,
consistency of the supplied annotation offsets, provenance indicators and train/test overlap.
No note text, span text or identifier is ever printed.

    python scripts/inspect_dataset.py --out results/00_inspection
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from deid.config import load_yaml, resolve_path
from deid.data.gold import PLACEHOLDER_RE
from deid.data.loader import load_annotation_meta, load_reference_pairs, load_unlabelled_notes

PROVENANCE_TERMS = ["synthetic", "generated", "fictional", "Here is", "```", "As an AI"]
MIMIC_HEADERS = ["Major Surgical or Invasive Procedure:", "Brief Hospital Course:", "Discharge Disposition:",
                 "Followup Instructions:", "Pertinent Results:"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_facts(path: Path) -> dict:
    raw = path.read_bytes()
    try:
        raw.decode("utf-8")
        utf8 = True
    except UnicodeDecodeError:
        utf8 = False
    return {"bytes": len(raw), "sha256": sha256(path), "utf8": utf8, "bom": raw[:3] == b"\xef\xbb\xbf",
            "crlf": raw.count(b"\r\n"), "lines": raw.count(b"\n")}


def length_stats(texts: list[str]) -> dict:
    L = np.array([len(t) for t in texts]) if texts else np.array([0])
    return {"n": len(texts), "min": int(L.min()), "median": float(np.median(L)), "mean": round(float(L.mean()), 1),
            "max": int(L.max()), "gt_5k": int((L > 5000).sum()), "gt_10k": int((L > 10000).sum()),
            "gt_20k": int((L > 20000).sum())}


def text_facts(texts: list[str]) -> dict:
    nonascii = Counter(c for t in texts for c in t if ord(c) > 127)
    return {
        "lengths": length_stats(texts),
        "exact_duplicates": len(texts) - len(set(texts)),
        "notes_with_non_ascii": sum(any(ord(c) > 127 for c in t) for t in texts),
        "top_non_ascii_codepoints": [(f"U+{ord(c):04X}", n) for c, n in nonascii.most_common(10)],
        "notes_with_U+FFFD": sum("�" in t for t in texts),
        "provenance_term_doc_counts": {k: sum(k.lower() in t.lower() for t in texts) for k in PROVENANCE_TERMS},
        "mimic_header_doc_counts": {k: sum(k in t for t in texts) for k in MIMIC_HEADERS},
        "notes_with_mimic_deid_marker___": sum("___" in t for t in texts),
    }


def shingles(t: str, k: int = 100, step: int = 50) -> set[str]:
    return {hashlib.md5(t[i:i + k].encode("utf-8")).hexdigest() for i in range(0, max(1, len(t) - k), step)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--out", default="results/00_inspection")
    args = ap.parse_args(argv)
    data = load_yaml(args.config)["data"]
    paths = {k: resolve_path(data[k]) for k in ("annotated_csv", "annotations_jsonl", "unlabelled_tsv")}

    report: dict = {"files": {k: {"name": p.name, **file_facts(p)} for k, p in paths.items()}}
    pairs, rep_pairs = load_reference_pairs(paths["annotated_csv"])
    meta, rep_meta = load_annotation_meta(paths["annotations_jsonl"])
    train, rep_train = load_unlabelled_notes(paths["unlabelled_tsv"])
    report["loader_validation"] = [rep_pairs.to_dict(), rep_meta.to_dict(), rep_train.to_dict()]
    report["has_ground_truth"] = {"train.tsv": False, "test (reference de-identified text)": True}

    raw = [p["raw"] for p in pairs]
    report["annotated_raw"] = text_facts(raw)
    report["unlabelled_train"] = text_facts([n.text for n in train])
    report["annotated_by_source_tier"] = dict(Counter(f"{m['source']}/{m['tier']}" for m in meta.values()))
    report["ids_in_both_csv_and_jsonl"] = len({p["note_id"] for p in pairs} & set(meta))

    # Only known labels are reported by name. Bracketed tokens that are not labels are often leaked
    # values in the reference (e.g. an upper-cased surname), so they are only counted, never named.
    from deid.config import load_labels
    from deid.data.gold import ReferenceLabels
    rl = ReferenceLabels(load_labels())
    ph: Counter = Counter()
    for p in pairs:
        for m in PLACEHOLDER_RE.finditer(p["reference"]):
            lab = m.group()[1:-1]
            status, _, canonical = rl.resolve(lab)
            if status == "invalid":
                ph["<not a label: possible leaked value>"] += 1
            elif canonical != lab:
                ph[f"<corrupted variant of {canonical}>"] += 1
            else:
                ph[lab] += 1
    report["reference_placeholder_counts"] = dict(ph.most_common())
    report["raw_contains_placeholder_like_tokens"] = sum(bool(PLACEHOLDER_RE.search(t)) for t in raw)

    # consistency of the supplied `changes` offsets (re-read, counts only)
    texts = {p["note_id"]: (p["raw"], p["reference"]) for p in pairs}
    ok_raw = ok_ref = total = 0
    docs_consistent = 0
    ops = Counter()
    with open(paths["annotations_jsonl"], encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("id") not in texts:
                continue
            r, d = texts[rec["id"]]
            bad = 0
            for c in rec.get("changes") or []:
                total += 1
                ops[c.get("operation")] += 1
                a = r[c["raw_start"]:c["raw_end"]] == c["raw_text"]
                b = d[c["deid_start"]:c["deid_end"]] == c["deid_text"]
                ok_raw += a
                ok_ref += b
                bad += (not a)
            docs_consistent += bad == 0
    report["supplied_change_offsets"] = {"changes": total, "operations": dict(ops),
                                         "raw_offsets_matching": ok_raw, "reference_offsets_matching": ok_ref,
                                         "raw_mismatch_rate": round(1 - ok_raw / total, 4) if total else None,
                                         "notes_fully_consistent": docs_consistent}

    tr = set().union(*(shingles(n.text) for n in train))
    ov = [len(s & tr) / max(1, len(s)) for s in (shingles(t) for t in raw)]
    report["train_test_overlap"] = {"exact_duplicates": len(set(raw) & {n.text for n in train}),
                                    "max_100char_shingle_overlap": round(max(ov), 4) if ov else 0.0}

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("has_ground_truth", "annotated_by_source_tier", "supplied_change_offsets",
                                             "train_test_overlap")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
