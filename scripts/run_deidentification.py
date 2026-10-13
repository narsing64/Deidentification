"""Run the selected de-identification pipeline on a file of notes.

Input formats: .csv / .tsv (choose the text column with --text-col, optional --id-col / --patient-col),
.jsonl (fields given by the same options) or .txt (one note per file).
Outputs (git-ignored, contain de-identified text that may still hold residual PHI):
  <out>/deidentified.jsonl     {"note_id", "text", "transformations": [offsets/category/action only]}
  <out>/run_report.json        counts only: detected / transformed / potential residual / review required
Nothing is printed except counts.

    set DEID_SECRET_KEY=<hex secret>   (Windows; use `export` on Linux/macOS)
    python scripts/run_deidentification.py --input notes.csv --text-col text --out results/deid_outputs/run1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd


def read_notes(path: Path, text_col: str, id_col: str | None, patient_col: str | None):
    if path.suffix.lower() == ".txt":
        yield path.stem, path.read_text(encoding="utf-8"), None
        return
    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                yield str(r.get(id_col, i) if id_col else i), r.get(text_col), (r.get(patient_col) if patient_col else None)
        return
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    if text_col not in df.columns:
        raise SystemExit(f"column {text_col!r} not found; available: {list(df.columns)}")
    for i, row in df.iterrows():
        yield (str(row[id_col]) if id_col else str(i)), row[text_col], (row[patient_col] if patient_col else None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--id-col")
    ap.add_argument("--patient-col", help="enables patient-level date-shift consistency")
    ap.add_argument("--model", default="selected", help="model key, 'selected' (configs/models.yaml) or 'none'")
    ap.add_argument("--no-regex", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    from deid.config import resolve_path
    from deid.pipeline import DeidPipeline
    from deid.validation.residual import ResidualReport

    pipe = DeidPipeline(None if args.model == "none" else args.model, use_regex=not args.no_regex)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = ResidualReport()
    actions, statuses = Counter(), Counter()
    n = skipped = 0
    t0 = time.perf_counter()
    with open(out / "deidentified.jsonl", "w", encoding="utf-8") as f:
        for note_id, text, patient in read_notes(Path(args.input), args.text_col, args.id_col, args.patient_col):
            if args.limit and n >= args.limit:
                break
            if not isinstance(text, str) or not text.strip():
                skipped += 1
                continue
            res = pipe.process(note_id, text, patient)
            total.merge(res.residual)
            for r in res.transform.records:
                actions[f"{r.label}:{r.action}"] += 1
                statuses[r.status] += 1
            f.write(json.dumps({"note_id": note_id, "text": res.text,
                                "transformations": [r.to_dict() for r in res.transform.records],
                                "review_required": res.residual.review_required}) + "\n")
            n += 1
    report = {"notes": n, "skipped_empty_or_invalid": skipped, "seconds": round(time.perf_counter() - t0, 2),
              "pipeline": pipe.describe(), "actions": dict(actions), "statuses": dict(statuses),
              "validation": total.to_dict()}
    (out / "run_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("notes", "skipped_empty_or_invalid", "seconds", "statuses")}, indent=1))
    print(json.dumps(report["validation"], indent=1))
    if report["pipeline"]["ephemeral_key"]:
        print("NOTE: no DEID_SECRET_KEY set - pseudonyms/date shifts are consistent within this run only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
