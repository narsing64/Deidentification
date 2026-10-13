"""Readers for the supplied dataset, with record-level validation.

The original files are only ever opened read-only. Validation problems are collected in a
`ValidationReport` (counts and record ids only, never text), and malformed records are skipped
instead of crashing the run.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from deid.schema import Note

LONG_NOTE_CHARS = 10_000


@dataclass
class ValidationReport:
    source: str
    n_rows: int = 0
    n_valid: int = 0
    issues: Counter = field(default_factory=Counter)
    issue_ids: dict[str, list[str]] = field(default_factory=dict)

    def add(self, issue: str, record_id: str) -> None:
        self.issues[issue] += 1
        self.issue_ids.setdefault(issue, []).append(record_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_rows": self.n_rows,
            "n_valid": self.n_valid,
            "issues": dict(self.issues),
            # ids only; capped so reports stay small
            "issue_ids": {k: v[:20] for k, v in self.issue_ids.items()},
        }


def _check_text(text: Any, rid: str, column: str, report: ValidationReport) -> bool:
    if not isinstance(text, str):
        report.add(f"{column}:not_a_string", rid)
        return False
    if not text.strip():
        report.add(f"{column}:empty", rid)
        return False
    if "�" in text:
        report.add(f"{column}:replacement_char", rid)  # kept, but flagged
    if len(text) > LONG_NOTE_CHARS:
        report.add(f"{column}:long_note_gt_{LONG_NOTE_CHARS}", rid)  # informational
    return True


def _read_table(path: Path, sep: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8")


def load_reference_pairs(csv_path: str | Path) -> tuple[list[dict[str, str]], ValidationReport]:
    """Load the annotated split: one raw note and its reference de-identified version per row."""
    path = Path(csv_path)
    df = _read_table(path, ",")
    report = ValidationReport(source=path.name, n_rows=len(df))
    missing = {"id", "raw", "de-identified"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns {sorted(missing)}")
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    for i, row in enumerate(df.itertuples(index=False)):
        rid = str(getattr(row, "id") or f"row{i}")
        if rid in seen:
            report.add("duplicate_id", rid)
            continue
        raw = row[df.columns.get_loc("raw")]
        ref = row[df.columns.get_loc("de-identified")]
        ok = _check_text(raw, rid, "raw", report) & _check_text(ref, rid, "reference", report)
        if not ok:
            continue
        seen.add(rid)
        records.append({"note_id": rid, "raw": raw, "reference": ref})
    report.n_valid = len(records)
    return records, report


_TIER_RE = re.compile(r"_(Hard|Medium)_", re.IGNORECASE)


def _tier(source: str, source_id: str) -> str:
    if source != "pdf":
        return source or "unknown"
    m = _TIER_RE.search(source_id + "_")
    return m.group(1).lower() if m else "base"


def load_annotation_meta(jsonl_path: str | Path) -> tuple[dict[str, dict[str, Any]], ValidationReport]:
    """Load per-note metadata from annotations.jsonl.

    The `changes` offsets in this file are NOT used as gold (55% do not match the raw text; see
    docs/01_dataset_inspection.md). Only source, tier and the `possible_leaks` strings are kept;
    the latter stay in memory to mark uncertain regions and are never written out.
    """
    path = Path(jsonl_path)
    report = ValidationReport(source=path.name)
    meta: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            report.n_rows += 1
            rid = f"line{lineno}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                report.add("invalid_json", rid)
                continue
            if not isinstance(rec, dict) or "id" not in rec:
                report.add("missing_id", rid)
                continue
            rid = str(rec["id"])
            if rid in meta:
                report.add("duplicate_id", rid)
                continue
            leaks = rec.get("possible_leaks") or []
            if not isinstance(leaks, list):
                report.add("possible_leaks_not_list", rid)
                leaks = []
            source = str(rec.get("source", "unknown"))
            meta[rid] = {
                "source": source,
                "tier": _tier(source, str(rec.get("source_id", ""))),
                "possible_leaks": [s for s in leaks if isinstance(s, str)],
                "n_changes": len(rec.get("changes") or []),
            }
    report.n_valid = len(meta)
    return meta, report


def load_unlabelled_notes(tsv_path: str | Path) -> tuple[list[Note], ValidationReport]:
    """Load the unannotated notes (train.tsv). Used for speed benchmarks only, never for metrics."""
    path = Path(tsv_path)
    df = _read_table(path, "\t")
    report = ValidationReport(source=path.name, n_rows=len(df))
    if "text" not in df.columns:
        raise ValueError(f"{path.name}: missing required column 'text'")
    id_col = df.columns[0] if df.columns[0] != "text" else None
    notes: list[Note] = []
    seen: set[str] = set()
    for i, row in enumerate(df.itertuples(index=False)):
        rid = f"train_{row[0] if id_col else i}"
        text = row[df.columns.get_loc("text")]
        if rid in seen:
            report.add("duplicate_id", rid)
            continue
        if not _check_text(text, rid, "text", report):
            continue
        seen.add(rid)
        notes.append(Note(note_id=rid, text=text))
    report.n_valid = len(notes)
    return notes, report
