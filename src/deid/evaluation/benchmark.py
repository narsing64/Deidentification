"""Benchmark plumbing shared by all experiments: data, timed runs, prediction files, scoring."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from deid.config import load_labels, load_yaml, resolve_path
from deid.data.gold import GoldDoc
from deid.data.loader import load_reference_pairs
from deid.evaluation.metrics import CorpusScore, score_document
from deid.schema import Entity, Region

TEST_LOG = "results/test_set_access.log"


@dataclass
class EvalNote:
    note_id: str
    text: str
    gold: list[Entity]
    ignore: list[Region]
    source: str = ""
    tier: str = ""
    unannotated: list[str] = field(default_factory=list)
    adjust_ignore: list[Region] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def load_eval_notes(split: str, bench_cfg: dict[str, Any] | None = None, *, allow_test: bool = False,
                    purpose: str = "") -> list[EvalNote]:
    """Load annotated notes of one split. The held-out test split requires an explicit opt-in,
    and every access is appended to results/test_set_access.log."""
    cfg = bench_cfg or load_yaml("configs/benchmark.yaml")
    if split not in ("validation", "test"):
        raise ValueError("split must be 'validation' or 'test'")
    if split == "test":
        if not allow_test:
            raise PermissionError("The held-out test split is reserved for the final evaluation (Experiment 8).")
        log = resolve_path(TEST_LOG)
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{purpose or 'unspecified'}\n")
    processed = resolve_path(cfg["data"]["processed_dir"])
    ids = set(json.loads((processed / "splits.json").read_text(encoding="utf-8"))[split])
    texts = {r["note_id"]: r["raw"] for r in load_reference_pairs(resolve_path(cfg["data"]["annotated_csv"]))[0]}
    notes = []
    with open(processed / "gold.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["note_id"] not in ids:
                continue
            doc = GoldDoc.from_dict(row)
            notes.append(EvalNote(doc.note_id, texts[doc.note_id], doc.spans, doc.ignore,
                                  row.get("source", ""), row.get("tier", ""),
                                  unannotated=doc.unannotated, adjust_ignore=doc.adjust_ignore))
    notes.sort(key=lambda n: n.note_id)
    return notes


def run_detector(detect: Callable[[str], list[Entity]], notes: Iterable[EvalNote],
                 progress: bool = False) -> tuple[dict[str, list[Entity]], dict[str, float]]:
    preds: dict[str, list[Entity]] = {}
    secs: dict[str, float] = {}
    notes = list(notes)
    for i, n in enumerate(notes):
        t0 = time.perf_counter()
        preds[n.note_id] = detect(n.text)
        secs[n.note_id] = time.perf_counter() - t0
        if progress and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(notes)} notes", flush=True)
    return preds, secs


def save_predictions(path: str | Path, preds: dict[str, list[Entity]], secs: dict[str, float] | None = None,
                     meta: dict[str, Any] | None = None) -> None:
    """Offsets + labels + scores only; never text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if meta:
            f.write(json.dumps({"_meta": meta}) + "\n")
        for nid in sorted(preds):
            row = {"note_id": nid, "entities": [e.to_dict() for e in preds[nid]]}
            if secs and nid in secs:
                row["secs"] = round(secs[nid], 5)
            f.write(json.dumps(row) + "\n")


def load_predictions(path: str | Path) -> tuple[dict[str, list[Entity]], dict[str, float], dict[str, Any]]:
    preds, secs, meta = {}, {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
                continue
            preds[row["note_id"]] = [Entity.from_dict(e) for e in row["entities"]]
            if "secs" in row:
                secs[row["note_id"]] = row["secs"]
    return preds, secs, meta


PROFILES = ("adjusted", "as_is")


def score_predictions(preds: dict[str, list[Entity]], notes: list[EvalNote],
                      bench_cfg: dict[str, Any] | None = None, bootstrap: bool = True,
                      profile: str = "adjusted") -> dict[str, Any]:
    """profile="as_is": the reference exactly as derived. profile="adjusted": additionally applies
    coverage rules R1/R2 (docs/04_reference_and_evaluation_protocol.md)."""
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    cfg = bench_cfg or load_yaml("configs/benchmark.yaml")
    ev = cfg.get("evaluation", {})
    cats = load_labels()["categories"]
    cs = CorpusScore()
    for n in notes:
        if profile == "adjusted":
            ds = score_document(n.gold, preds.get(n.note_id, []), list(n.ignore) + list(n.adjust_ignore),
                                excluded_labels=n.unannotated)
        else:
            ds = score_document(n.gold, preds.get(n.note_id, []), n.ignore)
        cs.add(n.note_id, ds)
    return cs.summary(bootstrap=ev.get("bootstrap_resamples", 1000) if bootstrap else 0,
                      seed=ev.get("bootstrap_seed", 7), confidence=ev.get("confidence", 0.95),
                      categories=cats)


def profile_view(note: EvalNote, profile: str) -> tuple[list[Entity], list[Region]]:
    """(reference spans, ignore regions) for a profile; excluded categories become whole-note
    filtering of both sides, applied by `filter_excluded`."""
    if profile == "adjusted":
        gold = [g for g in note.gold if g.label not in set(note.unannotated)]
        return gold, list(note.ignore) + list(note.adjust_ignore)
    return list(note.gold), list(note.ignore)


def filter_excluded(note: EvalNote, preds: list[Entity], profile: str) -> list[Entity]:
    if profile == "adjusted" and note.unannotated:
        ex = set(note.unannotated)
        return [p for p in preds if p.label not in ex]
    return preds


def evaluate_system(preds: dict[str, list[Entity]], notes: list[EvalNote], with_errors: bool = True) -> dict[str, Any]:
    """Scores under both reference profiles (+ sanitised error analysis on the adjusted profile)."""
    from deid.evaluation.error_analysis import error_report  # local import: avoids a cycle

    res: dict[str, Any] = {p: score_predictions(preds, notes, profile=p) for p in PROFILES}
    if with_errors:
        views = []
        for n in notes:
            gold, ign = profile_view(n, "adjusted")
            views.append(EvalNote(n.note_id, n.text, gold, ign, n.source, n.tier))
        res["errors_adjusted"] = error_report(
            views, {n.note_id: filter_excluded(n, preds.get(n.note_id, []), "adjusted") for n in notes})
    return res


def write_json(path: str | Path, obj: Any) -> None:
    path = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
