"""Shared evaluation metrics. Every system (regex, models, fusion) is scored by this module only.

Definitions
-----------
* Strict entity match: same start, same end AND same normalised category.
* Label-agnostic entity match: same start and same end; the category is ignored. Partial overlaps
  are NOT matches.
* One-to-one matching: predictions are de-duplicated on the scoring key before matching
  ((start, end, label) for strict, (start, end) for agnostic). Reference spans are unique and
  non-overlapping, so set intersection is an exact one-to-one matching. Duplicate predictions can
  therefore never add true positives. Two labels on one span count once under agnostic scoring;
  under strict scoring the wrong-label copy is a false positive.
* Ignore regions: predictions overlapping an ignore region are removed before scoring and counted
  separately. Reference spans never overlap ignore regions.
* Micro-averaging: TP/FP/FN are summed over all documents (and categories) before computing P/R/F1.
  Macro-F1 (unweighted mean over categories present in the reference) is reported as secondary.
* Empty denominators: P is None (undefined) when TP+FP = 0; R is None when TP+FN = 0;
  F1 = 2TP / (2TP+FP+FN), None when that denominator is 0. None is rendered as "n/a".
* Partial-overlap and character-coverage numbers are separate DIAGNOSTICS and never mixed into
  the exact-span metrics.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from deid.schema import Entity, Region


# ---------------------------------------------------------------- basic counts
@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __iadd__(self, other: "Counts") -> "Counts":
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def prf(self) -> dict[str, float | None]:
        return prf(self.tp, self.fp, self.fn)

    def to_dict(self) -> dict[str, Any]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, **self.prf()}


def prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return {"precision": p, "recall": r, "f1": f}


# ---------------------------------------------------------------- per-document scoring
@dataclass
class DocScore:
    strict: Counts = field(default_factory=Counts)
    agnostic: Counts = field(default_factory=Counts)
    per_category: dict[str, Counts] = field(default_factory=dict)
    confusion: Counter = field(default_factory=Counter)         # (reference label, predicted label)
    fn_breakdown: Counter = field(default_factory=Counter)      # missed | boundary | wrong_label
    fp_breakdown: Counter = field(default_factory=Counter)      # spurious | boundary | wrong_label
    wrong_label_spans: int = 0
    duplicate_predictions: int = 0
    ignored_predictions: int = 0
    overlap: Counter = field(default_factory=Counter)           # diagnostics only


def drop_ignored(pred: Iterable[Entity], ignore: Sequence[Region]) -> tuple[list[Entity], int]:
    kept, dropped = [], 0
    for p in pred:
        if any(p.start < r.end and r.start < p.end for r in ignore):
            dropped += 1
        else:
            kept.append(p)
    return kept, dropped


def _covered_chars(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _intersection_len(a: tuple[int, int], merged: list[tuple[int, int]]) -> int:
    return sum(max(0, min(a[1], e) - max(a[0], s)) for s, e in merged)


def score_document(gold: Sequence[Entity], pred: Sequence[Entity], ignore: Sequence[Region] = (),
                   excluded_labels: Iterable[str] = ()) -> DocScore:
    """`excluded_labels`: categories the reference did not annotate in this note. Reference spans
    and predictions with those labels are both removed (counted as ignored predictions)."""
    ds = DocScore()
    excluded = set(excluded_labels)
    if excluded:
        gold = [g for g in gold if g.label not in excluded]
        n_before = len(pred)
        pred = [p for p in pred if p.label not in excluded]
        ds.ignored_predictions += n_before - len(pred)
    pred, n_ign = drop_ignored(pred, ignore)
    ds.ignored_predictions += n_ign

    g_strict = {(e.start, e.end, e.label) for e in gold}
    if len(g_strict) != len(gold):
        raise ValueError("reference spans must be unique")
    p_strict = {(e.start, e.end, e.label) for e in pred}
    ds.duplicate_predictions = len(pred) - len(p_strict)

    tp = g_strict & p_strict
    ds.strict = Counts(len(tp), len(p_strict - g_strict), len(g_strict - p_strict))

    g_spans = {(s, e) for s, e, _ in g_strict}
    p_spans = {(s, e) for s, e, _ in p_strict}
    ds.agnostic = Counts(len(g_spans & p_spans), len(p_spans - g_spans), len(g_spans - p_spans))

    for lab in {l for *_, l in g_strict} | {l for *_, l in p_strict}:
        g_c = {x for x in g_strict if x[2] == lab}
        p_c = {x for x in p_strict if x[2] == lab}
        ds.per_category[lab] = Counts(len(g_c & p_c), len(p_c - g_c), len(g_c - p_c))

    gold_label = {(s, e): l for s, e, l in g_strict}
    pred_labels: dict[tuple[int, int], set[str]] = {}
    for s, e, l in p_strict:
        pred_labels.setdefault((s, e), set()).add(l)
    for span in g_spans & p_spans:
        for pl in sorted(pred_labels[span]):
            ds.confusion[(gold_label[span], pl)] += 1
        if gold_label[span] not in pred_labels[span]:
            ds.wrong_label_spans += 1

    # error breakdowns (strict view)
    p_list = sorted(p_spans)
    g_list = sorted(g_spans)
    for s, e, l in g_strict - p_strict:
        if (s, e) in p_spans:
            ds.fn_breakdown["wrong_label"] += 1
        elif any(ps < e and s < pe for ps, pe in p_list):
            ds.fn_breakdown["boundary"] += 1
        else:
            ds.fn_breakdown["missed"] += 1
    for s, e, l in p_strict - g_strict:
        if (s, e) in g_spans:
            ds.fp_breakdown["wrong_label"] += 1
        elif any(gs < e and s < ge for gs, ge in g_list):
            ds.fp_breakdown["boundary"] += 1
        else:
            ds.fp_breakdown["spurious"] += 1

    # partial-overlap diagnostics (label-agnostic)
    ds.overlap["gold"] = len(g_list)
    ds.overlap["gold_overlapped"] = sum(1 for gs, ge in g_list if any(ps < ge and gs < pe for ps, pe in p_list))
    ds.overlap["pred"] = len(p_list)
    ds.overlap["pred_overlapping"] = sum(1 for ps, pe in p_list if any(gs < pe and ps < ge for gs, ge in g_list))
    merged_pred = _covered_chars(p_list)
    ds.overlap["gold_chars"] = sum(e - s for s, e in g_list)
    ds.overlap["gold_chars_covered"] = sum(_intersection_len(g, merged_pred) for g in g_list)
    ds.overlap["gold_fully_covered"] = sum(1 for g in g_list if _intersection_len(g, merged_pred) == g[1] - g[0])
    return ds


# ---------------------------------------------------------------- corpus aggregation
@dataclass
class CorpusScore:
    docs: list[DocScore] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)

    def add(self, doc_id: str, ds: DocScore) -> None:
        self.doc_ids.append(doc_id)
        self.docs.append(ds)

    def _sum(self, attr: str) -> Counts:
        total = Counts()
        for d in self.docs:
            total += getattr(d, attr)
        return total

    def summary(self, bootstrap: int = 0, seed: int = 0, confidence: float = 0.95,
                categories: Sequence[str] | None = None) -> dict[str, Any]:
        strict, agnostic = self._sum("strict"), self._sum("agnostic")
        per_cat: dict[str, Counts] = {}
        conf, fnb, fpb, ov = Counter(), Counter(), Counter(), Counter()
        wrong = dup = ign = 0
        for d in self.docs:
            for lab, c in d.per_category.items():
                per_cat.setdefault(lab, Counts())
                per_cat[lab] += c
            conf.update(d.confusion)
            fnb.update(d.fn_breakdown)
            fpb.update(d.fp_breakdown)
            ov.update(d.overlap)
            wrong += d.wrong_label_spans
            dup += d.duplicate_predictions
            ign += d.ignored_predictions
        cats = list(categories) if categories else sorted(per_cat)
        ref_cats = [c for c in cats if c in per_cat and per_cat[c].tp + per_cat[c].fn > 0]
        f1s = [per_cat[c].prf()["f1"] for c in ref_cats]
        out: dict[str, Any] = {
            "documents": len(self.docs),
            "strict": strict.to_dict(),
            "agnostic": agnostic.to_dict(),
            "macro_f1_strict": float(np.mean([f for f in f1s if f is not None])) if f1s else None,
            "per_category": {c: per_cat[c].to_dict() for c in cats if c in per_cat},
            "wrong_label_exact_span": wrong,
            "confusion_exact_span": {f"{g}->{p}": n for (g, p), n in sorted(conf.items())},
            "fn_breakdown_strict": dict(fnb),
            "fp_breakdown_strict": dict(fpb),
            "duplicate_predictions": dup,
            "ignored_predictions": ign,
            "overlap_diagnostics": {
                "gold_overlap_recall": ov["gold_overlapped"] / ov["gold"] if ov["gold"] else None,
                "pred_overlap_precision": ov["pred_overlapping"] / ov["pred"] if ov["pred"] else None,
                "gold_char_coverage": ov["gold_chars_covered"] / ov["gold_chars"] if ov["gold_chars"] else None,
                "gold_fully_covered_rate": ov["gold_fully_covered"] / ov["gold"] if ov["gold"] else None,
            },
        }
        if bootstrap:
            out["bootstrap_ci"] = self.bootstrap(bootstrap, seed, confidence)
        return out

    def bootstrap(self, n: int, seed: int, confidence: float) -> dict[str, dict[str, list[float | None]]]:
        """Document-level percentile bootstrap CIs for micro P/R/F1."""
        rng = np.random.default_rng(seed)
        res: dict[str, dict[str, list[float | None]]] = {}
        for kind in ("strict", "agnostic"):
            arr = np.array([[getattr(d, kind).tp, getattr(d, kind).fp, getattr(d, kind).fn] for d in self.docs])
            if len(arr) == 0:
                continue
            samples = {"precision": [], "recall": [], "f1": []}
            for _ in range(n):
                tp, fp, fn = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
                for k, v in prf(int(tp), int(fp), int(fn)).items():
                    if v is not None:
                        samples[k].append(v)
            lo, hi = (1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100
            res[kind] = {k: ([float(np.percentile(v, lo)), float(np.percentile(v, hi))] if v else [None, None])
                         for k, v in samples.items()}
        return res


def fmt(x: float | None, digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"
