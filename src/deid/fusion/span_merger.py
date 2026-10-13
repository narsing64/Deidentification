"""Combine regex and model entities into one non-overlapping set (no agreement required).

Rules, applied in order and identically for every model:
1. Invalid offsets (outside the text, empty, reversed) are dropped. Edge whitespace is trimmed.
   Spans without any letter or digit are dropped.
2. Exact duplicates (same start, end, label) collapse into one entity. Sources are joined, and
   the highest score is kept.
3. Overlapping spans form clusters (connected components of character overlap).
   a. All spans in a cluster share a label (nested, partial or different-length detections of the
      same thing): the cluster becomes the UNION span. This is recall-first: no detected
      character is lost.
   b. Labels conflict, and the cluster contains regex spans of a regex-trusted category
      (dates, times, phones, emails, IDs, ...): those regex spans keep their exact
      boundaries and labels. Every other span keeps only the parts outside them (fragments with
      at least one letter/digit). This is how "2101-03-04 06:20AM" tagged DATE by a model becomes
      DATE + TIME.
   c. Labels conflict with no trusted regex span: the highest-score span wins (ties: longer span,
      then the category order in configs/labels.yaml). The losers keep only their parts outside
      the winner.
   The cluster is re-resolved until no overlaps remain. Rule (b)/(c) fragments mean coverage is
   never reduced by a label conflict.
4. Repeated entities: identical strings at different positions stay separate entities.
   Optionally (`propagate_repeats`), other exact occurrences of a detected PERSON string are also
   marked (source "propagated").
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from deid.schema import Entity

_ALNUM = re.compile(r"[^\W_]")
_EDGE = " \t\r\n\f\v,;:"
DEFAULT_ORDER = ["PERSON", "DATE", "TIME", "AGE", "ORGANIZATION", "LOCATION", "PHONE", "EMAIL", "ID", "OTHER_PHI"]


@dataclass
class FusionConfig:
    regex_trusted: set[str] = field(default_factory=lambda: {"DATE", "TIME", "PHONE", "EMAIL", "ID", "AGE", "OTHER_PHI"})
    propagate_repeats: bool = False
    category_order: list[str] = field(default_factory=lambda: list(DEFAULT_ORDER))
    regex_source: str = "regex"

    @classmethod
    def from_config(cls, detection_cfg: dict) -> "FusionConfig":
        f = detection_cfg.get("fusion", {}) or {}
        return cls(regex_trusted=set(f.get("regex_trusted_categories") or []),
                   propagate_repeats=bool(f.get("propagate_repeats", False)))


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s] in _EDGE:
        s += 1
    while e > s and text[e - 1] in _EDGE:
        e -= 1
    return s, e


def _join_sources(*names: str) -> str:
    parts = sorted({p for n in names for p in n.split("|") if p})
    return "|".join(parts)


def clean(text: str, ents: Iterable[Entity], stats: Counter) -> list[Entity]:
    out = []
    n = len(text)
    for e in ents:
        if not (0 <= e.start < e.end <= n):
            stats["invalid_offsets"] += 1
            continue
        s, t = _trim(text, e.start, e.end)
        if (s, t) != (e.start, e.end):
            stats["trimmed"] += 1
        if s >= t or not _ALNUM.search(text[s:t]):
            stats["dropped_no_alnum"] += 1
            continue
        out.append(e.with_(start=s, end=t))
    return out


def dedupe(ents: Iterable[Entity], stats: Counter) -> list[Entity]:
    best: dict[tuple, Entity] = {}
    for e in ents:
        k = (e.start, e.end, e.label)
        if k in best:
            stats["exact_duplicates"] += 1
            o = best[k]
            scores = [x for x in (o.score, e.score) if x is not None]
            best[k] = o.with_(source=_join_sources(o.source, e.source), score=max(scores) if scores else None)
        else:
            best[k] = e
    return list(best.values())


def _clusters(ents: list[Entity]) -> list[list[Entity]]:
    out: list[list[Entity]] = []
    end = -1
    for e in sorted(ents, key=lambda x: (x.start, -x.end)):
        if out and e.start < end:
            out[-1].append(e)
            end = max(end, e.end)
        else:
            out.append([e])
            end = e.end
    return out


def _subtract(text: str, e: Entity, blockers: Sequence[Entity]) -> list[Entity]:
    """Parts of `e` outside all blockers, trimmed; only fragments with a letter/digit survive."""
    pieces = [(e.start, e.end)]
    for b in sorted(blockers, key=lambda x: x.start):
        nxt = []
        for s, t in pieces:
            if b.end <= s or b.start >= t:
                nxt.append((s, t))
                continue
            if s < b.start:
                nxt.append((s, b.start))
            if b.end < t:
                nxt.append((b.end, t))
        pieces = nxt
    out = []
    for s, t in pieces:
        s, t = _trim(text, s, t)
        if s < t and _ALNUM.search(text[s:t]):
            out.append(e.with_(start=s, end=t))
    return out


class _Resolver:
    def __init__(self, text: str, cfg: FusionConfig, stats: Counter):
        self.text, self.cfg, self.stats = text, cfg, stats
        self.rank = {c: i for i, c in enumerate(cfg.category_order)}

    def _union(self, cluster: list[Entity]) -> Entity:
        s = min(e.start for e in cluster)
        t = max(e.end for e in cluster)
        scores = [e.score for e in cluster if e.score is not None]
        first = max(cluster, key=lambda e: (e.end - e.start))
        return first.with_(start=s, end=t, score=max(scores) if scores else None,
                           source=_join_sources(*(e.source for e in cluster)))

    def _is_anchor(self, e: Entity) -> bool:
        return self.cfg.regex_source in e.source.split("|") and e.label in self.cfg.regex_trusted

    def resolve(self, ents: list[Entity], depth: int = 0) -> list[Entity]:
        if depth > 20:
            raise RuntimeError("fusion did not converge")
        result: list[Entity] = []
        changed = False
        for cl in _clusters(ents):
            if len(cl) == 1:
                result.append(cl[0])
                continue
            changed = True
            labels = {e.label for e in cl}
            if len(labels) == 1:
                self.stats["union_same_label"] += 1
                result.append(self._union(cl))
                continue
            anchors = [e for e in cl if self._is_anchor(e)]
            if anchors:
                self.stats["conflict_regex_anchor"] += 1
                keep = self.resolve(anchors, depth + 1)
                others = [e for e in cl if not self._is_anchor(e)]
            else:
                self.stats["conflict_by_score"] += 1
                win = max(cl, key=lambda e: (e.score if e.score is not None else 0.0, e.end - e.start,
                                             -self.rank.get(e.label, 99)))
                keep = [win]
                others = [e for e in cl if e is not win]
            frags = [f for o in others for f in _subtract(self.text, o, keep)]
            result.extend(keep)
            result.extend(frags)
        if changed:
            # fragments of different losers may still overlap each other: resolve again
            if any(a.end > b.start for a, b in zip(sorted(result), sorted(result)[1:])):
                return self.resolve(result, depth + 1)
        return sorted(result, key=lambda e: (e.start, e.end))


def propagate(text: str, ents: list[Entity], stats: Counter, labels: Sequence[str] = ("PERSON",)) -> list[Entity]:
    found = list(ents)
    strings = {text[e.start:e.end] for e in ents if e.label in labels and len(text[e.start:e.end]) >= 3}
    for s in sorted(strings, key=len, reverse=True):
        for m in re.finditer(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text):
            if any(m.start() < e.end and e.start < m.end() for e in found):
                continue
            found.append(Entity(m.start(), m.end(), "PERSON", source="propagated", score=None, fine_label="repeat"))
            stats["propagated"] += 1
    return sorted(found, key=lambda e: (e.start, e.end))


def fuse(text: str, sources: Sequence[Iterable[Entity]], cfg: FusionConfig | None = None,
         return_stats: bool = False):
    cfg = cfg or FusionConfig()
    stats: Counter = Counter()
    ents = clean(text, (e for src in sources for e in src), stats)
    ents = dedupe(ents, stats)
    out = _Resolver(text, cfg, stats).resolve(ents)
    if cfg.propagate_repeats:
        out = propagate(text, out, stats)
    for a, b in zip(out, out[1:]):
        if a.end > b.start:
            raise AssertionError("fusion produced overlapping spans")
    return (out, dict(stats)) if return_stats else out
