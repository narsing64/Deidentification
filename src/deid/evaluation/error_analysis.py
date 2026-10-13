"""Sanitised error analysis: categories, shapes and template context words only.

No span text is ever emitted. A left-context word is used only if it lies OUTSIDE every reference
span, ignore region and predicted span of that note (otherwise it becomes "<phi>"). A context
word pair is printed only if it occurs in at least `min_context_docs` DIFFERENT notes; anything
rarer is "<rare>". A name repeated many times inside one note can therefore never pass.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from deid.privacy import safe_shape
from deid.schema import Entity, Region

_WORD = re.compile(r"[A-Za-z]+[:#.]?")
MIN_CONTEXT_DOCS = 10


def _left_ctx(text: str, start: int, blocked: Sequence[tuple[int, int]], n_words: int = 2) -> str:
    lo = max(0, start - 40)
    words = []
    for m in _WORD.finditer(text, lo, start):
        a, b = m.span()
        words.append("<phi>" if any(a < e and s < b for s, e in blocked) else m.group().lower())
    return " ".join(words[-n_words:]) or "<start>"


def _overlaps(a: Entity | Region, items: Sequence[Entity | Region]) -> list:
    return [x for x in items if a.start < x.end and x.start < a.end]


def classify_errors(text: str, gold: Sequence[Entity], pred: Sequence[Entity],
                    ignore: Sequence[Region] = (), note_id: str = "") -> tuple[list[dict], list[dict]]:
    """Return (false negatives, false positives) as sanitised records under STRICT matching."""
    pred = [p for p in pred if not _overlaps(p, ignore)]
    blocked = sorted([(x.start, x.end) for x in list(gold) + list(ignore) + list(pred)])
    p_keys = {(p.start, p.end, p.label) for p in pred}
    g_keys = {(g.start, g.end, g.label) for g in gold}
    fns, fps = [], []
    for g in gold:
        if (g.start, g.end, g.label) in p_keys:
            continue
        ov = _overlaps(g, pred)
        exact = [p for p in ov if (p.start, p.end) == (g.start, g.end)]
        kind = "wrong_label" if exact else ("boundary" if ov else "missed")
        if kind == "boundary":
            p = ov[0]
            kind += ":" + ("pred_longer" if p.start <= g.start and p.end >= g.end else
                           "pred_shorter" if p.start >= g.start and p.end <= g.end else "shifted")
            if len(ov) > 1:
                kind = "boundary:split_into_%d" % len(ov)
        fns.append({"label": g.label, "fine": g.fine_label, "kind": kind,
                    "pred_labels": sorted({p.label for p in (exact or ov)}),
                    "shape": safe_shape(text[g.start:g.end]), "ctx": _left_ctx(text, g.start, blocked), "note": note_id})
    for p in pred:
        if (p.start, p.end, p.label) in g_keys:
            continue
        ov = _overlaps(p, gold)
        exact = [g for g in ov if (g.start, g.end) == (p.start, p.end)]
        kind = "wrong_label" if exact else ("boundary" if ov else "spurious")
        fps.append({"label": p.label, "rule": p.fine_label, "source": p.source, "kind": kind,
                    "gold_labels": sorted({g.label for g in ov}),
                    "shape": safe_shape(text[p.start:p.end]), "ctx": _left_ctx(text, p.start, blocked), "note": note_id})
    return fns, fps


def aggregate(records: Iterable[dict], keys: Sequence[str], min_context_docs: int = MIN_CONTEXT_DOCS,
              top: int = 15) -> list[dict[str, Any]]:
    """Group sanitised records; contexts seen in fewer than `min_context_docs` notes -> '<rare>'."""
    records = list(records)
    docs_by_ctx: dict[str, set] = defaultdict(set)
    for r in records:
        docs_by_ctx[r["ctx"]].add(r.get("note", ""))
    groups: Counter = Counter()
    ctx_by_group: dict[tuple, Counter] = defaultdict(Counter)
    for r in records:
        k = tuple(r.get(x) if not isinstance(r.get(x), list) else ",".join(r[x]) for x in keys)
        groups[k] += 1
        ctx_by_group[k][r["ctx"] if len(docs_by_ctx[r["ctx"]]) >= min_context_docs else "<rare>"] += 1
    out = []
    for k, n in groups.most_common(top):
        out.append({**dict(zip(keys, k)), "count": n, "top_contexts": ctx_by_group[k].most_common(3)})
    return out


def error_report(notes, preds: dict[str, list[Entity]], min_context_docs: int = MIN_CONTEXT_DOCS) -> dict[str, Any]:
    fns, fps = [], []
    for n in notes:
        a, b = classify_errors(n.text, n.gold, preds.get(n.note_id, []), n.ignore, n.note_id)
        fns.extend(a)
        fps.extend(b)
    return {
        "fn_total": len(fns), "fp_total": len(fps),
        "fn_by_kind": dict(Counter(r["kind"] for r in fns).most_common()),
        "fp_by_kind": dict(Counter(r["kind"] for r in fps).most_common()),
        "fn_groups": aggregate(fns, ["label", "fine", "kind", "shape"], min_context_docs),
        "fp_groups": aggregate(fps, ["label", "rule", "kind", "shape"], min_context_docs),
    }


def render_markdown(rep: dict[str, Any], title: str) -> str:
    lines = [f"### {title}", "",
             f"False negatives: {rep['fn_total']} — by kind: {rep['fn_by_kind']}",
             f"False positives: {rep['fp_total']} — by kind: {rep['fp_by_kind']}", "",
             "| FN label | fine | kind | shape | n | frequent left context |", "|---|---|---|---|---|---|"]
    for g in rep["fn_groups"]:
        lines.append(f"| {g['label']} | {g['fine']} | {g['kind']} | `{g['shape']}` | {g['count']} | "
                     + "; ".join(f"{c} ({k})" for c, k in g["top_contexts"]) + " |")
    lines += ["", "| FP label | rule/fine | kind | shape | n | frequent left context |", "|---|---|---|---|---|---|"]
    for g in rep["fp_groups"]:
        lines.append(f"| {g['label']} | {g['rule']} | {g['kind']} | `{g['shape']}` | {g['count']} | "
                     + "; ".join(f"{c} ({k})" for c, k in g["top_contexts"]) + " |")
    return "\n".join(lines)
