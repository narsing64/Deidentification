"""Derive reference PHI spans by aligning each raw note with its reference de-identified version.

Why: the offsets in annotations.jsonl are inconsistent (docs/01_dataset_inspection.md, section 2.1).
The reference text replaces each PHI span with a placeholder such as ``[DATE]``.

Algorithm (documented in docs/04_gold_derivation.md):
1. Tokenise both texts into placeholder | word | single punctuation tokens. Whitespace is ignored,
   so whitespace re-flow in the reference cannot break the alignment.
2. Align token sequences with difflib (autojunk off).
3. Merge tiny punctuation-only "equal" blocks that sit between two changed blocks. This avoids
   spurious anchors such as the "-" inside a date range.
4. Each changed block that contains placeholders yields reference spans:
   * one placeholder  -> the whole raw block (trimmed of whitespace and leading/trailing ``:;,``)
   * several          -> split with label-typed patterns; accepted only if the split is unique
                         (lazy and greedy matching agree), otherwise an ignore region.
5. Uncertain material becomes an IGNORE region (excluded from scoring, counted):
   invalid placeholder labels (leaked values), unexplained word edits, excluded categories,
   unsplittable groups, and occurrences of strings flagged in `possible_leaks`.
No text is ever logged; only counts are returned.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, NamedTuple

from deid.schema import Entity, Region

PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z_]*\]")
TOKEN_RE = re.compile(r"\[[A-Z][A-Z_]*\]|\w+|[^\w\s]")
_ALNUM_RE = re.compile(r"[^\W_]")
_TRIM_CHARS = " \t\r\n\f\v:;,"

# ---------------------------------------------------------------- typed split patterns
_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
          r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
_ORD = r"(?:st|nd|rd|th)?"
DATE_P = (r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
          rf"|{_MONTH}\s+\d{{1,2}}{_ORD},?\s+\d{{4}}|\d{{1,2}}{_ORD}\s+{_MONTH},?\s+\d{{4}}"
          rf"|{_MONTH}\s+\d{{4}}|{_MONTH}\s+\d{{1,2}}{_ORD}|\d{{1,2}}[-/]\d{{1,2}}|\d{{4}})")
TIME_P = r"(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp]\.?[Mm]\.?)?|\d{1,2}\s*[AaPp]\.?[Mm]\.?)"
DATETIME_P = rf"(?:{DATE_P}(?:\s*,?\s*|T){TIME_P}|{TIME_P}|{DATE_P})"
PHONE_P = (r"(?:(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s*|\d{3}[\s.-])\d{3}[\s.-]?\d{4}"
           r"|\+\d{1,3}[\s.-]?\d[\d\s.-]{6,}\d)")
# Typed groups must start and end on token boundaries, so a greedy neighbour cannot cut a date,
# phone number or ID in half (e.g. "01/02/1980" must never split into "01/" + "02/19" + "80").
_B_START, _B_END = r"(?<![\w/.:-])", r"(?![\w/:-])"
TYPED_PATTERNS: dict[str, str] = {
    k: f"{_B_START}{v}{_B_END}" for k, v in {
        "DATE": DATE_P, "DATE_OF_BIRTH": DATE_P, "AUTOMATED_DATE": DATE_P,
        "TIME": TIME_P, "DATETIME": DATETIME_P, "TIMESTAMP": DATETIME_P,
        "SSN": r"\d{3}-\d{2}-\d{4}",
        "PHONE": PHONE_P,
        "EMAIL": r"[^\s@]+@[^\s@]+",
    }.items()
}


_DATE_COMPONENT_RE = re.compile(rf"\d{{1,4}}[-/.]\d{{1,2}}|{_MONTH}|\b\d{{4}}\b", re.IGNORECASE)


class Tok(NamedTuple):
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Tok]:
    return [Tok(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


# ---------------------------------------------------------------- label handling
class ReferenceLabels:
    """Maps reference placeholder labels to normalised categories (configs/labels.yaml)."""

    def __init__(self, labels_cfg: dict[str, Any]):
        ref = labels_cfg["reference"]
        self.map: dict[str, str] = dict(ref["map"])
        self.excluded: set[str] = set(ref.get("excluded") or [])
        self.corrupted = [(re.compile(p), v) for p, v in (ref.get("corrupted_patterns") or {}).items()]
        self.time_only_to_time: set[str] = set(ref.get("time_only_to_time") or [])

    def normalise_by_content(self, canonical_fine: str, norm: str, span: str) -> str:
        """Documented content rule: a time-only DATETIME/TIMESTAMP span is a TIME."""
        if canonical_fine in self.time_only_to_time and not _DATE_COMPONENT_RE.search(span):
            return "TIME"
        return norm

    def resolve(self, fine: str) -> tuple[str, str | None, str]:
        """Return (status, normalised_label, canonical_fine) with status in ok|excluded|invalid."""
        canonical = fine
        if fine not in self.map and fine not in self.excluded:
            for pat, target in self.corrupted:
                if pat.match(fine):
                    canonical = target
                    break
        if canonical in self.excluded:
            return "excluded", None, canonical
        if canonical in self.map:
            return "ok", self.map[canonical], canonical
        return "invalid", None, canonical


# ---------------------------------------------------------------- result container
@dataclass
class GoldDoc:
    note_id: str
    spans: list[Entity] = field(default_factory=list)
    ignore: list[Region] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)
    # "adjusted" evaluation profile (docs/04_reference_and_evaluation_protocol.md):
    unannotated: list[str] = field(default_factory=list)       # categories the reference skipped in this note
    adjust_ignore: list[Region] = field(default_factory=list)  # unlabelled full dates in annotated notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "spans": [e.to_dict() for e in self.spans],
            "ignore": [r.to_dict() for r in self.ignore],
            "unannotated": list(self.unannotated),
            "adjust_ignore": [r.to_dict() for r in self.adjust_ignore],
            "stats": dict(self.stats),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldDoc":
        reg = lambda xs: [Region(int(x["start"]), int(x["end"]), str(x["reason"])) for x in xs or []]
        return cls(
            note_id=d["note_id"],
            spans=[Entity.from_dict(x) for x in d["spans"]],
            ignore=reg(d["ignore"]),
            stats=Counter(d.get("stats") or {}),
            unannotated=list(d.get("unannotated") or []),
            adjust_ignore=reg(d.get("adjust_ignore")),
        )


# ---------------------------------------------------------------- helpers
def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in _TRIM_CHARS:
        start += 1
    while end > start and text[end - 1] in _TRIM_CHARS:
        end -= 1
    return start, end


def _is_punct_token(tok: str) -> bool:
    return not _ALNUM_RE.search(tok) and not PLACEHOLDER_RE.fullmatch(tok)


def _is_punct_only(toks: Iterable[Tok]) -> bool:
    return all(not _ALNUM_RE.search(t.text) for t in toks)


def _merge_punct_anchors(opcodes: list[tuple], a: list[Tok], b: list[Tok]) -> list[tuple]:
    """Fold short punctuation-only equal blocks into neighbouring changed blocks (see docstring)."""
    ops = [list(op) for op in opcodes]
    changed = True
    while changed:
        changed = False
        for k in range(1, len(ops) - 1):
            tag, i1, i2, j1, j2 = ops[k]
            prev, nxt = ops[k - 1], ops[k + 1]
            if tag != "equal" or prev[0] == "equal" or nxt[0] == "equal":
                continue
            if i2 - i1 > 2 or not _is_punct_only(a[i1:i2]):
                continue
            has_ph = any(PLACEHOLDER_RE.fullmatch(t.text) for t in b[prev[3]:prev[4]] + b[nxt[3]:nxt[4]])
            if not has_ph:
                continue
            ops[k - 1:k + 2] = [["replace", prev[1], nxt[2], prev[3], nxt[4]]]
            changed = True
            break
    return [tuple(op) for op in ops]


def _literal_pattern(lit: str) -> str:
    parts = re.findall(r"\s+|\S+", lit)
    return "".join(r"\s*" if p.isspace() else r"\s*" + re.escape(p) + r"\s*" for p in parts)


def _split_group(raw_seg: str, ref_seg: str) -> list[tuple[int, int, str]] | None:
    """Split a raw segment among several adjacent placeholders. Returns None if not unique."""
    parts: list[tuple[str, str]] = []
    pos = 0
    for m in PLACEHOLDER_RE.finditer(ref_seg):
        if m.start() > pos:
            parts.append(("lit", ref_seg[pos:m.start()]))
        parts.append(("ph", m.group()[1:-1]))
        pos = m.end()
    if pos < len(ref_seg):
        parts.append(("lit", ref_seg[pos:]))

    def build(typed: bool, greedy: bool) -> re.Pattern:
        out, g = [], 0
        for kind, val in parts:
            if kind == "lit":
                out.append(_literal_pattern(val))
            else:
                body = TYPED_PATTERNS.get(val) if typed else None
                body = body or ("(?:.+)" if greedy else "(?:.+?)")
                out.append(f"(?P<g{g}>{body})")
                g += 1
        return re.compile(r"\s*" + "".join(out) + r"\s*", re.DOTALL)

    def spans(m: re.Match | None) -> list[tuple[int, int]] | None:
        if m is None:
            return None
        res = []
        for name in sorted(m.groupdict(), key=lambda n: int(n[1:])):
            s, e = _trim(raw_seg, m.start(name), m.end(name))
            if s >= e or not _ALNUM_RE.search(raw_seg[s:e]):
                return None
            res.append((s, e))
        return res

    labels = [v for k, v in parts if k == "ph"]
    for typed in (True, False):
        lazy = spans(build(typed, False).fullmatch(raw_seg))
        greedy = spans(build(typed, True).fullmatch(raw_seg))
        if lazy is not None and lazy == greedy:
            return [(s, e, lab) for (s, e), lab in zip(lazy, labels)]
        if lazy is not None or greedy is not None:
            return None  # matches exist but the split is not unique
    return None


def _overlaps_any(s: int, e: int, items: Iterable[Entity | Region]) -> bool:
    return any(s < x.end and x.start < e for x in items)


# ---------------------------------------------------------------- main entry point
def derive_reference_spans(
    note_id: str,
    raw: str,
    reference: str,
    labels: ReferenceLabels,
    possible_leaks: Iterable[str] = (),
) -> GoldDoc:
    doc = GoldDoc(note_id=note_id)
    a, b = tokenize(raw), tokenize(reference)
    # Punctuation tokens are "junk": they may extend a matching block but never anchor one, so a
    # lone "." or "-" cannot pull a placeholder onto the wrong raw words.
    sm = difflib.SequenceMatcher(_is_punct_token, [t.text for t in a], [t.text for t in b], autojunk=False)
    opcodes = _merge_punct_anchors(sm.get_opcodes(), a, b)

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        ref_toks = b[j1:j2]
        phs = [t for t in ref_toks if PLACEHOLDER_RE.fullmatch(t.text)]
        if i2 == i1:  # material present only in the reference
            doc.stats["placeholder_without_raw" if phs else "reference_only_insertion"] += 1
            continue
        rs, re_ = a[i1].start, a[i2 - 1].end
        if not phs:
            if _ALNUM_RE.search(raw[rs:re_]):
                doc.ignore.append(Region(rs, re_, "unexplained_edit"))
                doc.stats["unexplained_edit"] += 1
            else:
                doc.stats["punctuation_edit"] += 1
            continue
        statuses = [labels.resolve(t.text[1:-1]) for t in phs]
        if any(st == "invalid" for st, _, _ in statuses):
            doc.ignore.append(Region(rs, re_, "invalid_placeholder"))
            doc.stats["invalid_placeholder"] += 1
            continue
        if len(ref_toks) == 1:
            s, e = _trim(raw, rs, re_)
            if (s, e) != (rs, re_):
                doc.stats["trimmed_boundary"] += 1
            pieces = [(s, e, phs[0].text[1:-1])] if s < e else []
        else:
            ref_seg = reference[ref_toks[0].start:ref_toks[-1].end]
            split = _split_group(raw[rs:re_], ref_seg)
            if split is None:
                doc.ignore.append(Region(rs, re_, "ambiguous_group"))
                doc.stats["ambiguous_group"] += 1
                doc.stats["ambiguous_group_placeholders"] += len(phs)
                continue
            doc.stats["split_group"] += 1
            pieces = [(rs + s, rs + e, lab) for s, e, lab in split]
        for s, e, fine in pieces:
            status, norm, canonical = labels.resolve(fine)
            if canonical != fine:
                doc.stats["corrupted_label_normalised"] += 1
            if status == "excluded":
                doc.ignore.append(Region(s, e, f"excluded:{canonical}"))
                doc.stats[f"excluded:{canonical}"] += 1
            else:
                final = labels.normalise_by_content(canonical, norm, raw[s:e])
                if final != norm:
                    doc.stats[f"content_rule:{canonical}->{final}"] += 1
                doc.spans.append(Entity(s, e, final, source="reference", fine_label=canonical))

    # Strings the reference generator itself flagged as possible leaks: uncertain, so ignore them.
    for leak in {x.strip() for x in possible_leaks if isinstance(x, str)}:
        if len(leak) < 3:
            continue
        pat = re.escape(leak)
        if re.match(r"\w", leak):
            pat = r"(?<!\w)" + pat
        if re.search(r"\w$", leak):
            pat = pat + r"(?!\w)"
        for m in re.finditer(pat, raw):
            if not _overlaps_any(m.start(), m.end(), doc.spans) and not _overlaps_any(m.start(), m.end(), doc.ignore):
                doc.ignore.append(Region(m.start(), m.end(), "possible_leak"))
                doc.stats["possible_leak_region"] += 1

    doc.spans.sort(key=lambda e: (e.start, e.end))
    doc.ignore.sort(key=lambda r: (r.start, r.end))
    doc.stats["spans"] = len(doc.spans)
    doc.stats["ignore_regions"] = len(doc.ignore)
    doc.stats["reference_placeholders"] = len(PLACEHOLDER_RE.findall(reference))
    return doc


# ---------------------------------------------------------------- reference coverage adjustments
# Reference-independent full-date patterns (complete day-month-year). Under the project policy a
# full date is always PHI, so an unlabelled one is a reference omission, never a negative.
_FULL_DATE_RES = [
    re.compile(r"(?<![\d/.-])(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})(?![\d-])"),
    re.compile(r"(?<![\d/.-])(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{4})(?![\d/])"),
]
UNANNOTATED_MIN_DATES = 3
UNANNOTATED_CATEGORIES = ["DATE", "TIME"]


def full_dates(raw: str) -> list[tuple[int, int]]:
    out = []
    for rx in _FULL_DATE_RES:
        for m in rx.finditer(raw):
            mo, d, y = int(m.group("m")), int(m.group("d")), int(m.group("y"))
            if 1 <= mo <= 12 and 1 <= d <= 31 and 1800 <= y <= 2299:
                out.append(m.span())
    return sorted(out)


def coverage_adjustments(raw: str, doc: GoldDoc) -> None:
    """Fill doc.unannotated / doc.adjust_ignore (rules R1 and R2, fixed on 2026-10-12 before any
    model was run; see docs/04_reference_and_evaluation_protocol.md).

    R1: >= 3 full dates in the note and the reference labels none -> DATE and TIME unannotated here.
    R2: otherwise, each full date not overlapped by any reference span or ignore region becomes an
        ignore region. Never adds reference spans, so no system can gain true positives from it.
    """
    dates = [(s, e) for s, e in full_dates(raw) if not _overlaps_any(s, e, doc.ignore)]
    covered = [(s, e) for s, e in dates if _overlaps_any(s, e, doc.spans)]
    doc.unannotated, doc.adjust_ignore = [], []
    if len(dates) >= UNANNOTATED_MIN_DATES and not covered:
        doc.unannotated = list(UNANNOTATED_CATEGORIES)
        doc.stats["R1_unannotated_note"] = 1
        return
    for s, e in dates:
        if not _overlaps_any(s, e, doc.spans):
            doc.adjust_ignore.append(Region(s, e, "R2_unlabelled_full_date"))
    doc.stats["R2_unlabelled_full_dates"] = len(doc.adjust_ignore)


# ---------------------------------------------------------------- verification
def validate_gold(raw: str, doc: GoldDoc) -> list[str]:
    """Structural checks. Returns a list of problem codes (empty means valid)."""
    problems: list[str] = []
    n = len(raw)
    for e in doc.spans:
        if not (0 <= e.start < e.end <= n):
            problems.append("span_out_of_bounds")
        elif raw[e.start].isspace() or raw[e.end - 1].isspace():
            problems.append("span_has_edge_whitespace")
        elif not _ALNUM_RE.search(raw[e.start:e.end]):
            problems.append("span_without_alnum")
    for x, y in zip(doc.spans, doc.spans[1:]):
        if y.start < x.end:
            problems.append("overlapping_spans")
    for r in doc.ignore:
        if not (0 <= r.start < r.end <= n):
            problems.append("ignore_out_of_bounds")
    return problems


def reconstruction_mismatch(raw: str, reference: str, doc: GoldDoc) -> int:
    """Re-apply spans as placeholders and count reference tokens left unexplained.

    Ignore regions are blanked on both sides, so the residual count is the reference noise that is
    neither a span nor an ignore region (whitespace is invisible to the tokenizer by design).
    """
    out, pos = [], 0
    marks = sorted([(e.start, e.end, f"[{e.fine_label}]") for e in doc.spans]
                   + [(r.start, r.end, " ") for r in doc.ignore])
    for s, e, rep in marks:
        if s < pos:
            continue
        out.append(raw[pos:s])
        out.append(rep)
        pos = e
    out.append(raw[pos:])
    rebuilt = [t.text for t in tokenize("".join(out))]
    ref = [t.text for t in tokenize(reference)]
    sm = difflib.SequenceMatcher(None, rebuilt, ref, autojunk=False)
    matched = sum(bl.size for bl in sm.get_matching_blocks())
    return len(ref) - matched
