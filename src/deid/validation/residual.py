"""Residual-PHI validation after transformation.

What it can and cannot establish
--------------------------------
The validator re-scans the OUTPUT with the shared regex rules, and checks whether any detected
original value still appears verbatim elsewhere in the output. It shares blind spots with the
detection pipeline: PHI that no detector recognises is invisible to it. "No residual found" is
therefore NOT a guarantee of complete de-identification.

The report separates four things, all as counts by category (never values):
* detected   - entities found by detection (before transformation)
* transformed - entities replaced (transformed + fallback), and retained ones
* potential residual - regex hits in the output outside generated replacements, plus verbatim
  repeats of detected values that survived elsewhere in the text
* review required - fallback transformations and every potential residual

For annotated data, `reference_residual` measures what fraction of each reference span's characters
were left untouched, using the exact original->output mapping of the transformation records.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from deid.detectors.regex_detector import RegexDetector
from deid.policies.transform import TransformResult
from deid.schema import Entity

_ALNUM = re.compile(r"[^\W_]")


@dataclass
class ResidualReport:
    detected: Counter = field(default_factory=Counter)
    transformed: Counter = field(default_factory=Counter)
    retained: Counter = field(default_factory=Counter)
    fallback: Counter = field(default_factory=Counter)
    potential_residual_regex: Counter = field(default_factory=Counter)
    potential_residual_repeat: Counter = field(default_factory=Counter)

    @property
    def review_required(self) -> int:
        return sum(self.fallback.values()) + sum(self.potential_residual_regex.values()) + sum(self.potential_residual_repeat.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": dict(self.detected), "transformed": dict(self.transformed),
            "retained": dict(self.retained), "fallback_review": dict(self.fallback),
            "potential_residual_regex": dict(self.potential_residual_regex),
            "potential_residual_repeated_value": dict(self.potential_residual_repeat),
            "review_required": self.review_required,
            "guarantee": "none - the validator shares blind spots with the detectors",
        }

    def merge(self, other: "ResidualReport") -> None:
        for name in ("detected", "transformed", "retained", "fallback", "potential_residual_regex", "potential_residual_repeat"):
            getattr(self, name).update(getattr(other, name))


class ResidualValidator:
    def __init__(self, regex: RegexDetector | None = None, min_repeat_len: int = 4):
        self.regex = regex or RegexDetector()
        self.min_repeat_len = min_repeat_len

    def validate(self, original: str, entities: Sequence[Entity], result: TransformResult) -> ResidualReport:
        rep = ResidualReport()
        rep.detected.update(e.label for e in entities)
        for r in result.records:
            if r.status == "retained":
                rep.retained[r.label] += 1
            elif r.status == "fallback":
                rep.fallback[r.label] += 1
                rep.transformed[r.label] += 1
            else:
                rep.transformed[r.label] += 1
        generated = result.generated_ranges()

        def inside_generated(s: int, e: int) -> bool:
            return any(s < ge and gs < e for gs, ge in generated)

        for hit in self.regex.detect(result.text):
            if not inside_generated(hit.start, hit.end):
                rep.potential_residual_regex[hit.label] += 1
        # verbatim repeats of detected (non-retained) values still present in the output
        retained_spans = {(r.orig_start, r.orig_end) for r in result.records if r.status == "retained"}
        values = {}
        for e in entities:
            if (e.start, e.end) in retained_spans:
                continue
            v = original[e.start:e.end]
            if len(v) >= self.min_repeat_len and _ALNUM.search(v):
                values.setdefault(v, e.label)
        for v, label in values.items():
            pat = (r"(?<!\w)" if re.match(r"\w", v) else "") + re.escape(v) + (r"(?!\w)" if re.search(r"\w$", v) else "")
            for m in re.finditer(pat, result.text):
                if not inside_generated(m.start(), m.end()):
                    rep.potential_residual_repeat[label] += 1
        return rep


# Leftovers that reveal nothing: punctuation, titles and credentials, and (Exp. 13) address/ID field words
# such as "H.No.", "Door No.", "Flat", "Opp.", "Near" when an annotation convention includes them in the span.
_NON_IDENTIFYING = re.compile(
    r"^(?:[\W_]|\b(?:Dr|Mr|Mrs|Ms|Miss|Prof|Smt|Shri|Sri|MD|DO|RN|NP|PhD|Jr|Sr|"
    r"H|House|Flat|Plot|Door|Shop|No|Opp|Opposite|Near|Behind|Beside|MRN|UHID|MRD|ID)\b)*$", re.IGNORECASE)


def reference_residual(gold: Sequence[Entity], result: TransformResult, text: str | None = None) -> dict[str, Any]:
    """Per reference span: fraction of its characters NOT covered by a non-retained transformation.
    With `text`, partial exposures are split into identifying vs non-identifying (only titles,
    credentials, punctuation or whitespace left), counts only."""
    covered = sorted((r.orig_start, r.orig_end) for r in result.records if r.status != "retained")
    by_cat: dict[str, Counter] = {}
    exposed_chars = total_chars = 0
    for g in gold:
        cov = sum(max(0, min(g.end, e) - max(g.start, s)) for s, e in covered)
        n = g.end - g.start
        frac_exposed = 1 - cov / n if n else 0.0
        c = by_cat.setdefault(g.label, Counter())
        c["spans"] += 1
        if frac_exposed >= 1.0:
            c["fully_exposed"] += 1
        elif frac_exposed > 0:
            c["partially_exposed"] += 1
            if text is not None:
                exposed = "".join(ch for i, ch in enumerate(text[g.start:g.end], g.start)
                                  if not any(s <= i < e for s, e in covered))
                kind = "nonidentifying" if _NON_IDENTIFYING.match(" ".join(exposed.split())) else "identifying"
                c[f"partially_exposed_{kind}"] += 1
        else:
            c["fully_protected"] += 1
        exposed_chars += n - cov
        total_chars += n
    return {"by_category": {k: dict(v) for k, v in sorted(by_cat.items())},
            "spans": len(gold),
            "fully_exposed": sum(c["fully_exposed"] for c in by_cat.values()),
            "partially_exposed": sum(c["partially_exposed"] for c in by_cat.values()),
            "exposed_char_fraction": exposed_chars / total_chars if total_chars else None}
