"""Configurable, category-specific transformations (configs/deid_policy.yaml).

Key management
--------------
Pseudonyms and date shifts derive from HMAC-SHA256 with a secret key read from an environment
variable (default DEID_SECRET_KEY). Consequences:
* Consistency: the same normalised value gives the same pseudonym in every note and every run
  that uses the same key. The same patient/scope gets the same date offset, so intervals between
  that patient's dates are preserved across notes.
* Safeguards: keep the key outside the repository (a secrets manager or an OS keystore), restrict
  access, never log it, and rotate it when linkage across releases is not desired. Rotating the key
  breaks linkage on purpose. Anyone holding the key can test guesses (e.g. hash a candidate name),
  so the key is as sensitive as the data.
* Without a key (`allow_ephemeral`), a random per-run key is used. Output is then consistent
  within the run only, and the run report says so.
No original value is ever logged. Transformation records hold offsets, categories and actions only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Sequence

from deid.schema import Entity

ACTIONS = {"redact", "mask", "placeholder", "pseudonymize", "date_shift", "retain"}


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class TransformRecord:
    orig_start: int
    orig_end: int
    out_start: int
    out_end: int
    label: str
    action: str
    status: str                    # transformed | retained | fallback

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TransformResult:
    text: str
    records: list[TransformRecord] = field(default_factory=list)
    key_is_ephemeral: bool = False

    def generated_ranges(self) -> list[tuple[int, int]]:
        return [(r.out_start, r.out_end) for r in self.records if r.status != "retained"]


# ---------------------------------------------------------------- dates
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"]
_MON_RE = r"(?P<mon>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_TIME_P = r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?[Mm]\.?)?|\d{1,2}\s?[AaPp]\.?[Mm]\.?)"
_TIME_TAIL = re.compile(rf"^(?P<sep>\s*,?\s*|T){_TIME_P}$")      # whole remainder is a time
_TIME_END = re.compile(rf"(?:\s*,?\s*|T){_TIME_P}$")               # time at the end of a string

_DATE_FORMS = [
    ("iso", re.compile(r"^(?P<y>\d{4})(?P<s>[-/.])(?P<m>\d{1,2})(?P=s)(?P<d>\d{1,2})")),
    ("mdy", re.compile(r"^(?P<m>\d{1,2})(?P<s>[-/.])(?P<d>\d{1,2})(?P=s)(?P<y>\d{4}|\d{2})")),
    ("mon_d_y", re.compile(rf"^{_MON_RE}(?P<dot>\.?)(?P<s1>\s+)(?P<d>\d{{1,2}})(?P<ord>st|nd|rd|th)?(?P<s2>,?\s+)(?P<y>\d{{4}})", re.I)),
    # 12 March 2024, 4th of March 2024, and the Indian forms 12-Mar-2024, 12-Mar-24, 05/Jan/2023
    ("d_mon_y", re.compile(rf"^(?P<d>\d{{1,2}})(?P<ord>st|nd|rd|th)?(?P<s1>[-/.]|\s+(?:of\s+)?){_MON_RE}(?P<dot>\.?)(?P<s2>[-/.]|,?\s+)(?P<y>\d{{4}}|\d{{2}})(?!\d)", re.I)),
    ("mon_y", re.compile(rf"^{_MON_RE}(?P<dot>\.?)(?P<s1>,?\s+(?:of\s+)?)(?P<y>\d{{4}})", re.I)),
    ("mon_y_apos", re.compile(rf"^{_MON_RE}(?P<dot>\.?)(?P<s1>\s?['’]\s?)(?P<y>\d{{2}})(?!\d)", re.I)),   # Mar'24
    ("mon_d", re.compile(rf"^{_MON_RE}(?P<dot>\.?)(?P<s1>\s+)(?P<d>\d{{1,2}})(?P<ord>st|nd|rd|th)?", re.I)),
    ("md", re.compile(r"^(?P<m>\d{1,2})/(?P<d>\d{1,2})")),
    ("y", re.compile(r"^(?P<y>(?:18|19|20|21)\d{2})")),
]


def _mon_index(name: str) -> int:
    n = name.lower()[:3]
    return [m[:3] for m in _MONTHS].index(n) + 1


def _fmt_month(orig: str, month: int) -> str:
    full = _MONTHS[month - 1]
    name = full if orig.lower() in _MONTHS else full[:3]
    return name.upper() if orig.isupper() else name.capitalize()


def _year(g: str) -> int:
    y = int(g)
    return y + (2000 if y < 50 else 1900) if len(g) == 2 else y


def shift_date_string(s: str, days: int, day_first: bool | None = None) -> str | None:
    """Shift a date written in one of the supported formats, keeping the format. None if unparseable.
    A trailing time component is kept verbatim (callers treat it separately).
    day_first: how to read an ambiguous numeric date such as 05/03/2024 (True = 5 March, Indian/UK;
    False/None = May 3, US). An unambiguous one (25/12/2024) is always read correctly."""
    for kind, rx in _DATE_FORMS:
        m = rx.match(s)
        if not m:
            continue
        rest = s[m.end():]
        if rest and not _TIME_TAIL.match(rest):
            continue
        g = m.groupdict()
        try:
            if kind in ("iso", "mdy"):
                y = int(g["y"])
                if len(g["y"]) == 2:
                    y += 2000 if y < 50 else 1900
                mo, d = int(g["m"]), int(g["d"])
                if kind == "mdy" and (mo > 12 >= d or (day_first and d <= 12)):   # day-first input
                    mo, d = d, mo
                    new = dt.date(y, mo, d) + dt.timedelta(days=days)
                    out = f"{new.day:0{len(g['m'])}d}{g['s']}{new.month:0{len(g['d'])}d}{g['s']}"
                else:
                    new = dt.date(y, mo, d) + dt.timedelta(days=days)
                    if kind == "iso":
                        out = f"{new.year:04d}{g['s']}{new.month:0{len(g['m'])}d}{g['s']}{new.day:0{len(g['d'])}d}"
                        return out + rest
                    out = f"{new.month:0{len(g['m'])}d}{g['s']}{new.day:0{len(g['d'])}d}{g['s']}"
                out += f"{new.year % 100:02d}" if len(g["y"]) == 2 else f"{new.year:04d}"
                return out + rest
            if kind in ("mon_d_y", "d_mon_y"):
                new = dt.date(_year(g["y"]), _mon_index(g["mon"]), int(g["d"])) + dt.timedelta(days=days)
                mon = _fmt_month(g["mon"], new.month) + g["dot"]
                ordn = _ordinal(new.day) if g.get("ord") else ""
                yy = f"{new.year % 100:02d}" if len(g["y"]) == 2 else f"{new.year}"
                if kind == "mon_d_y":
                    return f"{mon}{g['s1']}{new.day}{ordn}{g['s2']}{yy}" + rest
                dd = f"{new.day:0{len(g['d'])}d}" if not ordn else f"{new.day}"
                return f"{dd}{ordn}{g['s1']}{mon}{g['s2']}{yy}" + rest
            if kind == "mon_y":
                new = dt.date(int(g["y"]), _mon_index(g["mon"]), 15) + dt.timedelta(days=days)
                return f"{_fmt_month(g['mon'], new.month)}{g['dot']}{g['s1']}{new.year}" + rest
            if kind == "mon_y_apos":
                new = dt.date(_year(g["y"]), _mon_index(g["mon"]), 15) + dt.timedelta(days=days)
                return f"{_fmt_month(g['mon'], new.month)}{g['dot']}{g['s1']}{new.year % 100:02d}" + rest
            if kind == "mon_d":
                new = dt.date(2001, _mon_index(g["mon"]), int(g["d"])) + dt.timedelta(days=days)
                ordn = _ordinal(new.day) if g.get("ord") else ""
                return f"{_fmt_month(g['mon'], new.month)}{g['dot']}{g['s1']}{new.day}{ordn}" + rest
            if kind == "md":
                new = dt.date(2001, int(g["m"]), int(g["d"])) + dt.timedelta(days=days)
                return f"{new.month}/{new.day}" + rest
            if kind == "y":
                new = dt.date(int(g["y"]), 7, 1) + dt.timedelta(days=days)
                return f"{new.year}" + rest
        except ValueError:
            return None
    return None


def _ordinal(n: int) -> str:
    return "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# ---------------------------------------------------------------- engine
def _normalise_value(label: str, value: str) -> str:
    v = value.casefold()
    if label == "PERSON":
        v = re.sub(r"\b(dr|mr|mrs|ms|miss|prof|smt|shri|sri|shrimati|kumari|kum|late|mx|er|capt|col|maj|brig)\b\.?", " ", v)
        v = re.sub(r"[^\w\s'-]", " ", v)
    if label in ("ID", "PHONE"):
        v = re.sub(r"[^0-9a-z]", "", v)
    return " ".join(v.split())


class PolicyEngine:
    def __init__(self, policy: dict[str, Any], key: bytes | None = None):
        self.policy = policy
        self.categories: dict[str, dict] = {k: dict(v) for k, v in (policy.get("categories") or {}).items()}
        self.default = dict(policy.get("default") or {"action": "redact"})
        self.retain_allowed = set(policy.get("retain_allowed") or [])
        ds = policy.get("date_shift") or {}
        self.shift_scope = ds.get("scope", "patient")
        self.min_days, self.max_days = int(ds.get("min_days", -365)), int(ds.get("max_days", -1))
        self.day_first_cfg = str(ds.get("day_first", "auto")).lower()   # auto | true | false
        if self.min_days > self.max_days:
            raise PolicyError("date_shift.min_days must be <= max_days")
        if self.min_days <= 0 <= self.max_days:
            raise PolicyError("date_shift range must exclude 0 so dates always change")
        for cat, rule in list(self.categories.items()) + [("default", self.default)]:
            if rule.get("action") not in ACTIONS:
                raise PolicyError(f"{cat}: unknown action {rule.get('action')!r}")
            if rule["action"] == "retain" and cat not in self.retain_allowed:
                raise PolicyError(f"{cat}: 'retain' is not permitted unless listed in retain_allowed")
        self.key_is_ephemeral = False
        if key is None:
            kc = policy.get("key") or {}
            raw = os.environ.get(kc.get("env_var", "DEID_SECRET_KEY"))
            if raw:
                key = bytes.fromhex(raw) if re.fullmatch(r"(?:[0-9a-fA-F]{2}){16,}", raw) else raw.encode()
            elif kc.get("allow_ephemeral", True):
                key, self.key_is_ephemeral = secrets.token_bytes(32), True
            else:
                raise PolicyError("no de-identification key available")
        self._key = key
        nl = policy.get("name_linking") or {}
        self.link_names = bool(nl.get("enabled", True))
        # In-memory only (never written anywhere): canonical full-name keys seen per linking scope, so a
        # later "Mr. Roe" in another note of the same patient maps to the earlier "Jane Q. Roe".
        self._name_registry: dict[str, set[str]] = {}

    def rule_for(self, label: str) -> dict:
        return self.categories.get(label, self.default)

    def _digest(self, msg: str) -> str:
        return hmac.new(self._key, msg.encode("utf-8"), hashlib.sha256).hexdigest()

    def shift_days(self, scope_id: str) -> int:
        span = self.max_days - self.min_days + 1
        return self.min_days + int(self._digest(f"date-shift|{scope_id}"), 16) % span

    def pseudonym(self, label: str, value: str, prefix: str | None = None) -> str:
        return f"{prefix or label}_{self._digest(f'pseudo|{label}|{_normalise_value(label, value)}')[:8]}"

    @staticmethod
    def _name_tokens(value: str) -> list[str]:
        """Normalised name words without titles and single-letter initials ("Dr. Jane Q. Roe" -> jane, roe)."""
        return [w for w in _normalise_value("PERSON", value).replace("'", " ").replace("-", " ").split() if len(w) > 1]

    def person_link_keys(self, text: str, ents: Sequence[Entity], scope_id: str) -> dict[int, str]:
        """Canonical key per PERSON entity so that variants of one name share a pseudonym:
        * a multi-word name -> its first and last word, order-independent ("Roe, Jane" == "Jane Q. Roe");
        * a single word ("Roe", "Mr. Roe", "Jane") -> the full name in this note / patient scope that
          contains that word, if exactly one does; otherwise the word itself (no guessing between two
          people who share a surname)."""
        toks = {i: self._name_tokens(text[e.start:e.end]) for i, e in enumerate(ents) if e.label == "PERSON"}
        full = {i: " ".join(sorted({w[0], w[-1]})) for i, w in toks.items() if len(w) >= 2}
        registry = self._name_registry.setdefault(scope_id, set())
        registry.update(full.values())
        keys: dict[int, str] = {}
        for i, w in toks.items():
            if len(w) >= 2:
                keys[i] = full[i]
            elif len(w) == 1:
                cands = {k for k in registry if w[0] in k.split()}
                keys[i] = cands.pop() if len(cands) == 1 else w[0]
        return keys

    def note_day_first(self, values: Sequence[str]) -> bool | None:
        """`auto`: a note is day-first if one of its numeric dates can only be d/m (first part > 12)
        and none can only be m/d; month-first in the opposite case; undecided (None -> US reading)
        otherwise. All dates of one note are then read the same way, keeping their intervals intact."""
        if self.day_first_cfg in ("true", "yes", "1"):
            return True
        if self.day_first_cfg in ("false", "no", "0"):
            return False
        dmy = mdy = False
        for v in values:
            m = re.match(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](?:\d{4}|\d{2})", v)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                dmy |= a > 12 >= b
                mdy |= b > 12 >= a
        return True if dmy and not mdy else False if mdy and not dmy else None

    def _apply(self, label: str, value: str, rule: dict, scope_id: str, link_key: str | None = None,
               day_first: bool | None = None) -> tuple[str, str]:
        action = rule["action"]
        if action == "retain":
            return value, "retained"
        if action == "redact":
            return rule.get("replacement", "[REDACTED]"), "transformed"
        if action == "placeholder":
            return rule.get("replacement", f"[{label}]"), "transformed"
        if action == "mask":
            ch = rule.get("mask_char", "*")
            keep_first, keep_last = int(rule.get("keep_first", 0)), int(rule.get("keep_last", 0))
            idx = [i for i, c in enumerate(value) if c.isalnum()]
            visible = set(idx[:keep_first]) | (set(idx[-keep_last:]) if keep_last else set())
            return "".join(c if (not c.isalnum() or i in visible) else ch for i, c in enumerate(value)), "transformed"
        if action == "pseudonymize":
            return self.pseudonym(label, link_key if link_key else value, rule.get("prefix")), "transformed"
        if action == "date_shift":
            shifted = shift_date_string(value, self.shift_days(scope_id), day_first)
            if shifted is not None:
                tail = _TIME_END.search(shifted) if ":" in shifted or re.search(r"[AaPp]\.?[Mm]\.?$", shifted) else None
                if tail and tail.group("time") and label == "DATE" and "TIME" in self.categories:
                    t_new, _ = self._apply("TIME", tail.group("time"), self.rule_for("TIME"), scope_id)
                    shifted = shifted[:tail.start("time")] + t_new
                return shifted, "transformed"
            fb = {"action": rule.get("fallback", "placeholder")}
            out, _ = self._apply(label, value, fb, scope_id)
            return out, "fallback"
        raise PolicyError(f"unknown action {action}")

    def transform(self, text: str, entities: Sequence[Entity], note_id: str, patient_id: str | None = None) -> TransformResult:
        ents = sorted(entities, key=lambda e: (e.start, e.end))
        for a, b in zip(ents, ents[1:]):
            if a.end > b.start:
                raise ValueError("entities overlap; fuse them before transforming")
        if self.shift_scope == "patient" and patient_id:
            scope_id = f"patient:{patient_id}"
        elif self.shift_scope == "global":
            scope_id = "global"
        else:
            scope_id = f"note:{note_id}"
        for e in ents:
            if not (0 <= e.start < e.end <= len(text)):
                raise ValueError("entity offsets outside text")
        link_scope = f"patient:{patient_id}" if patient_id else f"note:{note_id}"
        link_keys = self.person_link_keys(text, ents, link_scope) if self.link_names else {}
        day_first = self.note_day_first([text[e.start:e.end] for e in ents if e.label == "DATE"])
        out: list[str] = []
        records: list[TransformRecord] = []
        pos = olen = 0
        for i, e in enumerate(ents):
            out.append(text[pos:e.start])
            olen += e.start - pos
            rule = self.rule_for(e.label)
            new, status = self._apply(e.label, text[e.start:e.end], rule, scope_id, link_keys.get(i), day_first)
            out.append(new)
            records.append(TransformRecord(e.start, e.end, olen, olen + len(new), e.label, rule["action"], status))
            olen += len(new)
            pos = e.end
        out.append(text[pos:])
        return TransformResult("".join(out), records, self.key_is_ephemeral)
