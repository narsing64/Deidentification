"""Helpers that let reports describe PHI without revealing it."""
from __future__ import annotations

import hashlib
import hmac
import re

_SHAPE_RE = re.compile(r"\d+|[A-Z][a-z]+|[A-Z]+|[a-z]+|\s+|.", re.DOTALL)


def safe_shape(text: str, max_len: int = 40) -> str:
    """Character-class shape: digits keep their count ("9999-99-99"), words collapse ("Xx", "XX", "x").

    Words are collapsed so name lengths are not revealed; digit counts are kept because they
    distinguish date/ID formats without revealing values.
    """
    out = []
    for m in _SHAPE_RE.finditer(text):
        t = m.group()
        if t.isdigit():
            out.append("9" * min(len(t), 6))
        elif t[0].isupper() and len(t) > 1 and t[1:].islower():
            out.append("Xx")
        elif t.isupper() and t.isalpha():
            out.append("XX" if len(t) > 1 else "X")
        elif t.isalpha():
            out.append("x")
        elif t.isspace():
            out.append(" ")
        else:
            out.append(t)
    s = "".join(out)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def keyed_hash(value: str, key: bytes, length: int = 16) -> str:
    """HMAC-SHA256 digest (hex prefix). Used for linkage without storing values."""
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()[:length]


def unkeyed_hash(value: str, length: int = 16) -> str:
    """Plain SHA-256 prefix. Only for in-memory grouping; never persist hashes of real PHI."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
