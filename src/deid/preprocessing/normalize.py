"""Detection-only text normalisation with an explicit offset map back to the original text.

The original note is never modified. Detectors may run on `NormalizedText.text`. Their spans are
mapped back with `to_original`, and every final span refers to the ORIGINAL input.

Only typographic variants are folded (curly quotes, dash variants, exotic spaces, bullets).
Digits, units, dosages and other clinically meaningful characters are never touched, so
normalisation cannot change clinical meaning. Most folds are 1:1 and keep offsets identical. The
optional whitespace collapse changes lengths and is handled by the offset map.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# one character -> one character (length preserving)
CHAR_FOLDS: dict[str, str] = {
    "‘": "'", "’": "'", "‚": "'", "′": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", "　": " ",
    "•": "*", "·": "*", "‣": "*", "●": "*",
    "…": ".",
}
_FOLD_TABLE = str.maketrans(CHAR_FOLDS)
_WS_RUN = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class NormalizedText:
    original: str
    text: str
    # to_orig[i] = original offset of normalised char i; to_orig[len(text)] = len(original)
    to_orig: tuple[int, ...]

    def to_original(self, start: int, end: int) -> tuple[int, int]:
        """Map a normalised [start, end) span to the original [start, end)."""
        if not 0 <= start <= end <= len(self.text):
            raise ValueError("span outside normalised text")
        if start == end:
            o = self.to_orig[start]
            return o, o
        return self.to_orig[start], self.to_orig[end - 1] + 1

    @property
    def is_identity_mapping(self) -> bool:
        return len(self.text) == len(self.original)


def normalize(text: str, collapse_spaces: bool = False) -> NormalizedText:
    folded = text.translate(_FOLD_TABLE)
    if not collapse_spaces:
        return NormalizedText(text, folded, tuple(range(len(text) + 1)))
    out_chars: list[str] = []
    to_orig: list[int] = []
    pos = 0
    for m in _WS_RUN.finditer(folded):
        out_chars.extend(folded[pos:m.start()])
        to_orig.extend(range(pos, m.start()))
        out_chars.append(" ")
        to_orig.append(m.start())
        pos = m.end()
    out_chars.extend(folded[pos:])
    to_orig.extend(range(pos, len(folded)))
    to_orig.append(len(text))
    return NormalizedText(text, "".join(out_chars), tuple(to_orig))
