"""Common internal representation shared by every component.

Entities never store the surface text. The (start, end) offsets into the ORIGINAL note are the
"safe reference" to the span; use `span_text` only where text is genuinely needed (transformation),
never for logging.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable


@dataclass(frozen=True, order=True)
class Entity:
    start: int                      # inclusive character offset into the original text
    end: int                        # exclusive character offset
    label: str                      # normalised PHI category (configs/labels.yaml)
    source: str = "unknown"         # "reference", "regex", a model key, or "fused"
    score: float | None = None      # confidence in [0, 1] when available
    fine_label: str | None = None   # label as emitted by the source, before normalisation

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("Entity offsets must be integers")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Entity") -> bool:
        return self.start < other.end and other.start < self.end

    def with_(self, **changes: Any) -> "Entity":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Entity":
        return cls(
            start=int(d["start"]),
            end=int(d["end"]),
            label=str(d["label"]),
            source=str(d.get("source", "unknown")),
            score=None if d.get("score") is None else float(d["score"]),
            fine_label=d.get("fine_label"),
        )


@dataclass(frozen=True)
class Region:
    """A character range excluded from scoring (uncertain reference, excluded category, ...)."""

    start: int
    end: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Note:
    note_id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


def span_text(text: str, ent: Entity | Region) -> str:
    return text[ent.start:ent.end]


def is_valid_span(ent: Entity | Region, text_len: int) -> bool:
    return 0 <= ent.start < ent.end <= text_len


def sort_entities(ents: Iterable[Entity]) -> list[Entity]:
    return sorted(ents, key=lambda e: (e.start, e.end, e.label, e.source))
