"""Seeded, stratified, group-aware validation/test split of the annotated notes."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable


class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group_notes(note_ids: list[str], keys_by_note: dict[str, set[str]]) -> list[list[str]]:
    """Connected components of notes that share any key (e.g. a hashed patient name)."""
    uf = _UnionFind(note_ids)
    owner: dict[str, str] = {}
    for nid in note_ids:
        for k in keys_by_note.get(nid, ()):
            if k in owner:
                uf.union(nid, owner[k])
            else:
                owner[k] = nid
    comps: dict[str, list[str]] = defaultdict(list)
    for nid in note_ids:
        comps[uf.find(nid)].append(nid)
    return sorted((sorted(c) for c in comps.values()), key=lambda c: c[0])


def stratified_group_split(
    notes: list[dict[str, Any]],
    groups: list[list[str]],
    validation_fraction: float,
    seed: int,
    stratify_by: list[str],
) -> dict[str, list[str]]:
    """Assign whole groups to validation until each stratum reaches its target share."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    info = {n["note_id"]: n for n in notes}
    strata: dict[tuple, list[list[str]]] = defaultdict(list)
    for g in groups:
        strata[tuple(info[g[0]].get(k) for k in stratify_by)].append(g)
    rng = random.Random(seed)
    validation: list[str] = []
    test: list[str] = []
    for key in sorted(strata, key=str):
        gs = strata[key]
        rng.shuffle(gs)
        size = sum(len(g) for g in gs)
        target = round(size * validation_fraction)
        taken = 0
        for g in gs:
            if taken + len(g) <= target or (taken == 0 and target > 0):
                validation.extend(g)
                taken += len(g)
            else:
                test.extend(g)
    return {"validation": sorted(validation), "test": sorted(test)}
