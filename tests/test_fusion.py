"""Span fusion rules (deid.fusion.span_merger). Synthetic text only."""
from deid.fusion.span_merger import FusionConfig, fuse
from deid.schema import Entity

TEXT = "Seen 2101-03-04 06:20AM by Dr. Jane Roe at Lakeside General Hospital, MRN 00123456. Jane Roe ok."


def at(sub, nth=0):
    i = -1
    for _ in range(nth + 1):
        i = TEXT.index(sub, i + 1)
    return i, i + len(sub)


def E(sub, label, source="model", score=0.9, nth=0):
    s, e = at(sub, nth)
    return Entity(s, e, label, source=source, score=score)


def view(ents):
    return [(TEXT[e.start:e.end], e.label) for e in ents]


def test_exact_duplicates_collapse_and_join_sources():
    out, st = fuse(TEXT, [[E("Jane Roe", "PERSON", "regex", 1.0)], [E("Jane Roe", "PERSON", "m", 0.8)]], return_stats=True)
    assert view(out) == [("Jane Roe", "PERSON")]
    assert out[0].source == "m|regex" and out[0].score == 1.0 and st["exact_duplicates"] == 1


def test_nested_and_partial_same_label_become_union():
    out = fuse(TEXT, [[E("Jane", "PERSON", "regex")], [E("Dr. Jane Roe", "PERSON")]])
    assert view(out) == [("Dr. Jane Roe", "PERSON")]
    out = fuse(TEXT, [[E("Lakeside General", "ORGANIZATION")], [E("General Hospital", "ORGANIZATION", "regex")]])
    assert view(out) == [("Lakeside General Hospital", "ORGANIZATION")]


def test_trusted_regex_splits_model_datetime():
    model = [E("2101-03-04 06:20AM", "DATE")]
    regex = [E("2101-03-04", "DATE", "regex", 1.0), E("06:20AM", "TIME", "regex", 1.0)]
    assert view(fuse(TEXT, [regex, model])) == [("2101-03-04", "DATE"), ("06:20AM", "TIME")]


def test_conflicting_labels_regex_anchor_keeps_model_remainder():
    model = [E("MRN 00123456", "ORGANIZATION")]
    regex = [E("00123456", "ID", "regex", 1.0)]
    assert view(fuse(TEXT, [regex, model])) == [("MRN", "ORGANIZATION"), ("00123456", "ID")]


def test_conflict_without_anchor_highest_score_wins_and_coverage_kept():
    a = E("Lakeside General Hospital", "ORGANIZATION", "m", 0.95)
    b = E("Jane Roe at Lakeside", "PERSON", "m2", 0.60)
    out = fuse(TEXT, [[a], [b]])
    assert view(out) == [("Jane Roe at", "PERSON"), ("Lakeside General Hospital", "ORGANIZATION")]


def test_same_span_conflicting_labels_untrusted():
    out = fuse(TEXT, [[E("Jane Roe", "PERSON", "m", 0.7)], [E("Jane Roe", "ORGANIZATION", "m2", 0.9)]])
    assert view(out) == [("Jane Roe", "ORGANIZATION")]


def test_invalid_offsets_and_whitespace_trimmed():
    bad = [Entity(-1, 3, "PERSON"), Entity(5, 5, "DATE"), Entity(10_000, 10_005, "ID"), Entity(4, 16, "DATE", source="m")]
    out, st = fuse(TEXT, [bad], return_stats=True)
    assert view(out) == [("2101-03-04", "DATE")]
    assert st["invalid_offsets"] == 3 and st["trimmed"] == 1


def test_repeated_entities_stay_separate_and_optional_propagation():
    first = E("Jane Roe", "PERSON")
    assert view(fuse(TEXT, [[first]])) == [("Jane Roe", "PERSON")]
    out = fuse(TEXT, [[first]], FusionConfig(propagate_repeats=True))
    assert view(out) == [("Jane Roe", "PERSON"), ("Jane Roe", "PERSON")]
    assert out[1].source == "propagated"


def test_output_never_overlaps():
    ents = [E("2101-03-04 06:20AM by Dr", "PERSON", "m", 0.5), E("06:20AM", "TIME", "regex", 1.0),
            E("Dr. Jane", "ORGANIZATION", "m2", 0.4), E("Jane Roe", "PERSON", "regex", 1.0)]
    out = fuse(TEXT, [ents])
    assert all(a.end <= b.start for a, b in zip(out, out[1:]))
    covered = set().union(*(set(range(e.start, e.end)) for e in out))
    s, e = at("06:20AM")
    assert set(range(s, e)) <= covered


def test_empty_inputs():
    assert fuse("", [[], []]) == [] and fuse(TEXT, []) == []
