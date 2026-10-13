"""Experiment 6 candidate rules and post-processing steps (synthetic text only)."""
from deid.detectors.regex_detector import RegexDetector
from deid.postprocess import (extend_person_names, min_model_score, postprocess_final, strip_person_titles,
                              time_only_date_to_time)
from deid.schema import Entity


def found(det, text):
    return [(text[e.start:e.end], e.label) for e in det.detect(text)]


def at(text, sub, label, source="m", score=0.9):
    i = text.index(sub)
    return Entity(i, i + len(sub), label, source=source, score=score)


def test_optional_rules_are_off_by_default_and_work_when_enabled():
    t = "Ms. Roe is a 1950 year old. Call +15551234567. Home: 12 Maple Grove, Springfield, New York - 62701."
    base = RegexDetector()
    assert ("1950", "DATE") not in found(base, t) and ("+15551234567", "PHONE") not in found(base, t)
    det = RegexDetector(optional_rules=["year_as_age", "year_as_age_old", "phone_intl", "address_zip"])
    got = found(det, t)
    assert ("1950", "DATE") in got and ("+15551234567", "PHONE") in got
    assert ("12 Maple Grove, Springfield, New York - 62701", "LOCATION") in got


def test_year_as_age_does_not_touch_real_ages_or_doses():
    det = RegexDetector(optional_rules=["year_as_age", "year_as_age_old"])
    assert found(det, "She is a 67 year old woman on 2000 mg daily, age 45.") == []


def test_org_min_words_and_generic_filter():
    t = "Seen at Trauma Center and Lakeside General Hospital INC and Medicover Hospitals."
    orgs = lambda d: [x for x, l in found(d, t) if l == "ORGANIZATION"]   # noqa: E731
    assert orgs(RegexDetector(org_generic_filter=False)) == ["Trauma Center", "Lakeside General Hospital INC", "Medicover Hospitals"]
    assert orgs(RegexDetector()) == ["Lakeside General Hospital INC", "Medicover Hospitals"]          # v2 default
    assert orgs(RegexDetector(org_min_words=3, org_generic_filter=False)) == ["Lakeside General Hospital INC"]


def test_min_model_score_never_drops_regex():
    t = "Jane Roe 2101-01-01"
    ents = [at(t, "Jane Roe", "PERSON", "m", 0.4), at(t, "2101-01-01", "DATE", "regex", 1.0)]
    assert [e.label for e in min_model_score(ents, 0.5)] == ["DATE"]
    assert len(min_model_score(ents, None)) == 2


def test_time_only_date_relabelled():
    t = "at 06:20AM on 2101-01-01 06:20AM"
    ents = [at(t, "06:20AM", "DATE"), at(t, "2101-01-01 06:20AM", "DATE")]
    assert [e.label for e in time_only_date_to_time(ents, t)] == ["TIME", "DATE"]


def test_strip_titles():
    t = "seen by Dr. Max Mustermann and Mrs Roe"
    ents = [at(t, "Dr. Max Mustermann", "PERSON"), at(t, "Mrs Roe", "PERSON")]
    assert [t[e.start:e.end] for e in strip_person_titles(ents, t)] == ["Max Mustermann", "Roe"]


def test_extend_names_over_adjacent_name_words_only():
    t = "Attending: Anna Maria Lopez Garcia MD. Seen with John Q Public Discharge today."
    ents = [at(t, "Maria Lopez", "PERSON"), at(t, "John", "PERSON")]
    out = [t[e.start:e.end] for e in extend_person_names(ents, t)]
    assert out == ["Anna Maria Lopez Garcia", "John Q Public"]   # stops at MD / Discharge


def test_extend_names_never_crosses_another_entity_or_line():
    t = "Jane Roe\nSmith Hospital"
    ents = [at(t, "Jane", "PERSON"), at(t, "Smith Hospital", "ORGANIZATION")]
    out = extend_person_names(ents, t)
    assert [t[e.start:e.end] for e in out] == ["Jane Roe", "Smith Hospital"]


def test_postprocess_final_is_identity_when_disabled():
    t = "Dr. Jane Roe 06:20AM"
    ents = [at(t, "Dr. Jane", "PERSON"), at(t, "06:20AM", "DATE")]
    assert postprocess_final(ents, t, {"postprocess": {}}) == sorted(ents, key=lambda e: (e.start, e.end))
