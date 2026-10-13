"""Transformation policies, pseudonym/date-shift consistency, residual validation. Synthetic only."""
import datetime as dt

import pytest

from deid.config import load_yaml
from deid.policies.transform import PolicyEngine, PolicyError, shift_date_string
from deid.schema import Entity
from deid.validation.residual import ResidualValidator, reference_residual

KEY = b"unit-test-key-not-secret-0123456789"


def engine(**cats):
    pol = load_yaml("configs/deid_policy.yaml")
    pol["categories"].update(cats)
    return PolicyEngine(pol, key=KEY)


def ent(text, sub, label, nth=0):
    i = -1
    for _ in range(nth + 1):
        i = text.index(sub, i + 1)
    return Entity(i, i + len(sub), label)


def test_each_action():
    text = "Jane Roe 555-123-4567 a@b.org Lakeside Hospital MRN 00123456 91"
    e = engine(PERSON={"action": "placeholder"}, PHONE={"action": "mask", "keep_last": 4},
               EMAIL={"action": "redact"}, ORGANIZATION={"action": "redact", "replacement": "[ORG]"},
               ID={"action": "pseudonymize", "prefix": "MRN"}, AGE={"action": "redact", "replacement": "[AGE 90+]"})
    ents = [ent(text, "Jane Roe", "PERSON"), ent(text, "555-123-4567", "PHONE"), ent(text, "a@b.org", "EMAIL"),
            ent(text, "Lakeside Hospital", "ORGANIZATION"), ent(text, "00123456", "ID"), ent(text, "91", "AGE")]
    out = e.transform(text, ents, "n1")
    assert out.text.startswith("[PERSON] ***-***-4567 [REDACTED] [ORG] MRN MRN_")
    assert out.text.endswith(" [AGE 90+]")
    assert [r.action for r in out.records] == ["placeholder", "mask", "redact", "redact", "pseudonymize", "redact"]
    for r in out.records:  # output offsets point at the replacement text
        assert out.text[r.out_start:r.out_end] != text[r.orig_start:r.orig_end]


def test_retain_requires_explicit_permission():
    pol = load_yaml("configs/deid_policy.yaml")
    pol["categories"]["AGE"] = {"action": "retain"}
    with pytest.raises(PolicyError):
        PolicyEngine(pol, key=KEY)
    pol["retain_allowed"] = ["AGE"]
    e = PolicyEngine(pol, key=KEY)
    out = e.transform("aged 95", [Entity(5, 7, "AGE")], "n")
    assert out.text == "aged 95" and out.records[0].status == "retained"


def test_unknown_action_and_bad_shift_range_rejected():
    pol = load_yaml("configs/deid_policy.yaml")
    pol["categories"]["PERSON"] = {"action": "shred"}
    with pytest.raises(PolicyError):
        PolicyEngine(pol, key=KEY)
    pol = load_yaml("configs/deid_policy.yaml")
    pol["date_shift"] = {"min_days": -5, "max_days": 5}
    with pytest.raises(PolicyError):
        PolicyEngine(pol, key=KEY)


def test_pseudonyms_consistent_across_notes_and_case_and_titles():
    e = engine()
    a = e.transform("Dr. Jane Roe", [Entity(4, 12, "PERSON")], "note-1").text
    b = e.transform("JANE  ROE called", [Entity(0, 9, "PERSON")], "note-2").text
    assert a[4:] == b[:len(a) - 4]
    other = engine().transform("John Roe", [Entity(0, 8, "PERSON")], "x").text
    assert other != a[4:]
    # a different key breaks linkage on purpose
    pol = load_yaml("configs/deid_policy.yaml")
    c = PolicyEngine(pol, key=b"another-key-another-key-another!").transform("Jane Roe", [Entity(0, 8, "PERSON")], "n").text
    assert c != a[4:]


@pytest.mark.parametrize("s,expected", [
    ("2101-03-04", "2101-02-25"), ("03/04/2101", "02/25/2101"), ("3/4/01", "2/25/01"),
    ("March 4, 2101", "February 25, 2101"), ("4th of March 2101", "25th of February 2101"),
    ("Mar. 4 2101", "Feb. 25 2101"), ("March 2101", "March 2101"), ("2101", "2101"),
    ("25/12/2101", "18/12/2101"), ("2101-03-04 06:20AM", "2101-02-25 06:20AM"), ("not a date", None),
    ("2101-02-30", None),
])
def test_shift_date_string_keeps_format(s, expected):
    assert shift_date_string(s, -7) == expected


def test_date_shift_preserves_intervals_within_scope_and_never_zero():
    e = engine()
    text = "Admit 2101-03-04, discharge 2101-03-10, follow-up 04/01/2101."
    ents = [ent(text, "2101-03-04", "DATE"), ent(text, "2101-03-10", "DATE"), ent(text, "04/01/2101", "DATE")]
    out = e.transform(text, ents, "note-A", patient_id="P1")
    got = [out.text[r.out_start:r.out_end] for r in out.records]
    d = [dt.date.fromisoformat(got[0]), dt.date.fromisoformat(got[1]), dt.datetime.strptime(got[2], "%m/%d/%Y").date()]
    assert (d[1] - d[0]).days == 6 and (d[2] - d[0]).days == 28
    shift = (d[0] - dt.date(2101, 3, 4)).days
    assert -730 <= shift <= -30
    # same patient, another note -> same shift; another patient -> independent shift
    again = e.transform("on 2101-03-04", [Entity(3, 13, "DATE")], "note-B", patient_id="P1").text
    assert again == f"on {got[0]}"
    assert e.shift_days("patient:P1") == e.shift_days("patient:P1")


def test_date_time_component_follows_time_policy_and_fallback_flagged():
    e = engine()
    out = e.transform("x 2101-03-04 06:20AM", [Entity(2, 20, "DATE")], "n")
    assert out.text.endswith(" [TIME]") and "06:20" not in out.text
    out = e.transform("on the 3rd day", [Entity(3, 10, "DATE")], "n")
    assert out.text == "on [DATE] day" and out.records[0].status == "fallback"


def test_overlapping_entities_rejected_and_empty_input():
    e = engine()
    with pytest.raises(ValueError):
        e.transform("Jane Roe", [Entity(0, 4, "PERSON"), Entity(2, 8, "PERSON")], "n")
    assert e.transform("", [], "n").text == ""


def test_residual_validator_counts_without_values():
    text = "Jane Roe seen 2101-03-04. Call 555-123-4567. Jane Roe again on 03/05/2101."
    ents = [ent(text, "Jane Roe", "PERSON"), ent(text, "2101-03-04", "DATE")]  # second name + date missed
    e = engine()
    res = e.transform(text, ents, "n")
    rep = ResidualValidator().validate(text, ents, res)
    d = rep.to_dict()
    assert d["detected"] == {"PERSON": 1, "DATE": 1} and d["transformed"] == {"PERSON": 1, "DATE": 1}
    assert d["potential_residual_regex"] == {"PHONE": 1, "DATE": 1}       # shifted date NOT re-flagged
    assert d["potential_residual_repeated_value"] == {"PERSON": 1}
    assert rep.review_required == 3
    assert "Jane" not in str(d) and "555" not in str(d)


def test_reference_residual_measures_exposure():
    text = "Jane Roe seen 2101-03-04 by Max Mustermann"
    gold = [ent(text, "Jane Roe", "PERSON"), ent(text, "2101-03-04", "DATE"), ent(text, "Max Mustermann", "PERSON")]
    pred = [ent(text, "Jane", "PERSON"), ent(text, "2101-03-04", "DATE")]
    res = engine().transform(text, pred, "n")
    rr = reference_residual(gold, res)
    assert rr["fully_exposed"] == 1 and rr["partially_exposed"] == 1
    assert rr["by_category"]["DATE"] == {"spans": 1, "fully_protected": 1}


def test_reference_residual_separates_title_only_exposure():
    text = "seen by Dr. Max Mustermann and Jane Roe"
    gold = [ent(text, "Dr. Max Mustermann", "PERSON"), ent(text, "Jane Roe", "PERSON")]
    pred = [ent(text, "Max Mustermann", "PERSON"), ent(text, "Roe", "PERSON")]
    res = engine().transform(text, pred, "n")
    rr = reference_residual(gold, res, text)
    c = rr["by_category"]["PERSON"]
    assert c["partially_exposed_nonidentifying"] == 1 and c["partially_exposed_identifying"] == 1
