"""Regex detector: structured PHI is found; clinical values are left alone. Synthetic text only."""
import pytest

from deid.detectors.regex_detector import RegexDetector


@pytest.fixture(scope="module")
def det():
    return RegexDetector()


def found(det, text):
    return [(text[e.start:e.end], e.label) for e in det.detect(text)]


@pytest.mark.parametrize("text,expected", [
    ("Admission Date: 2101-03-04", [("2101-03-04", "DATE")]),
    ("seen 03/04/2101 and 4.3.2101", [("03/04/2101", "DATE"), ("4.3.2101", "DATE")]),
    ("on March 4, 2101 and 4th of March 2101", [("March 4, 2101", "DATE"), ("4th of March 2101", "DATE")]),
    ("since Jan 2099", [("Jan 2099", "DATE")]),
    ("follow up on 3/14", [("3/14", "DATE")]),
    ("born in 1950.", [("1950", "DATE")]),
])
def test_dates(det, text, expected):
    assert found(det, text) == expected


def test_date_and_time_are_separate_spans(det):
    t = "2101-03-04 06:20AM BLOOD WBC-8.5 and 2101-03-04T14:05:00"
    assert found(det, t) == [("2101-03-04", "DATE"), ("06:20AM", "TIME"),
                             ("2101-03-04", "DATE"), ("14:05:00", "TIME")]


@pytest.mark.parametrize("text,expected", [
    ("Call (555) 123-4567 or 555-123-4567.", [("(555) 123-4567", "PHONE"), ("555-123-4567", "PHONE")]),
    ("Tel: +44 20 7946 0958", [("+44 20 7946 0958", "PHONE")]),
    ("mail j.doe99@example.org now", [("j.doe99@example.org", "EMAIL")]),
    ("SSN: 123-45-6789", [("123-45-6789", "ID")]),
    ("MRN: 00123456; Unique ID: AB12345C", [("00123456", "ID"), ("AB12345C", "ID")]),
    ("see https://example.org/x?id=1 from 10.0.0.12", [("https://example.org/x?id=1", "OTHER_PHI"), ("10.0.0.12", "OTHER_PHI")]),
])
def test_contacts_and_ids(det, text, expected):
    assert found(det, text) == expected


def test_names_by_context(det):
    t = "Name: Jane Q Roe\nAttending: Dr. Max Mustermann\nSeen by Mr. Ola Nordmann today. Signed: Ana Ruiz, MD"
    assert found(det, t) == [("Jane Q Roe", "PERSON"), ("Max Mustermann", "PERSON"),
                             ("Ola Nordmann", "PERSON"), ("Ana Ruiz", "PERSON")]


def test_name_field_does_not_take_next_column_or_medication(det):
    assert found(det, "Name: Jane Roe          Unit No: 12345") == [("Jane Roe", "PERSON"), ("12345", "ID")]
    assert found(det, "Medication Name: Lisinopril") == []


def test_organizations_and_addresses(det):
    t = "Transferred to Lakeside General Hospital. Lab: Sierra Peak Medical Institute INC. Home: 12 Oak Tree Lane, Springfield, IL 62701"
    assert found(det, t) == [("Lakeside General Hospital", "ORGANIZATION"),
                             ("Sierra Peak Medical Institute INC", "ORGANIZATION"),
                             ("12 Oak Tree Lane, Springfield, IL 62701", "LOCATION")]


def test_age_only_above_89(det):
    assert found(det, "A 67 year old man; his father is 93 years old; aged 91") == [("93", "AGE"), ("91", "AGE")]


@pytest.mark.parametrize("text", [
    "Metoprolol 25 mg PO BID, heparin 5000 units SC q8h, 2.5 mL/kg",
    "BP 120/80, HR 72, RR 16, SpO2 98%, T 98.6 F",
    "WBC-8.5 RBC-4.12 Hgb-12.1 Hct-36.4 MCV-88 Plt-245",
    "Dilute 1:100. INR 1.2. Ratio 2:1. Score 3/5. Grade 1/6 murmur.",
    "Take 1/2 tablet; 10-12 hours apart; ages 18-65 eligible; 2-3 times a day",
    "QTc 450 ms, EF 55-60%, K 4.2 mmol/L, glucose 105 mg/dL",
    "The patient may require 3 tablets.",
])
def test_clinical_values_not_flagged(det, text):
    assert found(det, text) == []


def test_unicode_and_offsets_refer_to_original(det):
    t = "Seen on 03/04/2101 – call “(555) 123-4567”"
    out = det.detect(t)
    assert [(t[e.start:e.end], e.label) for e in out] == [("03/04/2101", "DATE"), ("(555) 123-4567", "PHONE")]


def test_empty_and_whitespace_input(det):
    assert det.detect("") == [] and det.detect(" \n\t ") == []
