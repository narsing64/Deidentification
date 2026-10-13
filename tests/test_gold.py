"""Reference-span derivation on synthetic raw/reference pairs."""
import pytest

from deid.config import load_labels
from deid.data.gold import ReferenceLabels, derive_reference_spans, reconstruction_mismatch, validate_gold


@pytest.fixture(scope="module")
def labels():
    return ReferenceLabels(load_labels())


def spans(raw, doc):
    return [(raw[e.start:e.end], e.label, e.fine_label) for e in doc.spans]


def test_single_placeholders(labels):
    raw = "Pt Jane Roe seen on 03/04/2101 by Dr. Max Mustermann."
    ref = "Pt [PATIENT_NAME] seen on [DATE] by Dr. [DOCTOR_NAME]."
    doc = derive_reference_spans("n1", raw, ref, labels)
    assert spans(raw, doc) == [("Jane Roe", "PERSON", "PATIENT_NAME"),
                               ("03/04/2101", "DATE", "DATE"),
                               ("Max Mustermann", "PERSON", "DOCTOR_NAME")]
    assert validate_gold(raw, doc) == []
    assert reconstruction_mismatch(raw, ref, doc) == 0


def test_adjacent_placeholders_are_split_by_type(labels):
    raw = "Labs 2101-03-04 06:20AM BLOOD WBC-8.5"
    ref = "Labs [DATE] [TIME] BLOOD WBC-8.5"
    doc = derive_reference_spans("n2", raw, ref, labels)
    assert spans(raw, doc) == [("2101-03-04", "DATE", "DATE"), ("06:20AM", "TIME", "TIME")]


def test_name_dob_hospital_row(labels):
    raw = "Jane Ann Roe 01/02/1950 Lakeside General Hospital INC\nNext"
    ref = "[PATIENT_NAME] [DATE_OF_BIRTH] [HOSPITAL]\nNext"
    doc = derive_reference_spans("n3", raw, ref, labels)
    assert spans(raw, doc) == [("Jane Ann Roe", "PERSON", "PATIENT_NAME"),
                               ("01/02/1950", "DATE", "DATE_OF_BIRTH"),
                               ("Lakeside General Hospital INC", "ORGANIZATION", "HOSPITAL")]


def test_date_range_not_split_on_inner_hyphen(labels):
    raw = "Stay 2101-03-04 - 2101-03-09 noted"
    ref = "Stay [DATE] - [DATE] noted"
    doc = derive_reference_spans("n4", raw, ref, labels)
    assert spans(raw, doc) == [("2101-03-04", "DATE", "DATE"), ("2101-03-09", "DATE", "DATE")]


def test_ambiguous_group_becomes_ignore_region(labels):
    raw = "Lakeside General Jane Roe today"
    ref = "[HOSPITAL] [PATIENT_NAME] today"
    doc = derive_reference_spans("n5", raw, ref, labels)
    assert doc.spans == []
    assert [r.reason for r in doc.ignore] == ["ambiguous_group"]


def test_corrupted_and_invalid_placeholders(labels):
    raw = "Seen: R. Roe and Bob Stone."
    ref = "Seen: [RPATIENT_NAME] and [STONE]."
    doc = derive_reference_spans("n6", raw, ref, labels)
    assert spans(raw, doc) == [("R. Roe", "PERSON", "PATIENT_NAME")]
    assert doc.stats["corrupted_label_normalised"] == 1
    assert [r.reason for r in doc.ignore] == ["invalid_placeholder"]


def test_excluded_category_and_whitespace_noise(labels):
    raw = "Sex: F\nDOB: 1950-01-02"
    ref = "Sex: [SEX]\n\nDOB:   [DATE_OF_BIRTH]"
    doc = derive_reference_spans("n7", raw, ref, labels)
    assert spans(raw, doc) == [("1950-01-02", "DATE", "DATE_OF_BIRTH")]
    assert [r.reason for r in doc.ignore] == ["excluded:SEX"]


def test_unexplained_word_edit_and_possible_leak(labels):
    raw = "Call Bob later. Bob agreed on 2101-01-01."
    ref = "Call later. Bob agreed on [DATE]."
    doc = derive_reference_spans("n8", raw, ref, labels, possible_leaks=["Bob"])
    reasons = sorted(r.reason for r in doc.ignore)
    assert reasons == ["possible_leak", "unexplained_edit"]
    assert spans(raw, doc) == [("2101-01-01", "DATE", "DATE")]


def test_time_only_datetime_is_normalised_to_time(labels):
    raw = "Labs 2101-03-04 06:20AM and admit 2101-03-05 07:00PM"
    ref = "Labs [DATE] [DATETIME] and admit [DATETIME]"
    doc = derive_reference_spans("n10", raw, ref, labels)
    assert spans(raw, doc) == [("2101-03-04", "DATE", "DATE"),
                               ("06:20AM", "TIME", "DATETIME"),
                               ("2101-03-05 07:00PM", "DATE", "DATETIME")]


def test_coverage_rule_r1_unannotated_note(labels):
    from deid.data.gold import coverage_adjustments
    raw = "A 2101-01-01 B 2101-01-02 06:20AM C 03/04/2101 seen by Jane Roe"
    ref = "A 2101-01-01 B 2101-01-02 06:20AM C 03/04/2101 seen by [PATIENT_NAME]"
    doc = derive_reference_spans("r1", raw, ref, labels)
    coverage_adjustments(raw, doc)
    assert doc.unannotated == ["DATE", "TIME"] and doc.adjust_ignore == []


def test_coverage_rule_r2_unlabelled_date_ignored(labels):
    from deid.data.gold import coverage_adjustments
    raw = "A 2101-01-01 B 2101-01-02 C 2101-13-40"
    ref = "A [DATE] B 2101-01-02 C 2101-13-40"   # second date missed by the reference; third is invalid
    doc = derive_reference_spans("r2", raw, ref, labels)
    coverage_adjustments(raw, doc)
    assert doc.unannotated == []
    assert [(raw[r.start:r.end], r.reason) for r in doc.adjust_ignore] == [("2101-01-02", "R2_unlabelled_full_date")]


def test_leading_colon_trimmed(labels):
    raw = "DOB:1950-01-02 ok"
    ref = "DOB [DATE_OF_BIRTH] ok"
    doc = derive_reference_spans("n9", raw, ref, labels)
    assert spans(raw, doc) == [("1950-01-02", "DATE", "DATE_OF_BIRTH")]
