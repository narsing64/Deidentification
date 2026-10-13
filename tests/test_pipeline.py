"""Regex-only end-to-end pipeline smoke test (no model download needed). Synthetic text only."""
import pytest

from deid.pipeline import DeidPipeline

KEY = b"pipeline-test-key-not-secret-000000"


@pytest.fixture(scope="module")
def pipe():
    return DeidPipeline(model_key=None, key=KEY)


def test_end_to_end_regex_only(pipe):
    text = ("Name: Jane Roe\nAdmission Date: 2101-03-04 06:20AM\nCall (555) 123-4567. MRN: 00123456.\n"
            "Metoprolol 25 mg PO BID; BP 120/80.")
    out = pipe.process("n1", text)
    for secret in ("Jane Roe", "2101-03-04", "06:20AM", "(555) 123-4567", "00123456"):
        assert secret not in out.text
    for keep in ("Metoprolol 25 mg PO BID", "BP 120/80", "Admission Date:", "MRN:"):
        assert keep in out.text
    assert out.residual.review_required == 0
    assert {r.label for r in out.transform.records} == {"PERSON", "DATE", "TIME", "PHONE", "ID"}


def test_empty_note_and_type_check(pipe):
    assert pipe.process("e", "").text == ""
    assert pipe.process("w", "  \n ").text == "  \n "
    with pytest.raises(TypeError):
        pipe.process("x", None)


def test_consistency_across_notes(pipe):
    a = pipe.process("a", "Name: Jane Roe on 2101-03-04", patient_id="P1").text
    b = pipe.process("b", "Name: Jane Roe on 2101-03-04", patient_id="P1").text
    assert a == b
