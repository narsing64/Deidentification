"""Synthetic Indian dataset generator: spans must be exact, valid and complete (no real data involved)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_indian_synthetic as gen  # noqa: E402


def test_generated_spans_are_exact_and_qa_clean():
    words = gen.pool_words()
    notes = gen.build("unit", 60, seed=7)
    assert len(notes) == 60
    for n in notes:
        assert gen.qa_note(n, words) == [], n["note_id"]
        prev_end = -1
        for s in n["spans"]:
            assert 0 <= s["start"] < s["end"] <= len(n["text"])
            assert s["start"] >= prev_end                       # sorted, non-overlapping
            prev_end = s["end"]
            assert s["label"] in {"PERSON", "DATE", "TIME", "AGE", "ORGANIZATION", "LOCATION", "PHONE", "EMAIL", "ID", "OTHER_PHI"}


def test_build_is_deterministic():
    a = gen.build("x", 5, seed=11)
    b = gen.build("x", 5, seed=11)
    assert a == b


def test_aadhaar_values_pass_verhoeff_and_ages_follow_guideline():
    notes = gen.build("unit", 80, seed=3)
    for n in notes:
        t = n["text"]
        for s in n["spans"]:
            v = t[s["start"]:s["end"]]
            if s["label"] == "AGE":
                assert int(v) >= 90
        for m in re.finditer(r"Aadhaar(?: No\.)?\s*:?\s*\(?(\d{4}[ -]?\d{4}[ -]?\d{4})", t):
            assert gen.verhoeff_ok(re.sub(r"\D", "", m.group(1)))


def test_titles_and_degrees_are_outside_person_spans():
    notes = gen.build("unit", 60, seed=5)
    for n in notes:
        for s in n["spans"]:
            if s["label"] == "PERSON":
                v = n["text"][s["start"]:s["end"]]
                assert not re.match(r"(?i)(dr|mr|mrs|smt|shri|prof)\b\.?\s", v), v
                assert not re.search(r",\s*(MBBS|MD|MS)\b", v), v
