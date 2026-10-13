"""Dataset loading and schema validation with synthetic files."""
import json

import pytest

from deid.data.loader import load_annotation_meta, load_reference_pairs, load_unlabelled_notes
from deid.data.splits import group_notes, stratified_group_split


def test_reference_pairs_validation(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text(
        ',id,raw,de-identified\n'
        '0,A,"Pt seen 2101-01-01.","Pt seen [DATE]."\n'
        '1,B,"",""\n'
        '2,A,"dup","dup"\n'
        '3,C,"Line one\nline two é","Line one\nline two é"\n', encoding="utf-8")
    recs, rep = load_reference_pairs(p)
    assert [r["note_id"] for r in recs] == ["A", "C"]
    assert rep.issues["raw:empty"] == 1 and rep.issues["duplicate_id"] == 1
    assert recs[1]["raw"].endswith("é")


def test_missing_columns_raise(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("id,text\n1,x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_reference_pairs(p)


def test_annotation_meta_handles_malformed_lines(tmp_path):
    p = tmp_path / "a.jsonl"
    lines = [
        json.dumps({"id": "A", "source": "pdf", "source_id": "PDF_Deid_Deidentification_Hard_3", "possible_leaks": ["x"]}),
        "{not json",
        json.dumps({"source": "csv"}),
        "",
        json.dumps({"id": "B", "source": "csv", "source_id": "7", "possible_leaks": None}),
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    meta, rep = load_annotation_meta(p)
    assert meta["A"]["tier"] == "hard" and meta["B"]["tier"] == "csv"
    assert meta["B"]["possible_leaks"] == []
    assert rep.issues["invalid_json"] == 1 and rep.issues["missing_id"] == 1


def test_unlabelled_notes(tmp_path):
    p = tmp_path / "train.tsv"
    p.write_bytes('\ttext\n0\t"multi\nline"\n1\t   \n'.encode("utf-8"))  # keep LF on Windows
    notes, rep = load_unlabelled_notes(p)
    assert len(notes) == 1 and notes[0].text == "multi\nline"
    assert rep.issues["text:empty"] == 1


def test_empty_file(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text(",id,raw,de-identified\n", encoding="utf-8")
    recs, rep = load_reference_pairs(p)
    assert recs == [] and rep.n_rows == 0


def test_group_aware_split_keeps_groups_together():
    notes = [{"note_id": f"n{i}", "source": "csv", "tier": "csv"} for i in range(10)]
    groups = group_notes([n["note_id"] for n in notes], {"n1": {"h"}, "n7": {"h"}, "n3": {"k"}})
    assert ["n1", "n7"] in groups
    sp = stratified_group_split(notes, groups, 0.4, seed=3, stratify_by=["source", "tier"])
    assert len(sp["validation"]) + len(sp["test"]) == 10
    assert ("n1" in sp["validation"]) == ("n7" in sp["validation"])
    assert sp == stratified_group_split(notes, groups, 0.4, seed=3, stratify_by=["source", "tier"])
