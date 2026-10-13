"""Model adapter: windowing, sub-word handling, tag decoding, label mapping. No downloads needed:
a tiny local WordPiece vocabulary and a fake model whose logits depend only on the token id."""
import numpy as np
import pytest
import torch

from deid.config import load_labels
from deid.detectors.hf_detector import (HFTokenClassifier, ModelSpec, decode_tags, merge_adjacent,
                                        parse_label, plan_windows)
from deid.schema import Entity


# ---------------------------------------------------------------- pure helpers
@pytest.mark.parametrize("n,window,context", [(1, 10, 2), (10, 10, 2), (11, 10, 2), (57, 10, 2), (1000, 126, 32), (300, 382, 64)])
def test_windows_tile_every_token_once_with_context(n, window, context):
    ws = plan_windows(n, window, context)
    covered = [t for _, _, cs, ce in ws for t in range(cs, ce)]
    assert covered == list(range(n))                                  # no gaps, no overlaps
    for w0, w1, cs, ce in ws:
        assert w1 - w0 <= window and w0 <= cs < ce <= w1
        if n > window:
            assert cs - w0 >= min(context, cs) and w1 - ce >= min(context, n - ce)


def test_windows_validate_arguments():
    with pytest.raises(ValueError):
        plan_windows(10, 4, 2)
    assert plan_windows(0, 10, 2) == []


def test_parse_label_schemes():
    assert parse_label("B-first_name", "BIO") == ("B", "first_name")
    assert parse_label("U-PATIENT", "BILOU") == ("U", "PATIENT")
    assert parse_label("O", "BIO") == ("O", None)
    assert parse_label("HOSPITAL", "IO") == ("I", "HOSPITAL")
    assert parse_label("I-DATE", "IO") == ("I", "I-DATE")  # IO models use bare type names


def test_decode_bio_lenient():
    tags = [("B", "X"), ("I", "X"), ("O", None), ("I", "X"), ("B", "X"), ("I", "Y")]
    assert decode_tags(tags, "BIO") == [(0, 1, "X"), (3, 3, "X"), (4, 4, "X"), (5, 5, "Y")]


def test_decode_bilou_and_io():
    tags = [("B", "P"), ("L", "P"), ("U", "P"), ("I", "D"), ("L", "D")]
    assert decode_tags(tags, "BILOU") == [(0, 1, "P"), (2, 2, "P"), (3, 4, "D")]
    io = [("I", "DATE"), ("I", "DATE"), ("I", "TIME"), ("O", None), ("I", "DATE")]
    assert decode_tags(io, "IO") == [(0, 1, "DATE"), (2, 2, "TIME"), (4, 4, "DATE")]


def test_merge_adjacent_name_pieces_only():
    text = "Jane Roe, J. Doe 2101-01-01 2101-01-02"
    ents = [Entity(0, 4, "PERSON", fine_label="first_name"), Entity(5, 8, "PERSON", fine_label="last_name"),
            Entity(10, 11, "PERSON"), Entity(13, 16, "PERSON"),
            Entity(17, 27, "DATE"), Entity(28, 38, "DATE")]
    out = merge_adjacent(ents, text, ["PERSON", "LOCATION"])
    assert [(text[e.start:e.end], e.label) for e in out] == [
        ("Jane Roe", "PERSON"), ("J. Doe", "PERSON"), ("2101-01-01", "DATE"), ("2101-01-02", "DATE")]
    assert out[0].fine_label == "first_name+last_name"


def test_touching_fragments_of_any_category_are_rejoined():
    text = "on 03/04/2101 call (555) 123-4567."
    ents = [Entity(3, 5, "DATE"), Entity(5, 6, "DATE"), Entity(6, 8, "DATE"), Entity(8, 13, "DATE"),
            Entity(19, 20, "PHONE"), Entity(20, 33, "PHONE")]
    out = merge_adjacent(ents, text, [])
    assert [text[e.start:e.end] for e in out] == ["03/04/2101", "(555) 123-4567"]


@pytest.mark.parametrize("span,expected", [
    ("Roe,", "Roe"), ("(Roe)", "(Roe)"), ("(Roe", "Roe"), ("Roe)", "Roe"), ("\"Jane Roe\"", "Jane Roe"),
    ("Mustermann.", "Mustermann"), ("Jane Q.", "Jane Q."), ("Acme Inc.", "Acme Inc."), ("(555) 123-4567", "(555) 123-4567"),
    (":2101-01-01;", "2101-01-01"),
])
def test_trim_punctuation(span, expected):
    from deid.detectors.hf_detector import trim_punctuation
    s, e = trim_punctuation(span, 0, len(span))
    assert span[s:e] == expected


def test_label_maps_cover_all_shortlist_types():
    maps = load_labels()["models"]
    assert maps["i2b2_2014"]["STAFF"] == "PERSON" and maps["stanford_deid"]["HCW"] == "PERSON"
    assert maps["nemotron_pii"]["gender"] is None and maps["nemotron_pii"]["date_of_birth"] == "DATE"


# ---------------------------------------------------------------- end-to-end with a fake model
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "seen", "by", "jane", "roe", "on", "the", "ward", ".", "##s", "today",
         "2101", "-", "03", "04", "age", "67", "1950"]
LABELS = {0: "O", 1: "B-first_name", 2: "I-first_name", 3: "B-last_name", 4: "B-gender", 5: "B-date", 6: "B-age"}


class FakeModel:
    """Logit 10 for: jane->B-first_name, roe->B-last_name, ##s->I-first_name (should be ignored:
    only first sub-tokens count); everything else O. Records window lengths."""

    def __init__(self, tok):
        self.ids = {tok.convert_tokens_to_ids(t): lab for t, lab in
                    [("jane", 1), ("roe", 3), ("##s", 2), ("2101", 5), ("67", 6), ("1950", 6)]}
        self.special = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}
        self.lengths = []

    def __call__(self, input_ids, attention_mask):
        b, n = input_ids.shape
        logits = torch.zeros((b, n, len(LABELS)))
        for i in range(b):
            self.lengths.append(int(attention_mask[i].sum()))
            for j in range(n):
                logits[i, j, self.ids.get(int(input_ids[i, j]), 0)] = 10.0
        return type("Out", (), {"logits": logits})()


@pytest.fixture
def detector(tmp_path):
    from transformers import BertTokenizerFast
    vf = tmp_path / "vocab.txt"
    vf.write_text("\n".join(VOCAB), encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(vf), do_lower_case=True)
    det = object.__new__(HFTokenClassifier)
    det.spec = ModelSpec("fake", "fake", None, "nemotron_pii", "BIO", window_tokens=8, context_tokens=2)
    det.source, det.batch_size, det.decision, det.o_threshold = "fake", 3, "argmax", 0.5
    det.merge_categories, det.normalize_text, det.torch = ("PERSON",), True, torch
    det.tokenizer, det.n_special, det.pad_id = tok, 2, tok.pad_token_id
    det.tags = [parse_label(LABELS[i], "BIO") for i in range(len(LABELS))]
    det.label_map, det.o_index = load_labels()["models"]["nemotron_pii"], 0
    det.model, det.backend = FakeModel(tok), "torch"
    det.word_unit, det.age_min = "text", 90
    return det


def test_long_note_names_found_across_chunk_boundaries(detector):
    unit = "seen by Jane Roe on the ward . "
    text = unit * 12 + "Janes today"   # ~100 tokens >> 8-token windows
    ents = detector.detect(text)
    names = [(text[e.start:e.end], e.label, e.fine_label) for e in ents]
    assert names[:12] == [("Jane Roe", "PERSON", "first_name+last_name")] * 12
    assert names[12] == ("Janes", "PERSON", "first_name")  # word = first sub-token label, span = whole word
    assert len(names) == 13
    assert max(detector.model.lengths) <= 8                 # window incl. [CLS]/[SEP] never exceeded


def test_offsets_refer_to_original_text_with_unicode(detector):
    text = "seen by “Jane Roe” today"
    ents = detector.detect(text)
    assert [text[e.start:e.end] for e in ents] == ["Jane Roe"]


def test_empty_input(detector):
    assert detector.detect("") == [] and detector.detect("   \n") == []


def test_text_word_unit_keeps_hyphenated_date_whole(detector):
    """Only the first piece "2101" is labelled (inner pieces get O, as for a model trained on whole
    tokens). Text words recover the full date; tokenizer words leave a fragment."""
    text = "seen on 2101-03-04 by Jane Roe"
    assert [text[e.start:e.end] for e in detector.detect(text) if e.label == "DATE"] == ["2101-03-04"]
    detector.word_unit = "tokenizer"
    assert [text[e.start:e.end] for e in detector.detect(text) if e.label == "DATE"] == ["2101"]


def test_age_scope_rule(detector):
    text = "age 67 , age 1950 ."
    assert [(text[e.start:e.end], e.label) for e in detector.detect(text)] == [("1950", "DATE")]


def test_text_words_peel_edge_punctuation():
    from deid.detectors.hf_detector import text_words
    t = '(555) Roe, "J. Doe" 06:20AM.'
    assert [t[s:e] for s, e in text_words(t)] == ["(", "555", ")", "Roe", ",", '"', "J", ".", "Doe", '"', "06:20AM", "."]


def test_whitespace_only_tokens_belong_to_no_word():
    """RoBERTa emits a token for "\n" (offsets cover only whitespace) and empty-offset space tokens.
    They must not become the 'first sub-token' of the following word."""
    from deid.detectors.hf_detector import text_words
    t = "Roe\nDOB:  1950-01-02"
    offsets = [(0, 3), (3, 4), (4, 5), (5, 7), (7, 8), (9, 9), (9, 14), (14, 15), (15, 17), (17, 18), (18, 20)]
    ids = HFTokenClassifier.text_word_ids(offsets, text_words(t), t)
    words = text_words(t)
    assert ids[1] is None and ids[5] is None                      # "\n" and the empty-offset space token
    assert [t[slice(*words[i])] for i in (ids[0], ids[2], ids[6])] == ["Roe", "DOB", "1950-01-02"]
