"""Synthetic tests for the shared metric definitions (docs: deid.evaluation.metrics)."""
from deid.evaluation.metrics import CorpusScore, prf, score_document
from deid.schema import Entity, Region

TEXT = "Seen by John Smith on 2020-01-02."
NAME = Entity(8, 18, "PERSON")        # "John Smith"
DATE = Entity(22, 32, "DATE")         # "2020-01-02"


def test_exact_match_counts_for_both_metrics():
    ds = score_document([NAME], [Entity(8, 18, "PERSON", source="m")])
    assert (ds.strict.tp, ds.strict.fp, ds.strict.fn) == (1, 0, 0)
    assert (ds.agnostic.tp, ds.agnostic.fp, ds.agnostic.fn) == (1, 0, 0)


def test_wrong_label_is_strict_error_but_agnostic_hit():
    ds = score_document([NAME], [Entity(8, 18, "DATE")])
    assert (ds.strict.tp, ds.strict.fp, ds.strict.fn) == (0, 1, 1)
    assert (ds.agnostic.tp, ds.agnostic.fp, ds.agnostic.fn) == (1, 0, 0)
    assert ds.wrong_label_spans == 1
    assert ds.confusion[("PERSON", "DATE")] == 1
    assert ds.fn_breakdown["wrong_label"] == 1 and ds.fp_breakdown["wrong_label"] == 1


def test_partial_span_fails_both_metrics():
    ds = score_document([NAME], [Entity(8, 12, "PERSON")])  # "John"
    assert (ds.strict.tp, ds.strict.fp, ds.strict.fn) == (0, 1, 1)
    assert (ds.agnostic.tp, ds.agnostic.fp, ds.agnostic.fn) == (0, 1, 1)
    assert ds.fn_breakdown["boundary"] == 1 and ds.fp_breakdown["boundary"] == 1
    # the overlap diagnostic sees it, separately
    assert ds.overlap["gold_overlapped"] == 1 and ds.overlap["gold_chars_covered"] == 4


def test_missing_entity_and_extra_prediction():
    ds = score_document([NAME, DATE], [Entity(8, 18, "PERSON"), Entity(0, 4, "PERSON")])
    assert (ds.strict.tp, ds.strict.fp, ds.strict.fn) == (1, 1, 1)
    assert ds.fn_breakdown["missed"] == 1
    assert ds.fp_breakdown["spurious"] == 1


def test_duplicate_predictions_do_not_inflate_tp_or_fp():
    pred = [Entity(8, 18, "PERSON", source="a"), Entity(8, 18, "PERSON", source="b")]
    ds = score_document([NAME], pred)
    assert (ds.strict.tp, ds.strict.fp, ds.strict.fn) == (1, 0, 0)
    assert ds.duplicate_predictions == 1


def test_same_span_two_labels():
    ds = score_document([NAME], [Entity(8, 18, "PERSON"), Entity(8, 18, "ORGANIZATION")])
    assert (ds.strict.tp, ds.strict.fp, ds.strict.fn) == (1, 1, 0)       # wrong-label copy is FP
    assert (ds.agnostic.tp, ds.agnostic.fp, ds.agnostic.fn) == (1, 0, 0)  # span counted once
    assert ds.wrong_label_spans == 0


def test_overlapping_predictions():
    pred = [Entity(8, 18, "PERSON"), Entity(13, 25, "PERSON")]
    ds = score_document([NAME, DATE], pred)
    assert (ds.strict.tp, ds.strict.fp, ds.strict.fn) == (1, 1, 1)
    assert ds.fn_breakdown["boundary"] == 1   # DATE overlapped only partially


def test_ignore_regions_remove_predictions():
    ds = score_document([NAME], [Entity(8, 18, "PERSON"), Entity(0, 4, "ID")], ignore=[Region(0, 5, "possible_leak")])
    assert ds.ignored_predictions == 1
    assert (ds.strict.tp, ds.strict.fp) == (1, 0)


def test_empty_denominators():
    assert prf(0, 0, 0) == {"precision": None, "recall": None, "f1": None}
    assert prf(0, 0, 3) == {"precision": None, "recall": 0.0, "f1": 0.0}
    assert prf(0, 2, 0) == {"precision": 0.0, "recall": None, "f1": 0.0}
    ds = score_document([], [])
    assert ds.strict.prf()["f1"] is None


def test_per_category_and_micro_average():
    cs = CorpusScore()
    cs.add("d1", score_document([NAME, DATE], [Entity(8, 18, "PERSON")]))
    cs.add("d2", score_document([DATE], [Entity(22, 32, "DATE"), Entity(0, 4, "DATE")]))
    s = cs.summary()
    assert s["strict"]["tp"] == 2 and s["strict"]["fp"] == 1 and s["strict"]["fn"] == 1
    assert s["per_category"]["DATE"] == {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert s["per_category"]["PERSON"]["recall"] == 1.0


def test_bootstrap_is_deterministic():
    cs = CorpusScore()
    for i in range(5):
        cs.add(str(i), score_document([NAME], [Entity(8, 18, "PERSON")] if i % 2 else []))
    a = cs.bootstrap(200, seed=1, confidence=0.9)
    b = cs.bootstrap(200, seed=1, confidence=0.9)
    assert a == b
    lo, hi = a["strict"]["recall"]
    assert 0.0 <= lo <= hi <= 1.0
