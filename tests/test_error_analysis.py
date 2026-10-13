"""Error analysis must never leak span text or names through context words."""
from deid.evaluation.benchmark import EvalNote
from deid.evaluation.error_analysis import error_report, render_markdown
from deid.schema import Entity


def make_note(i: int, name: str = "Zelda Quixote") -> EvalNote:
    text = f"Admission Date: 2101-01-0{i % 9 + 1}\nName: {name} DOB 1950-01-01 {name} {name}"
    s = text.index(name)
    gold = [Entity(16, 26, "DATE"), Entity(s, s + len(name), "PERSON")]
    return EvalNote(f"n{i}", text, gold, [])


def test_names_in_context_are_masked_even_when_frequent():
    notes = [make_note(i) for i in range(12)]          # same name in 12 notes and repeated in each
    preds = {n.note_id: [] for n in notes}               # everything missed
    rep = error_report(notes, preds)
    md = render_markdown(rep, "x")
    assert "zelda" not in md.lower() and "quixote" not in md.lower()
    assert "admission date:" in md                       # genuine template context survives (12 notes)


def test_rare_context_is_hidden():
    notes = [make_note(i) for i in range(3)]
    rep = error_report(notes, {n.note_id: [] for n in notes})
    assert all(c == "<rare>" for g in rep["fn_groups"] for c, _ in g["top_contexts"])
