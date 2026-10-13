"""Offset-preserving normalisation."""
from deid.preprocessing.normalize import normalize


def test_typographic_folds_are_length_preserving():
    t = "O’Neil “seen” 3–4 • x y…"
    n = normalize(t)
    assert n.text == "O'Neil \"seen\" 3-4 * x y."
    assert n.is_identity_mapping
    assert n.to_original(0, 6) == (0, 6)


def test_digits_units_and_meaning_untouched():
    t = "Metoprolol 25 mg, K 4.2 mmol/L, 5°C, 10 µg"
    assert normalize(t).text == t  # nothing clinically meaningful is changed


def test_whitespace_collapse_maps_back_to_original():
    t = "Name:     Jane   Roe\tok"
    n = normalize(t, collapse_spaces=True)
    assert n.text == "Name: Jane Roe\tok"
    s = n.text.index("Jane Roe")
    os_, oe = n.to_original(s, s + len("Jane Roe"))
    assert t[os_:oe] == "Jane   Roe"
    assert n.to_original(len(n.text), len(n.text)) == (len(t), len(t))


def test_empty():
    n = normalize("")
    assert n.text == "" and n.to_orig == (0,)
