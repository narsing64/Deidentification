"""v2 regex rules (Indian formats, time fix, hospital names, ages) and name-variant linking. Synthetic only."""
import pytest

from deid.config import load_yaml
from deid.detectors.regex_detector import RegexDetector, verhoeff_valid
from deid.policies.transform import PolicyEngine
from deid.schema import Entity


@pytest.fixture(scope="module")
def det():
    return RegexDetector(optional_rules=["address_zip"])


def found(det, text):
    return [(text[e.start:e.end], e.label) for e in det.detect(text)]


def valid_aadhaar(base11: str) -> str:
    for d in "0123456789":
        if verhoeff_valid(base11 + d):
            return base11 + d
    raise AssertionError


# ---------------------------------------------------------------- time fix
def test_time_never_swallows_sentence_full_stop(det):
    assert found(det, "seen at 10:30 AM. Next") == [("10:30 AM", "TIME")]
    assert found(det, "at 9 pm. Later") == [("9 pm", "TIME")]
    assert found(det, "at 10:30 a.m. today") == [("10:30 a.m.", "TIME")]


@pytest.mark.parametrize("text,expected", [
    ("at 14:05:30 today", "14:05:30"), ("at 23:59 today", "23:59"), ("at 1430 hrs", "1430 hrs"),
    ("at 08.00 hrs", "08.00 hrs"), ("at 10.30 am", "10.30 am"), ("at 0800h", "0800h"),
])
def test_time_formats(det, text, expected):
    assert found(det, text) == [(expected, "TIME")]


def test_decimals_and_ratios_are_not_times(det):
    assert found(det, "Hb 10.30 g/dL, ratio 1:100, INR 2.30") == []


# ---------------------------------------------------------------- Indian phones and identifiers
@pytest.mark.parametrize("text,expected", [
    ("Mob: 9876543210.", "9876543210"), ("call +91 98765 43210 now", "+91 98765 43210"),
    ("call +91-9876543210 now", "+91-9876543210"), ("call 098765-43210 now", "098765-43210"),
    ("ph 040-23456789 now", "040-23456789"), ("ph +91 40 2345 6789 now", "+91 40 2345 6789"),
])
def test_indian_phones(det, text, expected):
    assert (expected, "PHONE") in found(det, text)


def test_aadhaar_checksum_and_masked():
    det = RegexDetector()
    a = valid_aadhaar("23456789012")
    spaced = f"{a[:4]} {a[4:8]} {a[8:]}"
    assert found(det, f"Aadhaar {spaced} on file") == [(spaced, "ID")]
    bad = a[:-1] + str((int(a[-1]) + 1) % 10)
    assert found(det, f"value {bad[:4]} {bad[4:8]} {bad[8:]}") == []           # fails Verhoeff
    assert found(det, "Aadhaar No: XXXX XXXX 1234") == [("XXXX XXXX 1234", "ID")]


@pytest.mark.parametrize("text,expected", [
    ("PAN ABCPE1234F", ("ABCPE1234F", "ID")),
    ("ABHA 12-3456-7890-1234", ("12-3456-7890-1234", "ID")),
    ("ABHA address ramesh.k@abdm", ("ramesh.k@abdm", "ID")),
    ("UHID: MH/2024/00123", ("MH/2024/00123", "ID")),
    ("IP No: 24-56789", ("24-56789", "ID")),
    ("OPD No. 778899", ("778899", "ID")),
    ("Passport No: K1234567", ("K1234567", "ID")),
    ("pacemaker S/N: PM2024X99", ("PM2024X99", "ID")),
    ("vehicle TS 09 EA 1234 involved", ("TS 09 EA 1234", "OTHER_PHI")),
])
def test_indian_ids(det, text, expected):
    assert expected in found(det, text)


# ---------------------------------------------------------------- dates, addresses, names, hospitals
@pytest.mark.parametrize("text,expected", [
    ("DOA: 12-Mar-2024", "12-Mar-2024"), ("on 12-Mar-24.", "12-Mar-24"), ("on 05/Jan/2023", "05/Jan/2023"),
    ("on 12.03.2024", "12.03.2024"), ("DOD: 12/03/2024", "12/03/2024"),
])
def test_indian_dates(det, text, expected):
    assert (expected, "DATE") in found(det, text)


def test_address_line_and_pin(det):
    t = "Address: H.No 12-3-45, Banjara Hills, Hyderabad - 500034\nDiagnosis: CAD"
    assert found(det, t) == [("H.No 12-3-45, Banjara Hills, Hyderabad - 500034", "LOCATION")]
    assert found(det, "PIN code: 560034") == [("560034", "LOCATION")]
    assert found(det, "Please address the patient's concerns today.") == []


def test_relation_names_and_shorthand(det):
    t = "Ramesh Kumar S/O Suresh Kumar; Smt. Lakshmi Devi wife of Late Venkat Rao; CT ABDOMEN W/O CONTRAST"
    assert [x for x, l in found(det, t) if l == "PERSON"] == ["Ramesh Kumar", "Suresh Kumar", "Lakshmi Devi", "Venkat Rao"]
    assert found(det, "findings s/o pneumonia; CT w/o contrast; h/o DM; c/o chest pain; r/o MI") == []


def test_indian_honorific_titles(det):
    assert found(det, "seen by Shri Anil Sharma and Smt. Kavita Rao") == [("Anil Sharma", "PERSON"), ("Kavita Rao", "PERSON")]


def test_hospital_names_two_words_generic_rejected(det):
    t = "Medicover Hospitals, Apollo Hospitals Pvt Ltd, CARE Hospitals, KIMS Hospital, Yashoda Hospitals"
    assert [x for x, l in found(det, t) if l == "ORGANIZATION"] == [
        "Medicover Hospitals", "Apollo Hospitals Pvt Ltd", "CARE Hospitals", "KIMS Hospital", "Yashoda Hospitals"]
    assert found(det, "Seen at Trauma Center, Medical Center and Primary Health Centre.") == []


# ---------------------------------------------------------------- ages in years / months / days
def test_age_never_read_from_inside_a_date_or_number():
    det0 = RegexDetector(age_min=0)
    assert [l for _, l in found(det0, "Age: 2101-03-04 seen; age 45.5 kg")] == ["DATE"]


def test_ages_default_policy_only_90_plus(det):
    assert found(det, "Age/Sex: 92 Y/M; 45/M; aged 3 months; 10 days old") == [("92", "AGE")]


def test_ages_all_units_when_policy_lowers_age_min():
    det = RegexDetector(age_min=0)
    t = "Age/Sex: 45/M. Baby aged 3 months; 10 days old; age: 5 yrs 3 months; 2 weeks old"
    got = [x for x, l in found(det, t) if l == "AGE"]
    assert got == ["45", "3 months", "10 days", "5 yrs 3 months", "2 weeks"]


def test_clinical_values_still_untouched(det):
    assert found(det, "Metoprolol 25 mg BID, BP 120/80, Hb 10.2 g/dL, fever for 3 days, 2/3 doses") == []


# ---------------------------------------------------------------- name-variant linking (policy)
def test_name_variants_share_one_pseudonym_within_and_across_notes():
    eng = PolicyEngine(load_yaml("configs/deid_policy.yaml"), key=b"link-test-key-not-secret-0000000000")
    t1 = "Name: Jane Q. Roe. Ms. Roe was seen. Dr. Max Mustermann; later Dr. Mustermann."
    ents1 = [Entity(t1.index(x), t1.index(x) + len(x), "PERSON") for x in ("Jane Q. Roe", "Roe was", "Max Mustermann", "Mustermann.")]
    ents1 = [e.with_(end=e.end - (4 if t1[e.start:e.end] == "Roe was" else 1 if t1[e.start:e.end].endswith(".") else 0)) for e in ents1]
    out1 = eng.transform(t1, ents1, "n1", patient_id="P1")
    ps = [out1.text[r.out_start:r.out_end] for r in out1.records]
    assert ps[0] == ps[1] and ps[2] == ps[3] and ps[0] != ps[2]
    t2 = "Patient Jane Roe and Mr. Roe reviewed."
    ents2 = [Entity(8, 16, "PERSON"), Entity(25, 28, "PERSON")]
    out2 = eng.transform(t2, ents2, "n2", patient_id="P1")
    assert [out2.text[r.out_start:r.out_end] for r in out2.records] == [ps[0], ps[0]]


def test_shared_surname_is_not_guessed():
    eng = PolicyEngine(load_yaml("configs/deid_policy.yaml"), key=b"link-test-key-not-secret-0000000000")
    t = "Jane Roe and John Roe; Roe called."
    ents = [Entity(0, 8, "PERSON"), Entity(13, 21, "PERSON"), Entity(23, 26, "PERSON")]
    out = eng.transform(t, ents, "n", patient_id="P9")
    p = [out.text[r.out_start:r.out_end] for r in out.records]
    assert p[0] != p[1] and p[2] not in (p[0], p[1])


# ---------------------------------------------------------------- date shifting: Indian formats, day-first
def test_shift_indian_date_formats():
    from deid.policies.transform import shift_date_string as shift
    assert shift("12-Mar-2024", -7) == "05-Mar-2024"
    assert shift("12-Mar-24", -7) == "05-Mar-24"
    assert shift("05/Jan/2023", -7) == "29/Dec/2022"
    assert shift("05/03/2024", -7) == "04/26/2024"                    # US reading by default
    assert shift("05/03/2024", -7, day_first=True) == "27/02/2024"    # 5 March in a day-first note


def test_note_level_day_first_keeps_intervals():
    eng = PolicyEngine(load_yaml("configs/deid_policy.yaml"), key=b"date-test-key-not-secret-000000000")
    t = "DOA 05/03/2024, DOD 18/03/2024"           # 18/03 can only be day-first -> whole note is day-first
    ents = [Entity(4, 14, "DATE"), Entity(20, 30, "DATE")]
    out = eng.transform(t, ents, "n", patient_id="P5")
    import datetime as dt
    a, b = (dt.datetime.strptime(out.text[r.out_start:r.out_end], "%d/%m/%Y").date() for r in out.records)
    assert (b - a).days == 13                       # 5 March -> 18 March interval preserved
