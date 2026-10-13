"""v3 regex rules: more Indian-English and general PHI patterns, plus gazetteers. Synthetic text only."""
import pytest

from deid.config import load_yaml
from deid.detectors.regex_detector import RegexDetector, luhn_valid
from deid.postprocess import strip_person_titles
from deid.schema import Entity


@pytest.fixture(scope="module")
def det():
    return RegexDetector.from_config(load_yaml("configs/detection.yaml"))


def found(det, text):
    return [(text[e.start:e.end], e.label) for e in det.detect(text)]


def luhn_card(prefix15: str) -> str:
    for d in "0123456789":
        if luhn_valid(prefix15 + d):
            return prefix15 + d
    raise AssertionError


# ---------------------------------------------------------------- dates and times
@pytest.mark.parametrize("text,expected", [
    ("DOA: 12.03.24 admitted", "12.03.24"), ("DOS - 2024/03/14 done", "2024/03/14"),
    ("LMP: 05-01-2024.", "05-01-2024"), ("EDD 12 10 2024 noted", "12 10 2024"),
    ("Date of surgery: 3/7/2024 uneventful", "3/7/2024"),
])
def test_labelled_dates(det, text, expected):
    assert (expected, "DATE") in found(det, text)


def test_labelled_date_must_be_a_valid_date(det):
    assert found(det, "DOA: 45.67.89 corrupted") == []


def test_month_apostrophe_and_oclock(det):
    assert found(det, "Last seen Mar'24 and March '23") == [("Mar'24", "DATE"), ("March '23", "DATE")]
    assert found(det, "seen at 10 o'clock") == [("10 o'clock", "TIME")]


# ---------------------------------------------------------------- phones
@pytest.mark.parametrize("text,expected", [
    ("Mob 9848 022 338 now", "9848 022 338"), ("Ph (040) 2345 6789 now", "(040) 2345 6789"),
    ("Alt No: 9123456780", "9123456780"), ("Emergency contact no: 040-23456789", "040-23456789"),
])
def test_more_indian_phones(det, text, expected):
    assert (expected, "PHONE") in found(det, text)


# ---------------------------------------------------------------- identifiers
def test_payment_and_tax_ids(det):
    card = luhn_card("453914880343646")
    spaced = " ".join(card[i:i + 4] for i in range(0, 16, 4))
    assert (spaced, "ID") in found(det, f"card {spaced} charged")
    bad = card[:-1] + str((int(card[-1]) + 1) % 10)
    assert found(det, f"ref {' '.join(bad[i:i + 4] for i in range(0, 16, 4))} only") == []   # fails Luhn
    assert found(det, "UPI ramesh.k@okaxis paid") == [("ramesh.k@okaxis", "ID")]
    assert found(det, "GSTIN 36AABCU9603R1ZM") == [("36AABCU9603R1ZM", "ID")]


@pytest.mark.parametrize("text,expected", [
    ("DL KA0120110012345 valid", "KA0120110012345"), ("EPIC ABC1234567 issued", "ABC1234567"),
    ("Reg: TSMC/FMR/12345", "TSMC/FMR/12345"), ("CGHS Ben ID: 12345678", "12345678"),
    ("ESIC No: 3412345678", "3412345678"), ("Ration card no: AP123456789", "AP123456789"),
    ("Staff ID: E12345", "E12345"), ("A/c No: 123456789012", "123456789012"), ("Medicare ID: 1EG4TE5MK73", "1EG4TE5MK73"),
])
def test_more_ids(det, text, expected):
    assert (expected, "ID") in found(det, text)


def test_network_device_vehicle(det):
    assert found(det, "MAC 00:1A:2B:3C:4D:5E") == [("00:1A:2B:3C:4D:5E", "OTHER_PHI")]
    ip6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
    assert found(det, f"IPv6 {ip6}") == [(ip6, "OTHER_PHI")]
    assert found(det, "car 22 BH 1234 AB parked") == [("22 BH 1234 AB", "OTHER_PHI")]


# ---------------------------------------------------------------- names
def test_name_labels_and_nested_titles(det):
    t = ("Name of the patient: Kavita Rao. Patient's name: Anil Kumar. Examined by: Dr. (Mrs.) Anita Rao. "
         "Prof. Dr. Ravi Teja. Capt. Arjun Singh")
    assert [x for x, l in found(det, t) if l == "PERSON"] == ["Kavita Rao", "Anil Kumar", "Anita Rao", "Ravi Teja", "Arjun Singh"]


def test_indian_degree_suffixes(det):
    assert found(det, "Anil Sharma, MBBS, DNB; Ravi Kumar (MCh)") == [("Anil Sharma", "PERSON"), ("Ravi Kumar", "PERSON")]


def test_strip_nested_titles():
    t = "Dr. (Mrs.) Anita Rao and Late Shri Venkat Rao"
    ents = [Entity(0, 20, "PERSON"), Entity(25, 45, "PERSON")]
    assert [t[e.start:e.end] for e in strip_person_titles(ents, t)] == ["Anita Rao", "Venkat Rao"]


# ---------------------------------------------------------------- addresses and gazetteers
def test_landmark_and_rural_addresses(det):
    assert found(det, "Opp. SBI Bank, Ameerpet, Hyderabad - 500016.") == [("SBI Bank, Ameerpet, Hyderabad - 500016", "LOCATION")]
    assert found(det, "Vill. Rampur, P.O. Sonari, Dist. Jamshedpur") == [
        ("Rampur", "LOCATION"), ("Sonari", "LOCATION"), ("Jamshedpur", "LOCATION")]


def test_gazetteers(det):
    t = "Referred from AIIMS to NIMHANS; follow-up at Apollo; lives in Hyderabad and VISAKHAPATNAM"
    assert found(det, t) == [("AIIMS", "ORGANIZATION"), ("NIMHANS", "ORGANIZATION"), ("Apollo", "ORGANIZATION"),
                             ("Hyderabad", "LOCATION"), ("VISAKHAPATNAM", "LOCATION")]


# ---------------------------------------------------------------- ages
def test_age_decimals_and_month_spellings():
    det0 = RegexDetector(age_min=0)
    t = "Age: 2.5 years. Baby 6 mnths old. aged about 45 years"
    assert [x for x, l in found(det0, t) if l == "AGE"] == ["2.5", "6 mnths", "45"]


# ---------------------------------------------------------------- things that must NOT be flagged
@pytest.mark.parametrize("text", [
    "Lt. knee pain, Adv. CT scan; severe Mitral Stenosis, MS; Uncontrolled Diabetes, DM",
    "near total occlusion; behind the ear; near Total Occlusion, Left Main stenosis; Hb 10.30 g/dL",
    "St. John's wort; BP 120/80 at 0800; Platelets 250000; INR 1.2; 3 days ago; Gen. Surgery dept",
    "patient reports MS symptoms, dist. radius fracture, Col. ascending; findings s/o pneumonia; CT w/o contrast",
])
def test_v3_clinical_negatives(det, text):
    assert found(det, text) == []


def test_shift_month_apostrophe_dates():
    from deid.policies.transform import shift_date_string as shift
    assert shift("Mar'23", -40) == "Feb'23"
    assert shift("March '24", -40) == "February '24"
