"""v5 (Experiment 14) rules and post-processing. Synthetic examples only."""
import pytest

from deid.config import load_yaml
from deid.detectors.regex_detector import RegexDetector
from deid.postprocess import drop_non_phi, drop_state_names, fix_person_initials, postprocess_final, trim_markup
from deid.schema import Entity


@pytest.fixture(scope="module")
def det():
    return RegexDetector.from_config(load_yaml("configs/detection.yaml"))


def found(det, text):
    return [(e.label, text[e.start:e.end]) for e in det.detect(text)]


@pytest.mark.parametrize("text,value", [
    ("**Discharge Summary for MED70192262**", "MED70192262"),
    ("Chanchal Shelton-Reynolds, HPBN-96445655, Y4422665", "HPBN-96445655"),
    ("Refund Request for ECOM-60303 (MRN-881046) on Sunday", "MRN-881046"),
    ("**Patient ID:** I2810490 | **Appointment Date:**", "I2810490"),
    ("PrescriptionRequest for N6137594", "N6137594"),
])
def test_unlabelled_and_markdown_identifiers(det, text, value):
    assert ("ID", value) in found(det, text)


@pytest.mark.parametrize("text", [
    "ICD-10: S72001A", "COVID-19 negative", "G2P1L1A0 at 38 weeks", "PAN ABCPE1234F", "Hb 10.2, TLC 12,400/cumm",
    "BP 120/80, PR 88/min", "CPT 99213 billed", "lot 4512",
])
def test_codes_and_values_are_not_shape_ids(det, text):
    assert not [x for x in found(det, text) if x[0] == "ID" and x[1] not in ("ABCPE1234F",)]


@pytest.mark.skipif(not __import__("deid.config", fromlist=["resolve_path"]).resolve_path(
    "data/processed/common_lowercase_words.txt").exists(), reason="needs data/processed/common_lowercase_words.txt")
def test_surname_after_title_and_relative_names(det):
    assert ("PERSON", "Kumari") in found(det, "Review with Dr. Kumari on Monday")
    assert ("PERSON", "Kumari") in found(det, "SR: Dr Kumari    Unit Head: Prof. Pallavi Bhosale")
    assert not [x for x in found(det, "SR: Dr Kumari    Unit Head: Prof. Pallavi Bhosale") if x[1] == "Head"]
    assert ("PERSON", "Harjeet") in found(det, "Contact of attendant (son Harjeet): 9876543210")
    assert not [x for x in found(det, "His daughter Was present. Mother Has diabetes.") if x[0] == "PERSON"]


def test_more_address_forms(det):
    assert ("LOCATION", "Kaloor") in found(det, "Rajagiri Hospital\nKaloor, Kochi - 682655, Kerala | Ph: 0484 2345678")
    assert ("LOCATION", "Anakapalli") in found(det, "Native place - At/Post Anakapalli, Tal. Nakkapalli")
    got = found(det, "250, Shivaji Path, Near Water Tank, Badambadi, Cuttack - 753141 | Ph: 0671 2345678")
    assert ("LOCATION", "250, Shivaji Path, Near Water Tank, Badambadi, Cuttack - 753141") in got
    for t in ("Department of Medicine, Hyderabad", "2, Right Bundle Branch Block. 1, Main issue."):
        assert not [x for x in found(det, t) if x[0] == "LOCATION" and x[1] != "Hyderabad"], t
    assert ("LOCATION", "Rampur") in found(det, "Native: Vill. Rampur, P.O. Sonari")


def test_organisation_fields_and_scans(det):
    assert ("ORGANIZATION", "State Bank of India") in found(det, "Employer:\tState Bank of India | Aadhaar")
    assert ("ORGANIZATION", "Sri Lakshmi Scans") in found(det, "Sri Lakshmi Scans | Door No. 5/2")
    assert not [x for x in found(det, "CT Scans done. Employer: None") if x[0] == "ORGANIZATION"]
    assert ("ORGANIZATION", "KIDWAI MEMORIAL INSTITUTE OF ONCOLOGY") in found(det, "KIDWAI MEMORIAL INSTITUTE OF ONCOLOGY\nBengaluru")


def _ents(text, *items, src="model"):
    out = []
    for s, lab in items:
        i = text.index(s)
        out.append(Entity(i, i + len(s), lab, source=src))
    return out


def test_v5_filters():
    t = "Age/Gender: 45Y/F | Mob 98 | Ward: MICU | Inj. Lasix 40 mg | Blood Urea 35 | Name: [Blank] | Ed Smith came"
    ents = _ents(t, ("45Y/F", "ID"), ("Mob", "ID"), ("MICU", "PERSON"), ("Lasix", "PERSON"), ("Urea", "PERSON"),
                 ("Blank", "PERSON"), ("Ed Smith", "PERSON"))
    assert [t[e.start:e.end] for e in drop_non_phi(ents, t)] == ["Ed Smith"]


def test_markup_trim_states_and_initials():
    t = "**John Doe**, GA, Chandigarh, Kerala; by Dr. Debashis M.. Inj; Patient:  H Raghavendra; MOHD. ARIF MIRZA"
    assert [t[e.start:e.end] for e in trim_markup(_ents(t, ("**John Doe**", "PERSON")), t)] == ["John Doe"]
    locs = _ents(t, ("GA", "LOCATION"), ("Chandigarh", "LOCATION"), ("Kerala", "LOCATION"))
    assert [t[e.start:e.end] for e in drop_state_names(locs, t)] == ["Chandigarh"]
    es = _ents(t, ("Debashis", "PERSON"), ("Raghavendra", "PERSON"), ("MOHD", "PERSON"), ("ARIF MIRZA", "PERSON"))
    assert [t[e.start:e.end] for e in fix_person_initials(es, t)] == ["Debashis M.", "H Raghavendra", "MOHD. ARIF MIRZA"]


def test_rule_found_ids_without_digits_are_kept():
    """UPI and ABHA addresses can be letters only; only model noise without digits ("Mob") is dropped."""
    t = "Bill paid via UPI ramesh.k@okaxis. ABHA: sunitak@abdm. Mob 98"
    ents = _ents(t, ("ramesh.k@okaxis", "ID"), ("sunitak@abdm", "ID"), src="regex") + _ents(t, ("Mob", "ID"))
    assert [t[e.start:e.end] for e in drop_non_phi(ents, t)] == ["ramesh.k@okaxis", "sunitak@abdm"]


def test_kumari_alone_is_kept_as_a_surname():
    cfg = load_yaml("configs/detection.yaml")
    t = "Review with Dr. Kumari tomorrow."
    assert [t[e.start:e.end] for e in postprocess_final(_ents(t, ("Kumari", "PERSON"), src="regex"), t, cfg)] == ["Kumari"]
