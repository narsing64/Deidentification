"""v4 (Experiment 13) rules and post-processing: generalised identifiers, addresses, organisations, relations,
and filters for things that cannot be PHI. Synthetic examples only."""
import pytest

from deid.config import load_yaml
from deid.detectors.regex_detector import RegexDetector
from deid.postprocess import (_vocabulary, drop_non_phi, drop_state_names, fix_person_initials, postprocess_final,
                              strip_person_titles)
from deid.schema import Entity


def _vocab_available() -> bool:
    from deid.config import resolve_path
    return resolve_path(load_yaml("configs/detection.yaml")["regex"]["org_common_words"]).exists()


# The ordinary-vocabulary list is built from the licensed unlabelled notes (scripts/build_common_words.py) and is
# not distributed; tests that need it are skipped when it is absent.
needs_vocab = pytest.mark.skipif(not _vocab_available(), reason="needs data/processed/common_lowercase_words.txt")


@pytest.fixture(scope="module")
def det():
    return RegexDetector.from_config(load_yaml("configs/detection.yaml"))


def found(det, text):
    return [(e.label, text[e.start:e.end]) for e in det.detect(text)]


@pytest.mark.parametrize("text,value", [
    ("UHID No.: SH-751675", "SH-751675"),
    ("MR No. : AH.0397836236", "AH.0397836236"),
    ("Hosp. No.:  MI-324724", "MI-324724"),
    ("Adm. No.: ADM/2022/18210", "ADM/2022/18210"),
    ("MLC No. 374/24 registered", "374/24"),
    ("Policy No.: P/537533/01/2024/978319", "P/537533/01/2024/978319"),
    ("Claim / Intimation No.: CLM2408181324", "CLM2408181324"),
    ("Visit No.:\t8773403", "8773403"),
    ("Accession No.: CT/24/94441", "CT/24/94441"),
    ("SID: ND2201245371", "ND2201245371"),
    ("Nikshay ID: 59689757", "59689757"),
])
def test_labelled_identifiers(det, text, value):
    assert ("ID", value) in found(det, text)


def test_combined_identifier_fields(det):
    t = "UHID / IP No.:  SVM.0766044532 / IPD-2024-84775"
    got = found(det, t)
    assert ("ID", "SVM.0766044532") in got and ("ID", "IPD-2024-84775") in got


@pytest.mark.parametrize("text", [
    "Room No. 12", "Bed No. 1234", "Ward No. 22", "Batch No. 4512", "Platelets - 150000 /cumm",
    "Date No. 12/03/2024", "Token No. 145",
])
def test_non_identifier_numbers_untouched(det, text):
    assert not [x for x in found(det, text) if x[0] in ("ID", "LOCATION")]


@needs_vocab
def test_relations_in_name_lines_but_not_shorthand(det):
    assert ("PERSON", "VENKATESH BABU") in found(det, "Name: Mr SAI KIRAN BABU s/o VENKATESH BABU")
    assert ("PERSON", "ABDUL RAHMAN QURESHI") in found(det, "Relation: W/O Shri ABDUL RAHMAN QURESHI")
    for t in ("CT chest w/o contrast.", "CT W/O CONTRAST", "Findings s/o pneumonia.", "C/o: Intermittent fever"):
        assert not [x for x in found(det, t) if x[0] == "PERSON"], t


def test_upper_case_titles_and_mitral_regurgitation(det):
    assert ("PERSON", "Gounder") in found(det, "SR: DR. Gounder")
    assert not [x for x in found(det, "Echo: moderate MR Severe TR") if x[0] == "PERSON"]
    assert not [x for x in found(det, "Address of Patient:\tVill. Annur, P.O. Pollachi") if x[0] == "PERSON"]


def test_house_number_addresses_and_pin_codes(det):
    got = found(det, "Star Hospitals\nA-684, Shanti Kunj, Malakpet, Hyderabad - 500263 | Ph: 040 2345678")
    assert ("LOCATION", "A-684, Shanti Kunj, Malakpet, Hyderabad - 500263") in got
    assert ("LOCATION", "411294") in found(det, "Wakad, Pune - 411294")
    assert ("LOCATION", "Kodungallur") in found(det, "Place:\tKodungallur")
    assert not [x for x in found(det, "GCS 14/15, Hb 10.2, 2-3 times a day, Lt. knee") if x[0] == "LOCATION"]


def test_organisation_fields_employers_and_generic_types(det):
    assert ("ORGANIZATION", "Indian Railways") in found(det, "Employer : Indian Railways")
    assert ("ORGANIZATION", "State Bank of India") in found(det, "He works at State Bank of India.")
    assert ("ORGANIZATION", "St. Mary's Maternity Home") in found(det, "Referred from St. Mary's Maternity Home, Patna")
    for t in ("COURSE IN THE HOSPITAL: managed", "Civil Hospital, Aluva", "Linked to DOTS centre at", "works in the ICU"):
        assert not [x for x in found(det, t) if x[0] == "ORGANIZATION"], t
    assert ("ORGANIZATION", "CARE Hospitals") in found(det, "Referred to CARE Hospitals")


def _ents(text, *items):
    out = []
    for s, lab in items:
        i = text.index(s)
        out.append(Entity(i, i + len(s), lab, source="model"))
    return out


@needs_vocab
def test_drop_non_phi_filters():
    t = ("Dr. Ramesh, MBBS, MD (Ortho). Kernig's sign neg. Thomas splint. normalised at 18 hrs; at 14:30 hrs; "
         "GCS 15/15; call 108. ABDOMEN & PELVIS. Civil Hospital. Apollo Hospitals. Mr Thomas Kurian came on 12/03.")
    ents = _ents(t, ("MD", "LOCATION"), ("Ortho", "LOCATION"), ("Kernig's", "PERSON"), ("Thomas", "PERSON"),
                 ("18 hrs", "TIME"), ("14:30 hrs", "TIME"), ("15/15", "DATE"), ("108", "OTHER_PHI"),
                 ("ABDOMEN & PELVIS", "ORGANIZATION"), ("Civil Hospital", "ORGANIZATION"),
                 ("Apollo Hospitals", "ORGANIZATION"), ("Thomas Kurian", "PERSON"), ("12/03", "DATE"))
    vocab = _vocabulary(load_yaml("configs/detection.yaml")["regex"]["org_common_words"])
    kept = [t[e.start:e.end] for e in drop_non_phi(ents, t, vocab)]
    assert kept == ["14:30 hrs", "Apollo Hospitals", "Thomas Kurian", "12/03"]


def test_state_names_are_kept_but_cities_removed():
    t = "Guntur, Andhra Pradesh"
    kept = [t[e.start:e.end] for e in drop_state_names(_ents(t, ("Guntur", "LOCATION"), ("Andhra Pradesh", "LOCATION")), t)]
    assert kept == ["Guntur"]


def test_person_initials_and_title_only_spans():
    t = "Seen by Dr. Anju N. today; JR: Dr. Ramesh V.  Patient:\tN Kalaiselvi | Mr. Mohd. Arif Shaikh | DR. Shah"
    es = _ents(t, ("Anju N", "PERSON"), ("Ramesh", "PERSON"), ("Kalaiselvi", "PERSON"), ("Mohd", "PERSON"),
               ("Arif Shaikh", "PERSON"), ("DR", "PERSON"))
    got = [t[e.start:e.end] for e in fix_person_initials(strip_person_titles(es, t), t)]
    assert got == ["Anju N.", "Ramesh V.", "N Kalaiselvi", "Mohd. Arif Shaikh"]


def test_postprocess_final_uses_v4_steps():
    cfg = load_yaml("configs/detection.yaml")
    t = "Kernig's sign negative at 18 hrs."
    assert postprocess_final(_ents(t, ("Kernig's", "PERSON"), ("18 hrs", "TIME")), t, cfg) == []
