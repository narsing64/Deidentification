"""Shared rule-based detector for structured PHI. Identical for every model candidate.

Every rule has a name (kept as `fine_label`), a normalised category, a pattern, an optional
value group (so field labels such as "MRN:" are not redacted) and an optional validator.
Guards keep clinical values untouched:
* dates need a valid month/day and a year, or an explicit date context;
* times need a valid hour:minute (two-digit minutes), an am/pm marker or an "hrs" suffix;
* numeric matches immediately followed by a dose/measurement unit are rejected;
* ages are only flagged when the age in years is >= `age_min` (HIPAA Safe Harbor: >89). Years,
  months, weeks and days are all understood, so lowering `age_min` also covers infant ages;
* Aadhaar numbers must pass the Verhoeff checksum; PAN must have a valid holder-type letter.
Rules were written from general clinical-text knowledge, validation-split statistics (US/MIMIC-style
notes) and, in v2, Indian record formats requested by the user (phones, Aadhaar, PAN, ABHA, UHID/IP
numbers, S/O-D/O-W/O relatives, honorifics, PIN codes, hospital names, "hrs" times).
v4 (Experiment 13, decided on DEV data only) generalises instead of listing formats:
* any "<identifier label> No./ID/Number: <value>" with a value of >= 3 digits, incl. "/" and "." inside, and
  combined fields ("UHID / IP No.: X / Y");
* unlabelled house-number addresses, PIN codes after a city/state, "Place:"/"City:" fields;
* "Employer:"/"Name of the Hospital:" fields and "works at ..." employers;
* lower-case and W/O relation markers, accepted only if the name is not ordinary vocabulary;
* upper-case titles (DR., SMT.), nursing titles (Sr., Staff Nurse); upper-case MR/MS are never titles.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from deid.preprocessing.normalize import normalize
from deid.schema import Entity

MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
         r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
UNIT = (r"(?:mg|mcg|ug|µg|g|kg|lbs?|ml|mL|L|dL|cc|units?|U|IU|mmol|mEq|meq|mmHg|%|bpm|cm|mm|"
        r"tabs?|caps?|puffs?|drops?|mins?|days?|weeks?|wks?|months?)")
_UNIT_AFTER = re.compile(rf"\s*{UNIT}(?![A-Za-z])")
NAME_WORD = r"[A-Z][a-zA-Z'\-]+"
AMPM = r"(?:[AaPp]\.[Mm]\.|[AaPp][Mm])"          # "AM", "pm", "a.m." -- never swallows a sentence full stop
# v4: upper-case forms and nursing titles. Upper-case "MR"/"MS" are NOT titles (mitral regurgitation /
# stenosis, "MR head") and "SN" is too ambiguous.
TITLES = (r"(?:Dr|DR|Mr|Mrs|MRS|Ms|Miss|MISS|Prof|PROF|Smt|SMT|Shri|SHRI|Sri|SRI|Shrimati|Kumari|Kum|Mx|Er|"
          r"Capt|Col|Maj|Brig|Sr|Staff[ \t]+Nurse|Sister)")
# "Dr. (Mrs.)", "Prof. Dr.", "Late Shri" ("Lt."/"Adv." are NOT titles here: Indian notes use them for
# "left" and "advised")
TITLES2 = TITLES.replace("|Kumari|Kum|", "|")     # v5: after "Dr." these are surnames ("Dr. Kumari")
TITLE_SEQ = rf"(?:Late[ \t]+)?{TITLES}\.?(?:[ \t]*\([ \t]*{TITLES}\.?[ \t]*\)|[ \t]+{TITLES2}\.?)?"
CRED_SUFFIX = (r"(?:MD|M\.D\.|DO|RN|NP|PA-C|PhD|MBBS|DNB|MCh|M\.Ch|FRCS|MRCP|DCH|DGO|BDS|MDS|BAMS|BHMS|BUMS|"
               r"FACS|FICS|MRCOG|DMRD)")  # MS / DM left out: also mitral stenosis / diabetes mellitus
CREDENTIALS = {"MD", "M.D.", "DO", "RN", "NP", "PA", "PA-C", "PhD", "MBBS", "DDS", "FNP", "CNP", "Jr", "Sr",
               "MS", "DNB", "DM", "MCh", "FRCS", "MRCP", "DCH", "DGO", "BDS", "MDS", "BAMS", "BHMS", "BUMS",
               "FACS", "FICS", "MRCOG", "DMRD"}
NAME_STOP = {
    "The", "A", "An", "And", "Or", "Of", "In", "On", "At", "To", "For", "With", "From", "By", "As",
    "Was", "Is", "Has", "Had", "Will", "He", "She", "His", "Her", "They", "This", "That", "Who",
    "Patient", "Pt", "Unknown", "None", "Not", "Na", "N", "Discharge", "Admission", "Admit", "Date",
    "Time", "Service", "Medicine", "Surgery", "Attending", "Department", "Dept", "Hospital", "Clinic",
    "Center", "Office", "Team", "Name", "Doctor", "Physician", "Provider", "Nurse", "Follow", "Up",
    "Male", "Female", "Sex", "Age", "DOB", "MRN", "ID", "Phone", "Fax", "Address", "Signed",
    "Unit", "No", "Number", "Room", "Bed", "Floor", "Ward", "Allergies", "Chief", "Complaint",
    "Mr", "Mrs", "Ms", "Dr", "Smt", "Shri", "Sri", "Shrimati", "Kumari", "Kum", "Late", "Baby", "Master",
    "Consultant", "Resident", "Guardian", "Relative", "Attendant", "Informant", "Lanka",
    "Mx", "Er", "Capt", "Col", "Maj", "Brig",
    "Self", "SELF", "Head", "HEAD", "Blank",          # v5: "Referred by: SELF", "Unit Head:", "[Blank]"
}
ORG_LEADING_STOP = {
    "The", "A", "An", "At", "To", "From", "In", "Of", "And", "Or", "Was", "Were", "Is", "Patient",
    "Pt", "He", "She", "Admitted", "Transferred", "Discharged", "Seen", "Presented", "Followed",
    "Outside", "Local", "Our", "This", "That", "Your", "Facility", "Footer", "Header", "Information",
    "Referred", "Shifted", "Brought", "Treated", "Visited", "Consulted",
}
# Name words that only describe a TYPE of facility. "Trauma Center" or "Medical Center" is not an
# identifier; "Medicover Hospitals" or "Apollo Hospitals" is (v2: replaces the >= 3 words rule).
ORG_GENERIC = {
    "trauma", "cancer", "heart", "cardiac", "eye", "dental", "medical", "general", "community", "children",
    "childrens", "child", "women", "womens", "maternity", "rehabilitation", "rehab", "psychiatric", "mental",
    "surgical", "surgery", "emergency", "urgent", "primary", "care", "day", "diagnostic", "imaging", "dialysis",
    "wound", "sleep", "pain", "fertility", "teaching", "private", "government", "govt", "district", "regional",
    "local", "outside", "referring", "same", "other", "another", "specialty", "speciality", "multispeciality",
    "multispecialty", "multi", "super", "tertiary", "secondary", "university", "academic", "veterans", "military",
    "state", "county", "city", "municipal", "public", "health", "healthcare", "nursing", "skilled", "long", "term",
    "acute", "chronic", "rural", "urban", "transplant", "kidney", "liver", "lung", "neuro", "neurology",
    "orthopedic", "orthopaedic", "spine", "burn", "stroke", "vascular", "oncology", "pediatric", "paediatric",
    "infusion", "endoscopy", "radiology", "laboratory", "blood", "bank", "physical", "therapy", "occupational",
    "speech", "hearing", "vision", "cardiology", "diabetes", "wellness", "family", "internal", "medicine",
    "senior", "assisted", "living", "home", "behavioral", "addiction", "recovery", "treatment", "ambulatory",
    "outpatient", "inpatient", "main", "north", "south", "east", "west", "central", "new", "old", "nearest",
    "nearby", "previous", "prior", "his", "her", "their", "your", "our", "this", "that", "the", "a", "an",
    "primary health", "phc", "chc",
    # v4: generic facility types in Indian public health (a type, not a name)
    "civil", "sub-district", "sub", "divisional", "taluk", "taluka", "tehsil", "mandal", "block", "area", "zonal",
    "cantonment", "railway", "esi", "esic", "cghs", "dots", "uphc", "sdh", "referral", "base", "in", "of", "for",
    "course", "and",
    "cashless", "hospitalisation", "hospitalization", "request", "form",     # v5
}
_GENERIC_ACRONYMS = {"phc", "chc", "uphc", "dots", "esi", "esic", "cghs", "sdh", "icu", "opd", "ipd", "er", "ed",
                     "ct", "mri", "usg", "pet", "xray", "x-ray", "hrct", "ecg", "echo"}
_ORG_SUFFIX = (r"(?:Hospitals?|Medical[ \t]+Cent(?:er|re)s?|Health[ \t]+Cent(?:er|re)s?|Health[ \t]*care|"
               r"Health[ \t]+System|Health[ \t]+City|Medicity|Clinics?|Polyclinic|Nursing[ \t]+Home|Maternity[ \t]+Home|"
               r"Institute(?:[ \t]+(?:of|for)[ \t]+(?-i:[A-Z][A-Za-z&]*)(?:[ \t]+(?:(?:and|&)[ \t]+)?(?-i:[A-Z][A-Za-z&]*)){0,3})?|"
               r"Medical[ \t]+College(?:[ \t]+(?:and|&)[ \t]+Hospitals?)?|Scans|"
               r"Medical[ \t]+Group|Infirmary|Hospice|Laborator(?:y|ies)|Labs|Diagnostics?|Pharmacy|Dispensary|"
               r"Sanatorium|Cent(?:er|re)s?|Associates)")
_ORG_SUFFIX_RE = re.compile(rf"(?i)(?<![A-Za-z]){_ORG_SUFFIX}(?![A-Za-z])")
# v4: generic "<label> No./ID/Number: <value>" identifiers. The label must contain an identifier noun AND an
# identifier marker (or be an ID acronym); the value must contain >= 3 digits and must not be a date.
ID_NOUN = (r"(?:UHID|U\.H\.I\.D|MRN|MRD|MR|CR|IPD?|OPD?|Hosp(?:ital)?|Patient|Pt|Visit|Accession|Acc|Adm(?:n|ission)?|"
           r"Reg(?:n|d|istration)?|MLC|FIR|Claim|Intimation|Policy|Card|Member(?:ship)?|Beneficiary|Insurance|TPA|Lab|"
           r"Sample|Specimen|Barcode|Bill|Receipt|Invoice|Ref(?:erence)?|File|Folio|Case|Episode|Encounter|Account|Acct|"
           r"Order|Requisition|Report|Booking|Appointment|Nikshay|ESIC?|Employee|Emp|Staff|Aadhaa?r|ABHA|PAN|Passport|"
           r"Voter|Licen[cs]e|Certificate|Cert|Unique|Health|Ration|Customer|Client|Subscriber|Unit)")
ID_LABEL = (rf"(?:\b{ID_NOUN}\.?(?:[ \t]*/?[ \t]*{ID_NOUN}\.?){{0,2}}[ \t]*(?:No|Nos|Number|Num|ID|Id|#|Code)\b\.?"
            rf"|\b(?:UHID|MRN|MRD|SID|CRNO)\b)")
ID_VALUE = r"(?=[A-Za-z0-9./-]*\d)[A-Za-z0-9](?:[A-Za-z0-9./-]*[A-Za-z0-9])?"
_DATE_LIKE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$|^\d{1,2}:\d{2}")
HOUSE_NO = (r"(?:#[ \t]?\d{1,5}[A-Z]?|[A-Z]{1,2}-\d{1,5}[A-Z]?|\d{1,4}(?:-\d{1,4}){1,3}(?:/[A-Z0-9]{1,4})?|"
            r"\d{2,4}/\d{1,4}[A-Z]?)")
STREET_WORD = re.compile(
    r"\b(?:Road|Rd|Street|St|Lane|Nagar|Colony|Cross|Main|Layout|Sector|Block|Phase|Apartments?|Residency|Towers?|"
    r"Enclave|Heights|Complex|Society|Marg|Path|Bazaa?r|Chowk|Gali|Mohalla|Puram|Pet|Peth|Ganj|Bagh|Vihar|Park|"
    r"Avenue|Extension|Extn|Circle|Market|Gardens?|Kunj|Niwas|Bhavan|Sadan|Villa|Plaza|Halli|Palli|Pally)\b")
IN_STATES_NAMES = {
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Orissa", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Chandigarh", "Puducherry", "Jammu and Kashmir", "Ladakh",
    "Andaman and Nicobar Islands", "Lakshadweep", "Dadra and Nagar Haveli and Daman and Diu"}
_IN_STATES = ("AP|AR|AS|BR|CG|CH|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL|LA|LD|MH|ML|MN|MP|MZ|NL|OD|OR|PB|PY|RJ|SK|"
              "TG|TN|TR|TS|UA|UK|UP|WB|AN")


@dataclass(frozen=True)
class Rule:
    name: str
    label: str
    pattern: re.Pattern
    group: int | str = 0
    validator: Callable[[re.Match, str], bool] | None = None
    unit_guard: bool = False
    priority: int = 50          # lower wins when two regex matches overlap with equal length
    optional: bool = False      # active only when listed in regex.optional_rules


# ---------------------------------------------------------------- validators
def _valid_md(m_: int, d_: int) -> bool:
    return 1 <= m_ <= 12 and 1 <= d_ <= 31


def _v_iso(m: re.Match, _t: str) -> bool:
    y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
    return 1800 <= y <= 2299 and _valid_md(mo, d)


def _v_numeric(m: re.Match, _t: str) -> bool:
    a, b = int(m.group("a")), int(m.group("b"))
    return _valid_md(a, b) or _valid_md(b, a)


def _v_month_day(m: re.Match, _t: str) -> bool:
    return 1 <= int(m.group("d")) <= 31


def _v_md_ctx(m: re.Match, _t: str) -> bool:
    a, b = (int(x) for x in m.group(1).split("/"))
    return _valid_md(a, b)


def _v_ip(m: re.Match, _t: str) -> bool:
    return all(0 <= int(x) <= 255 for x in m.group(0).split("."))


_VD = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
       [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
       [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
       [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VP = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
       [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
       [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VD[c][_VP[i % 8][int(ch)]]
    return c == 0


def _v_aadhaar(m: re.Match, _t: str) -> bool:
    return verhoeff_valid(re.sub(r"\D", "", m.group(0)))


def luhn_valid(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def _v_card(m: re.Match, _t: str) -> bool:
    digits = re.sub(r"\D", "", m.group(0))
    return 13 <= len(digits) <= 19 and luhn_valid(digits) and len(set(digits)) > 2


def _v_any_date(m: re.Match, _t: str) -> bool:
    """A labelled numeric date is accepted if ANY reading (y-m-d, d-m-y, m-d-y) is a valid date."""
    parts = [int(x) for x in re.findall(r"\d+", m.group(1))]
    if len(parts) != 3:
        return False
    a, b, c = parts
    return ((a >= 1000 and _valid_md(b, c)) or (c >= 1000 or c < 100) and (_valid_md(b, a) or _valid_md(a, b)))


def _v_id_value(m: re.Match, _t: str) -> bool:
    v = m.group(1)
    return sum(ch.isdigit() for ch in v) >= 3 and not _DATE_LIKE.match(v)


_UNIT_YEARS = {"y": 1.0, "m": 1 / 12, "w": 1 / 52, "d": 1 / 365}


def _unit_kind(u: str | None) -> str:
    if not u:
        return "y"
    u = u.lower()
    return "w" if u.startswith("w") else "d" if u.startswith("d") else "m" if u.startswith("m") else "y"


def age_in_years(m: re.Match) -> float:
    d = m.groupdict()
    total = float(d["n"]) * _UNIT_YEARS[_unit_kind(d.get("u"))]
    if d.get("n2"):
        total += float(d["n2"]) * _UNIT_YEARS[_unit_kind(d.get("u2") or "m")]
    return total


_NAME_CONTEXT_BAD = re.compile(r"(?i)(?:medication|drug|test|brand|generic|facility|product|file|user|device|study)\s*$")


def _v_name_field(m: re.Match, text: str) -> bool:
    return not _NAME_CONTEXT_BAD.search(text[max(0, m.start() - 25):m.start()])


# ---------------------------------------------------------------- rules
def _r(name, label, pattern, group=0, validator=None, unit_guard=False, priority=50, flags=0,
       optional=False) -> Rule:
    return Rule(name, label, re.compile(pattern, flags), group, validator, unit_guard, priority, optional)


def build_rules(age_min: int = 90) -> list[Rule]:
    v_age = lambda m, _t: age_in_years(m) >= age_min    # noqa: E731
    AGE_UNIT = r"(?:years?|yrs?|y|months?|mths?|mnths?|mon|mos?|weeks?|wks?|days?|d)"
    return [
        # --- contact / network identifiers (HIPAA #4-#6, #14, #15)
        _r("email", "EMAIL", r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w-])", priority=5),
        _r("url", "OTHER_PHI", r"\b(?:https?://|www\.)[^\s<>\"')\]]+", priority=5),
        _r("ipv4", "OTHER_PHI", r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", validator=_v_ip, priority=10),
        _r("ssn", "ID", r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])", priority=10),
        _r("ssn_ctx", "ID", r"(?i)\b(?:SSN|social\s+security(?:\s+(?:no\.?|number|#))?)\s*[:#]?\s*(\d{9}|\d{3}\s\d{2}\s\d{4})\b",
           group=1, priority=10),
        _r("phone_us", "PHONE",
           r"(?<![\w+])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\w)", priority=15),
        _r("phone_ctx", "PHONE",
           r"(?i)\b(?:phone|ph|tel(?:ephone)?|fax|cell|mobile|mob|pager|contact(?:\s+number)?|whatsapp|"
           r"alt(?:ernate)?(?:\s+(?:no|number|contact))?|emergency\s+(?:no|number|contact\s+no))\b\.?\s*(?:no\.?|number|#)?\s*[:#.]?\s*"
           r"(\+?\d[\d\s().-]{5,}\d)(?![\w-])", group=1, priority=16),
        # India: mobile numbers start with 6-9 (10 digits), optional +91 / 0091 / 0 prefix
        _r("phone_in_mobile", "PHONE", r"(?<![\w+])(?:(?:\+|00)91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?![\d-])", priority=14),
        # India: landlines with STD code, written with +91 or a leading 0 (e.g. 040-23456789)
        _r("phone_in_landline", "PHONE",
           r"(?<![\w+])(?:(?:\+|00)91[\s-]?|0)\d{2,4}[\s-](?:\d{6,8}|\d{3,4}[\s-]\d{4})(?![\d-])", priority=14),
        # --- national / health identifiers (HIPAA #7-#11, #18)
        _r("aadhaar", "ID", r"(?<![\d-])(?<!\d[ -])[2-9]\d{3}([\s-]?)\d{4}\1\d{4}(?![\d-])(?![ -]\d)", validator=_v_aadhaar, priority=9),
        _r("aadhaar_masked", "ID", r"(?i)\b(?:aadhaa?r|UID)\b[^\n:]{0,15}[:#]?\s*((?:[X*]{4}[\s-]?){2}\d{4})", group=1, priority=9),
        _r("pan", "ID", r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b", priority=9),
        _r("abha_number", "ID", r"(?<![\d-])\d{2}-\d{4}-\d{4}-\d{4}(?![\d-])", priority=9),
        _r("abha_address", "ID", r"(?i)(?<![\w.])[a-z0-9._]{3,}@(?:abdm|sbx|ndhm)\b", priority=4),
        _r("id_ctx", "ID",
           r"(?i)\b(?:MRN|medical\s+record(?:\s+(?:number|no\.?|#))?|record\s*(?:number|no\.?|#)|"
           r"patient\s*(?:id|number|no\.?|#)|unique\s+id|unit\s*(?:no\.?|number|#)|account\s*(?:number|no\.?|#)|"
           r"acct\.?\s*(?:no\.?|#)?|insurance\s*(?:id|number|no\.?|#)|member\s*(?:id|number|#)|"
           r"policy\s*(?:number|no\.?|#)|subscriber\s*(?:id|#)|group\s*(?:number|no\.?|#)|NPI|DEA|"
           r"license\s*(?:number|no\.?|#)|medicare(?:\s*(?:id|number|no\.?|#))?|medicaid(?:\s*(?:id|number|no\.?|#))?|"
           r"MBI|HICN|a/c\.?\s*(?:no\.?|number)?|(?<![A-Za-z])ID)\s*[:#]?\s*"
           r"((?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{3,})(?![\w-])", group=1, priority=20),
        # India: hospital / insurance / government ID fields; values may contain "/" (IP/2024/0123)
        _r("id_ctx_in", "ID",
           r"(?i)\b(?:UHID|U\.H\.I\.D\.?|IPD?\.?\s*(?:no|number)|OPD?\.?\s*(?:no|number)|"
           r"reg(?:istration)?\.?\s*(?:no|number)|MRD\s*(?:no|number)|CR\s*(?:no|number)|hospital\s*(?:no|number|id)|"
           r"admission\s*(?:no|number)|bill\s*(?:no|number)|claim\s*(?:no|number)|TPA\s*(?:id|no)|card\s*(?:no|number)|"
           r"lab\s*(?:no|id)|sample\s*(?:no|id)|passport(?:\s*(?:no|number))?|voter\s*id|EPIC(?:\s*no)?|DL\s*no|"
           r"driving\s*licen[cs]e(?:\s*no)?|emp(?:loyee)?\.?\s*(?:id|no|code)|staff\s*(?:id|no)|serial\s*(?:no|number)|S/N|"
           r"device\s*id|ABHA(?:\s*(?:no|number|id))?|regd\.?\s*no|regn\.?\s*no|ration\s*card(?:\s*no)?|"
           r"ESIC?(?:\s*(?:no|number|ip\s*no))?|CGHS(?:\s*(?:card|id|no|ben(?:eficiary)?\s*id))?|ECHS(?:\s*(?:card|no))?|"
           r"PM-?JAY(?:\s*id)?|ayushman(?:\s*bharat)?(?:\s*card)?(?:\s*no)?)"
           r"\.?\s*[:#.\-]?\s*((?=[A-Za-z0-9/-]*\d)[A-Za-z0-9][A-Za-z0-9/-]{2,})(?![\w/-])", group=1, priority=19),
        # India: vehicle registration (HIPAA #12 vehicle identifiers), e.g. TS 09 EA 1234
        _r("vehicle_in", "OTHER_PHI", rf"\b(?:{_IN_STATES})[ -]?\d{{1,2}}[ -]?[A-Z]{{1,3}}[ -]?\d{{4}}\b", priority=12),
        # --- dates
        _r("date_iso", "DATE", r"(?<![\d/.-])(?P<y>\d{4})(?P<s>[-/.])(?P<m>\d{1,2})(?P=s)(?P<d>\d{1,2})(?![\d/-]|\.\d)",
           validator=_v_iso, unit_guard=True, priority=25),
        _r("date_numeric", "DATE",
           r"(?<![\d/.-])(?P<a>\d{1,2})(?P<s>[-/.])(?P<b>\d{1,2})(?P=s)(?:\d{4}|\d{2})(?![\d/-]|\.\d)",
           validator=_v_numeric, unit_guard=True, priority=25),
        _r("date_text_mdy", "DATE", rf"(?i)\b{MONTH}\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
           validator=_v_month_day, priority=24),
        _r("date_text_dmy", "DATE", rf"(?i)\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH}\.?,?\s+\d{{4}}\b",
           validator=_v_month_day, priority=24),
        # common in Indian records: 12-Mar-2024, 12-Mar-24, 05/Jan/2023, 12 Mar 24
        _r("date_text_dmy_short", "DATE",
           rf"(?i)\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?[-/. ]{MONTH}\.?[-/., ]+(?:\d{{4}}|\d{{2}})(?![\d/-])",
           validator=_v_month_day, priority=24),
        _r("date_text_my", "DATE", rf"(?i)\b{MONTH}\.?,?\s+(?:of\s+)?(?:19|20|21)\d{{2}}\b", priority=26),
        _r("date_text_md", "DATE", rf"\b{MONTH}\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b",
           validator=_v_month_day, unit_guard=True, priority=27),
        _r("date_md_ctx", "DATE", r"(?i)\b(?:on|dated?|since|from|until|till|by)\s+(\d{1,2}/\d{1,2})(?![\d/])",
           group=1, validator=_v_md_ctx, unit_guard=True, priority=28),
        _r("year_ctx", "DATE",
           r"(?i)\b(?:in|since|from|until|till|during|year|born|circa)\s+((?:18|19|20|21)\d{2})(?![\d.,/-]?\d)",
           group=1, unit_guard=True, priority=29),
        # --- times: 12/24-hour, optional seconds, "a.m."/"AM" (never the sentence full stop)
        _r("time_hm", "TIME",
           rf"(?<![\d:.])(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s?{AMPM})?(?![\d:])",
           unit_guard=True, priority=30),
        _r("time_ampm", "TIME", rf"(?<![\d:.])(?:1[0-2]|0?[1-9])\s?{AMPM}(?![A-Za-z])", priority=31),
        _r("time_dot_ampm", "TIME", rf"(?<![\d:.])(?:1[0-2]|0?[1-9])\.[0-5]\d\s?{AMPM}(?![A-Za-z])", priority=30),
        # 24-hour clock with "hrs": 1430 hrs, 14:30 hrs, 08.00 hrs, 0800h
        _r("time_hrs", "TIME", r"(?<![\d:.])(?:[01]\d|2[0-3])[:.]?[0-5]\d(?::[0-5]\d)?\s?(?:hrs?|hours|h)\b", priority=30),
        # --- age: years / months / weeks / days, filtered by age_min in YEARS (HIPAA: > 89)
        _r("age_years", "AGE", r"(?i)\b(?P<n>\d{1,3})\s*(?:-|\s)?(?:years?|yrs?|y/?o|yo)\b",
           group="n", validator=v_age, priority=35),
        _r("age_old", "AGE",
           rf"(?i)\b(?P<v>(?P<n>\d{{1,3}})\s*[-\s]?(?P<u>months?|mths?|mnths?|mon|mos?|weeks?|wks?|days?))[-\s]old\b",
           group="v", validator=v_age, priority=34),
        _r("age_ctx", "AGE",
           rf"(?i)\b(?:age[d:]?|age\s*/\s*(?:sex|gender))\s*[:\-]?\s*(?:about\s+)?(?P<v>(?P<n>\d{{1,3}}(?:\.\d)?)(?![\d./-])(?:\s*(?P<u>{AGE_UNIT})\b)?"
           rf"(?:\s*(?P<n2>\d{{1,2}})\s*(?P<u2>months?|mths?|m|days?|d)\b)?)",
           group="v", validator=v_age, unit_guard=True, priority=35),
        # "45/M", "45 Y/F", "3 M / Male" (Indian & US registration lines)
        _r("age_sex", "AGE", r"(?i)\b(?<![\d./-])(?P<v>(?P<n>\d{1,3})(?![\d.])\s*(?P<u>y|yrs?|years?|m|mths?|months?|d|days?)?)\s*/\s*(?:M|F|Male|Female)\b",
           group="v", validator=v_age, priority=36),
        # --- names announced by context (value only; titles and credentials excluded)
        _r("name_field", "PERSON",
           r"(?i:\b(?:patient\s+name|pt\.?\s+name|name|attending(?:\s+physician)?|physician|provider|"
           r"doctor(?:\s+name)?|surgeon|referring(?:\s+(?:physician|provider|doctor))?|"
           r"primary\s+care\s+(?:physician|provider)|pcp|(?:electronically\s+)?signed\s+by|dictated\s+by|"
           r"transcribed\s+by|cc|consultant|primary\s+consultant|treating\s+(?:doctor|consultant)|"
           r"admitting\s+(?:doctor|consultant)|referred\s+by|ref\.?\s+by|anaesthetist|anesthetist|resident|"
           r"prepared\s+by|verified\s+by|authori[sz]ed\s+by|guardian|father'?s?\s+name|mother'?s?\s+name|"
           r"husband'?s?\s+name|spouse'?s?\s+name|attendant|informant|next\s+of\s+kin|nok|"
           r"relative(?:'s)?\s+name|emergency\s+contact|patient'?s\s+name|pt'?s\s+name|"
           r"name\s+of\s+(?:the\s+)?(?:patient|doctor|consultant|surgeon|attendant|informant|relative|guardian|"
           r"father|mother|husband|wife|spouse)|signature\s+of\s+(?:the\s+)?(?:doctor|consultant|resident)|"
           r"examined\s+by|seen\s+by|reviewed\s+by|operated\s+by|surgery\s+by|"
           # v4: more Indian form labels (a colon is still required)
           r"(?<![A-Za-z][ \t])(?:patient|pt)|treating\s+doctor|name\s+of\s+(?:the\s+)?treating\s+doctor|under\s+care\s+of|"
           r"obstetrician|pa?ediatrician|radiologist|pathologist|unit\s+head|hod|jr|sr|ref\.?\s+dr\.?|"
           r"[sdwb]/o(?:\s*/\s*[sdwb]/o)*)\s*:[ \t]*)"
           rf"(?:{TITLE_SEQ}[ \t]+)?"
           # middle initials with or without a period; wide gaps (3+ spaces) = next column
           rf"({NAME_WORD}(?:[ \t]{{1,2}}(?:[A-Z]\.|[A-Z](?=[ \t]{{1,2}}[A-Z])|{NAME_WORD})){{0,3}})",
           group=1, validator=_v_name_field, priority=40),
        _r("name_title", "PERSON",
           rf"\b{TITLE_SEQ}[ \t]+((?:[A-Z]\.[ \t]*)?{NAME_WORD}(?:[ \t]{{1,2}}{NAME_WORD})?)",
           group=1, priority=41),
        _r("name_credential", "PERSON",
           rf"\b({NAME_WORD}(?:[ \t]+[A-Z]\.)?[ \t]+{NAME_WORD}),?[ \t]+\(?{CRED_SUFFIX}\b",
           group=1, priority=42),
        # India: relatives named in registration lines. Only UPPER-CASE S/O, D/O, B/O directly after a name
        # ("Ramesh Kumar S/O Suresh Kumar") or the long forms ("son of", "wife of" ...). Lower-case s/o, w/o,
        # h/o, c/o and upper-case W/O are clinical shorthand (suggestive of, without, history of,
        # complains of; "CT W/O CONTRAST") and are ignored.
        _r("name_relation", "PERSON",
           rf"(?:(?<=[A-Za-z])[ \t]+(?:S/O|D/O|B/O)|\b(?i:son|daughter|wife|husband|baby)[ \t]+of)"
           rf"\.?[ \t]*:?[ \t]*(?:(?:{TITLES}|Late)\.?[ \t]+)?({NAME_WORD}(?:[ \t]{{1,2}}(?:{NAME_WORD}|[A-Z]\.|[A-Z](?![a-z]))){{0,3}})",
           group=1, priority=40),
        # ... and the person named BEFORE an upper-case relation marker ("Ramesh Kumar S/O ...")
        _r("name_before_relation", "PERSON",
           rf"((?:{TITLES}\.?[ \t]+)?{NAME_WORD}(?:[ \t]{{1,2}}(?:{NAME_WORD}|[A-Z]\.)){{0,3}})[ \t]+(?:S/O|D/O|B/O)\b",
           group=1, priority=40),
        # --- organisations and addresses
        _r("organization", "ORGANIZATION",
           rf"\b((?:(?:St\.?|ST\.?|Saint|[A-Z][a-zA-Z&'\-]+)[ \t]+){{1,5}}(?i:{_ORG_SUFFIX})"
           r"(?:[ \t]+(?:INC|Inc\.?|LLC|Ltd\.?|Corp\.?|PLLC|PC|Pvt\.?[ \t]+Ltd\.?))?)(?![A-Za-z])",
           group=1, priority=45),
        _r("street_address", "LOCATION",
           r"\b(\d{1,5}[ \t]+(?:[A-Z][a-zA-Z'.\-]*[ \t]+){1,4}"
           r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|"
           r"Circle|Cir|Parkway|Pkwy|Highway|Hwy|Square|Sq|Nagar|Marg|Colony|Layout)\.?"
           r"(?:,?[ \t]+(?:Apt|Suite|Ste|Unit)\.?[ \t]*#?\w+)?"
           r"(?:,[ \t]*[A-Z][a-zA-Z.'\-]+(?:[ \t]+[A-Z][a-zA-Z.'\-]+)*){0,2}"
           r"(?:,?[ \t]+[A-Z]{2})?(?:[ \t]+(?:-[ \t]+)?\d{5,6}(?:-\d{4})?)?)", group=1, priority=46),
        _r("address_zip", "LOCATION",
           r"\b(\d{1,5}[ \t]+[A-Z][a-zA-Z'.\-]*(?:[ \t]+[A-Z][a-zA-Z'.\-]*){0,3}"
           r"(?:,[ \t]*[A-Z][a-zA-Z.'\-]*(?:[ \t]+[A-Z][a-zA-Z.'\-]*){0,3}){1,3}"
           r"(?:[ \t]*-[ \t]*|,?[ \t]+)\d{5,6}(?:-\d{4})?)(?![\d-])", group=1, priority=46, optional=True),
        # India: house / flat / plot numbers and "Address:" / "R/O" lines up to the end of the line
        # ("r/o" is NOT used: in clinical shorthand it means "rule out")
        _r("address_line", "LOCATION",
           r"(?i:\b(?:address|addr|residence)\s*[:\-]|\bresident\s+of\b\s*[:\-]?|"
           r"\b(?:H\.?\s*No|house\s*no|flat\s*no|plot\s*no|door\s*no)\b\.?\s*[:\-]?)"
           r"\s*([#\w][^\n;|\t]{2,120}?)(?=\s*(?:\n|;|\||$|(?i:phone|ph|mob|mobile|tel|contact|email)\b)|\t| {3,})",
           group=1, priority=47),
        _r("pin_ctx", "LOCATION", r"(?i)\b(?:PIN(?:\s*code)?|pincode|postal\s*code|zip(?:\s*code)?)\s*[:\-]?\s*(\d{6}|\d{5}(?:-\d{4})?)\b",
           group=1, priority=44),
        # ================= v3: more Indian-English and general patterns =================
        # dates announced by a label, in any numeric order: DOA/DOD/DOB/DOS/LMP/EDD, "date of surgery" ...
        _r("date_ctx_label", "DATE",
           r"(?i)\b(?:DOA|DOD|DOB|DOS|DOE|DOI|LMP|EDD|D\.O\.[ABDS]\.?|date\s+of\s+(?:birth|admission|discharge|surgery|"
           r"procedure|operation|death|expiry|visit|consultation|delivery|injury|onset|joining|report|sample\s+collection))"
           r"\s*[:\-]?\s*((?:\d{1,4}[-/. ]){2}\d{1,4})(?![\d/-])", group=1, validator=_v_any_date, priority=23),
        # "Mar'24", "March '24", "Jan' 23"
        _r("date_month_apostrophe", "DATE", rf"\b{MONTH}\.?\s?['’]\s?\d{{2}}\b", priority=26),
        # "10 o'clock"
        _r("time_oclock", "TIME", r"(?i)\b(?:1[0-2]|0?[1-9])\s?o['’]\s?clock\b", priority=31),
        # India: mobiles written 4-3-3 ("9848 022 338") and landlines with a bracketed STD code
        _r("phone_in_mobile_433", "PHONE", r"(?<![\w+])(?:(?:\+|00)91[\s-]?)?[6-9]\d{3}[\s-]\d{3}[\s-]\d{3}(?![\d-])", priority=14),
        _r("phone_in_landline_paren", "PHONE",
           r"(?<![\w+])(?:(?:\+|00)91[\s-]?)?\(0?\d{2,4}\)[\s-]?\d{3,4}[\s-]?\d{4}(?![\d-])", priority=14),
        # payment / tax / licence / voter / council identifiers (HIPAA #10, #11, #18)
        _r("upi_id", "ID", r"(?i)(?<![\w.])[a-z0-9._-]{2,}@(?:ok(?:axis|hdfcbank|icici|sbi)|ybl|paytm|upi|apl|ibl|axl|"
           r"icici|sbi|hdfcbank|axisbank|kotak|yesbank|freecharge|jio|airtel|barodampay|pnb)\b", priority=4),
        _r("card_number", "ID", r"(?<![\d-])(?:\d{4}[\s-]?){3}\d{4}(?![\d-])", validator=_v_card, priority=9),
        _r("gstin", "ID", r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b", priority=9),
        _r("driving_licence_in", "ID", rf"\b(?:{_IN_STATES})[ -]?\d{{2}}[ -]?(?:19|20)\d{{2}}[ -]?\d{{7}}\b", priority=9),
        _r("voter_id_epic", "ID", r"\b[A-Z]{3}\d{7}\b", priority=10),
        _r("medical_council_reg", "ID",
           r"\b((?:MCI|NMC|DMC|KMC|MMC|TNMC|APMC|TSMC|WBMC|GMC|PMC|UPMC|RMC|BMC|HMC|OMC|CGMC|JMC|TCMC|KSMC)"
           r"[/ \-]?(?:[A-Z]{1,4}[/ \-])?\d{3,7})\b", group=1, priority=10),
        # device / network identifiers (HIPAA #13, #15) and the new BH-series vehicle plates (#12)
        _r("mac_address", "OTHER_PHI", r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b", priority=10),
        _r("ipv6", "OTHER_PHI", r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b", priority=10),
        _r("vehicle_in_bh", "OTHER_PHI", r"\b\d{2}[ -]?BH[ -]?\d{4}[ -]?[A-Z]{1,2}\b", priority=12),
        # India: landmark addresses ("Opp. SBI Bank, Ameerpet, Hyderabad - 500016") ...
        _r("address_landmark", "LOCATION",
           r"(?i:\b(?:near|opposite|behind|beside|next\s+to)\b|\bopp\b\.?)[ \t]+"
           r"((?:[A-Z][\w.&'\-]*[ \t]+){0,4}[A-Z][\w.&'\-]*,[ \t]*[A-Z][\w.\-]*(?:[ \t]+[A-Z][\w.\-]*){0,2}"
           r"(?:(?:,[ \t]*[A-Z][\w.\-]*(?:[ \t]+[A-Z][\w.\-]*){0,2})+(?:[ \t]*-?[ \t]*[1-9]\d{2}[ ]?\d{3})?"
           r"|[ \t]*-?[ \t]*[1-9]\d{2}[ ]?\d{3}))(?![\d])", group=1, priority=47),
        # ... and rural address units: Vill., P.O., Dist., Tq., Teh./Tehsil, Taluk(a), Mandal
        _r("address_rural", "LOCATION",
           r"\b(?:At[ \t]*/[ \t]*Post|At[ \t]*&[ \t]*Post|Vill\.|Village[ \t]*:|P\.[ \t]?O\.|Post[ \t]*:|Dist\.|District[ \t]*:|Tq\.|Teh\.|Tehsil[ \t]*:?|"
           r"Taluka?[ \t]*:?|Mandal[ \t]*:?)[ \t]*([A-Z][\w\-]*(?:[ \t]+[A-Z][\w\-]*){0,2})", group=1, priority=47),
        # ================= v4 (Experiment 13): generalised from DEV error analysis =================
        # any "<identifier label>: <value>" (UHID No., MR No., Hosp. No., Adm. No., MLC No., Policy No. P/1/2,
        # Claim / Intimation No., Visit No., Accession No., SID ...)
        _r("id_label_generic", "ID", rf"(?i:{ID_LABEL})[ \t*]*[:#.\-]?[ \t*]*({ID_VALUE})(?![\w/-])",
           group=1, validator=_v_id_value, priority=21),
        # combined fields: "UHID / IP No.: AB.123 / IP24123" -> the second value
        _r("id_label_pair", "ID", rf"(?i:{ID_LABEL})[ \t*]*[:#.\-]?[ \t*]*{ID_VALUE}[ \t]*/[ \t]*({ID_VALUE})(?![\w/-])",
           group=1, validator=_v_id_value, priority=21),
        # relatives after lower-case / W/O relation markers, accepted only if no captured word is ordinary
        # vocabulary ("s/o pneumonia", "W/O CONTRAST" are clinical shorthand) - checked in detect()
        _r("name_relation_lc", "PERSON",
           rf"(?<=[A-Za-z.:])[ \t]+(?:[sdw]/o|[SDW]/o|W/O)\.?[ \t]*:?[ \t]*(?:{TITLE_SEQ}[ \t]+)?"
           rf"((?:{NAME_WORD}|[A-Z]\.)(?:[ \t]{{1,2}}(?:{NAME_WORD}|[A-Z]\.)){{0,3}})", group=1, priority=41),
        # unlabelled street addresses that start with a house number ("#634, 14th Cross, Wakad, Pune - 411294");
        # accepted only with a street word or a listed city (detect())
        _r("address_houseno", "LOCATION",
           rf"(?m)(?:^|(?<=[:|][ \t])|(?<=[ \t]{{2}}))({HOUSE_NO},[ \t]*(?:[A-Z0-9][\w.'()&-]*(?:[ \t]+[A-Z0-9(][\w.'()&-]*){{0,3}}"
           rf"[ \t]*,[ \t]*){{0,4}}[A-Z][\w.'()-]*(?:[ \t]+[A-Z0-9(][\w.'()-]*){{0,3}}(?:[ \t]*[-,]?[ \t]*[1-9]\d{{2}}[ ]?\d{{3}})?)",
           group=1, priority=46),
        # PIN code right after a listed city or a state ("Hyderabad - 500034", "Kolkata 700334") - detect()
        _r("pin_after_place", "LOCATION",
           r"\b([A-Z][A-Za-z.]+(?:[ \t]+[A-Z][A-Za-z.]+){0,2})[ \t]*[-,]?[ \t]*([1-9]\d{2}[ ]?\d{3})(?![\d-])",
           group=2, priority=44),
        # location given in a labelled field ("Place: Kodungallur", "City: Nashik", "Native of: ...")
        _r("location_field", "LOCATION",
           r"(?i:\b(?:place|native(?:[ \t]+place)?|village|town|city|district|taluka?|tehsil|mandal|locality|"
           r"home[ \t]*town|native[ \t]+of|resident[ \t]+of|hailing[ \t]+from)\b)[ \t]*:[ \t]*"
           r"([A-Z0-9][\w.'-]*(?:[ \t][A-Z0-9(][\w.'()-]*){0,3})", group=1, priority=45),
        # organisation given in a labelled field ("Employer: Indian Railways", "Name of the Hospital: X") - detect()
        _r("org_field", "ORGANIZATION",
           r"(?i:\b(?:employer|company|organi[sz]ation|hospital(?:[ \t]+name)?|clinic[ \t]+name|lab(?:oratory)?[ \t]+name|"
           r"name[ \t]+of[ \t]+(?:the[ \t]+)?(?:hospital|employer|company|clinic|laboratory|lab))\b)[ \t]*:[ \t]*"
           r"([A-Z0-9][\w.&'-]*(?:[ \t](?:of|and|for|&|[A-Z0-9&(][\w.&'()-]*)){0,6})", group=1, priority=45),
        # employer in narrative ("works at State Bank of India", "employed with Tata Steel") - detect()
        _r("org_employer", "ORGANIZATION",
           r"(?i:\b(?:works?|working|employed|employee)[ \t]+(?:at|in|with|for|by|of)\b)[ \t]+"
           r"([A-Z][\w&.'-]*(?:[ \t]+(?:of|and|&|[A-Z][\w&.'-]*)){0,5})", group=1, priority=45),
        # ================= v5 (Experiment 14): generalised from DEV error analysis (Indian dev2, Gretel validation) ====
        # unlabelled record numbers with a strong identifier shape: letters + >= 5 digits ("MED38763308",
        # "HPBN-96445655", "MRN-881046") or one letter + >= 6 digits ("N6137594"); ICD codes, doses, vehicle numbers
        # and PAN do not have this shape
        _r("id_shape", "ID", r"(?<![\w/.-])((?:[A-Z]{2,6}-?\d{5,}|[A-Z]-?\d{6,}))(?![\w/-])", group=1, priority=22),
        # "Dr. Kumari" / "Dr. Devi": the word after the title is the surname
        _r("name_title_surname", "PERSON", r"\b(?:Dr|DR|Prof|PROF)\.?[ \t]+(Kumari|KUMARI|Devi|DEVI)\b(?![ \t]{1,2}[A-Z][a-z])",
           group=1, priority=41),
        # a relative named after the relation word ("son Harjeet", "wife Kamla Devi") - detect(): no ordinary words
        _r("name_after_relative", "PERSON",
           rf"(?i:\b(?:son|daughter|brother|sister|wife|husband|father|mother|nephew|niece|son-in-law|daughter-in-law|"
           rf"grandson|granddaughter|uncle|aunt|cousin)\b)[ \t,]+({NAME_WORD}(?:[ \t]{{1,2}}{NAME_WORD}){{0,2}})",
           group=1, priority=43),
        # a locality at the start of an address line, before a listed city ("Buxi Bazar, Cuttack - 753403") - detect()
        _r("address_locality_city", "LOCATION",
           r"(?m)(?:^|(?<=\|[ \t])|(?<=:[ \t]))([A-Z][\w.'()-]*(?:[ \t]+[A-Z0-9(][\w.'()-]*){0,2})(?=[ \t]*,[ \t]*[A-Z])",
           group=1, priority=48),
        # "250, Shivaji Path, Near Water Tank, Badambadi": a plain house number needs a street word in the first part
        _r("address_number_street", "LOCATION",
           rf"(?m)(?:^|(?<=[:|][ \t])|(?<=[ \t]{{2}}))(\d{{1,4}}[A-Z]?,[ \t]*[A-Z][\w.'-]*(?:[ \t]+[A-Z0-9][\w.'-]*){{0,3}}[ \t]+"
           rf"(?:Road|Rd|Street|Lane|Nagar|Colony|Cross|Layout|Sector|Marg|Path|Bazaa?r|Chowk|Gali|Mohalla|Puram|Pet|Peth|"
           rf"Ganj|Bagh|Vihar|Avenue|Apartments?|Residency|Towers?|Enclave|Heights)\b"
           rf"(?:[ \t]*,[ \t]*[A-Z0-9][\w.'()&-]*(?:[ \t]+[A-Z0-9(][\w.'()&-]*){{0,3}}){{0,4}}(?:[ \t]*[-,]?[ \t]*[1-9]\d{{2}}[ ]?\d{{3}})?)",
           group=1, priority=46),
        # ---- optional candidates (Experiment 6): active only when listed in regex.optional_rules ----
        _r("year_as_age", "DATE", r"(?i)(?:\bis\s+an?\s+|\bage[d:]?\s*:?\s*)((?:18|19|20|21)\d{2})\b(?![.,/-]\d)",
           group=1, priority=29, optional=True),
        _r("year_as_age_old", "DATE", r"(?i)\b((?:18|19|20|21)\d{2})\s*-?\s*(?:years?|yrs?)\s*-?\s*old\b",
           group=1, priority=29, optional=True),
        _r("phone_intl", "PHONE", r"(?<![\w+])\+\d{7,15}(?!\d)", priority=15, optional=True),
    ]


# ---------------------------------------------------------------- post-processing of captures
def _clean_person(text: str, s: int, e: int) -> tuple[int, int] | None:
    words = [(m.start() + s, m.end() + s, m.group()) for m in re.finditer(r"\S+", text[s:e])]
    while words and (words[-1][2].rstrip(".,") in CREDENTIALS or words[-1][2].rstrip(".,") in NAME_STOP):
        words.pop()
    while words and words[0][2].rstrip(".,") in NAME_STOP:
        words.pop(0)
    if not words:
        return None
    s, e = words[0][0], words[-1][1]
    while e > s and text[e - 1] in ".,;:":
        # keep the period of a trailing initial ("J.") but not sentence punctuation
        if text[e - 1] == "." and e - 2 >= s and text[e - 2].isupper() and (e - 3 < s or not text[e - 3].isalpha()):
            break
        e -= 1
    return (s, e) if e > s else None


_ORG_FORMS = {"INC", "Inc", "Inc.", "LLC", "Ltd", "Ltd.", "Corp", "Corp.", "PLLC", "PC", "Pvt", "Pvt."}


def _clean_org(text: str, s: int, e: int, min_words: int = 2, generic_filter: bool = True,
               common_words: frozenset[str] = frozenset()) -> tuple[int, int] | None:
    """Accept "<name words> <facility suffix>".
    * `min_words`: words excluding legal forms (INC/LLC/Pvt Ltd); 2 = at least one name word.
    * `generic_filter`: reject when every name word only describes a facility TYPE ("Trauma Center",
      "Medical Center", "Primary Health Centre"); an upper-case acronym ("KIMS Hospitals",
      "CARE Hospitals") is never generic."""
    words = [(m.start() + s, m.end() + s, m.group()) for m in re.finditer(r"\S+", text[s:e])]
    while len(words) > 1 and words[0][2] in ORG_LEADING_STOP:
        words.pop(0)
    if not words or len([w for w in words if w[2] not in _ORG_FORMS]) < max(2, min_words):
        return None
    s = words[0][0]
    if generic_filter:
        suf = _ORG_SUFFIX_RE.search(text, s, e)
        name_words = [w for w in words if suf is None or w[1] <= suf.start()]
        if not name_words:
            return None
        all_upper = text[s:e].upper() == text[s:e]

        def generic(w: str) -> bool:
            core = re.sub(r"['’]s$", "", w).strip(".,&").lower()
            # acronyms (KIMS, CARE, AIIMS) are names - but not in an ALL-CAPS heading ("COURSE IN THE HOSPITAL")
            if core in _GENERIC_ACRONYMS:
                return True                                    # "CT Scans", "PHC", "DOTS centre"
            if len(w) > 1 and w.isupper() and len(w) <= 6 and not all_upper:
                return False
            return core in ORG_GENERIC or core in common_words
        saint = any(w[2].rstrip(".").lower() in ("st", "saint") for w in name_words)   # "St. Mary's" is a name
        if not saint and all(generic(w[2]) for w in name_words):
            return None
    if not text[s].isupper():
        return None
    return s, words[-1][1]


def load_gazetteer(path: str) -> list[str]:
    from deid.config import resolve_path
    p = resolve_path(path)
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def gazetteer_rules(gazetteers: dict[str, list[str]]) -> list[Rule]:
    """One rule per category from editable public-name lists (configs/gazetteers/*.txt): whole-word,
    case-sensitive match of each entry as written or in UPPER CASE; longest entries tried first."""
    rules = []
    for label, paths in gazetteers.items():
        names = sorted({n for p in paths for n in load_gazetteer(p)}, key=len, reverse=True)
        if not names:
            continue
        alts = "|".join(re.escape(v).replace(r"\ ", r"[ \t]+") for n in names for v in dict.fromkeys((n, n.upper())))
        rules.append(_r(f"gazetteer_{label.lower()}", label, rf"(?<![\w-])(?:{alts})(?![\w-])", priority=48))
    return rules


_AGE_YEARS_ONLY = re.compile(r"(\d{1,3}(?:\.\d)?)\s*(?:years?|yrs?|y)\Z", re.IGNORECASE)


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s] in " \t\r\n,;:":
        s += 1
    while e > s and text[e - 1] in " \t\r\n,;:":
        e -= 1
    return s, e


class RegexDetector:
    """Runs all enabled rules; overlapping regex matches keep the longest (then highest priority)."""

    source = "regex"

    def __init__(self, age_min: int = 90, disabled_rules: list[str] | tuple = (), normalize_text: bool = True,
                 optional_rules: list[str] | tuple = (), org_min_words: int = 2, org_generic_filter: bool = True,
                 org_common_words: str | None = None, gazetteers: dict[str, list[str]] | None = None):
        on, off = set(optional_rules), set(disabled_rules)
        self.rules = [r for r in build_rules(age_min) if r.name not in off and (not r.optional or r.name in on)]
        self.normalize_text = normalize_text
        self.org_min_words = org_min_words
        self.org_generic_filter = org_generic_filter
        self.rules += [r for r in gazetteer_rules(gazetteers or {}) if r.name not in off]
        self.places = frozenset(n for lab, paths in (gazetteers or {}).items() if lab == "LOCATION"
                                for p in paths for n in load_gazetteer(p)) | frozenset(IN_STATES_NAMES)
        self.org_common_words: frozenset[str] = frozenset()
        if org_common_words:
            from deid.config import resolve_path
            path = resolve_path(org_common_words)
            if path.exists():   # optional: built by scripts/build_common_words.py from unlabelled notes
                self.org_common_words = frozenset(w.strip() for w in path.read_text(encoding="utf-8").split() if w.strip())

    @classmethod
    def from_config(cls, cfg: dict) -> "RegexDetector":
        rc = cfg.get("regex", {}) or {}
        return cls(age_min=int(rc.get("age_min", 90)), disabled_rules=rc.get("disabled_rules") or [],
                   normalize_text=bool(cfg.get("normalize_for_detection", True)),
                   optional_rules=rc.get("optional_rules") or [], org_min_words=int(rc.get("org_min_words", 2)),
                   org_generic_filter=bool(rc.get("org_generic_filter", True)),
                   org_common_words=rc.get("org_common_words"), gazetteers=rc.get("gazetteers"))

    def _vocab(self, w: str) -> bool:
        core = re.sub(r"['’]s$", "", w).strip(".,&()").lower()
        return not core or core in ORG_GENERIC or core in self.org_common_words

    def _accept_v4(self, name: str, m: re.Match, text: str, s: int, e: int) -> bool:
        """Context checks of the v4 rules (all other rules: always True)."""
        if name == "name_relation_lc":     # at least first + last name, none of them ordinary vocabulary
            words = re.findall(r"[A-Za-z]{2,}", text[s:e])
            return bool(self.org_common_words) and len(words) >= 2 and not any(w.lower() in self.org_common_words for w in words)
        if name == "address_houseno":
            v = text[s:e]
            return bool(STREET_WORD.search(v)) or any(re.search(rf"(?<!\w){re.escape(c)}(?!\w)", v) for c in self.places)
        if name == "pin_after_place":
            words = m.group(1).split()
            return any(" ".join(words[i:]) in self.places for i in range(len(words)))
        if name == "org_field":          # explicit label: accept anything but a "no organisation" answer
            v = text[s:e].strip(" .").lower()
            return v not in ("none", "na", "n/a", "nil", "self", "unemployed", "retired", "homemaker", "housewife",
                             "student", "not applicable", "-", "private", "business")
        if name == "org_employer":       # narrative: a proper name (>= 2 capitalised words or a non-dictionary word)
            words = text[s:e].split()
            while words and words[-1].lower() in ("of", "and", "&"):
                words.pop()
            caps = [w for w in words if w[:1].isupper()]
            return bool(words) and (len(caps) >= 2 or not all(self._vocab(w) for w in words))
        if name == "location_field":     # rural labels are handled by address_rural
            return not re.match(r"(?i)(?:vill|village|p\.?[ \t]?o|post|dist|district|tq|teh|tehsil|taluka?|mandal)\b", text[s:e])
        if name == "name_after_relative":
            words = re.findall(r"[A-Za-z]{2,}", text[s:e])
            return bool(self.org_common_words) and bool(words) and not any(w.lower() in self.org_common_words for w in words)
        if name == "address_locality_city":
            v = text[s:e]
            first = v.split()[0].rstrip(".")
            if _ORG_SUFFIX_RE.search(v) or first in NAME_STOP or re.match(rf"{TITLES}\b", v) or \
                    first.lower() in ("department", "dept", "unit", "ward", "division", "section", "consultant") or \
                    all(self._vocab(w) for w in v.split()):
                return False
            nxt = re.match(r"[ \t]*,[ \t]*([A-Z][A-Za-z.]+(?:[ \t]+[A-Z][A-Za-z.]+){0,2})", text[e:e + 60])
            if not nxt:
                return False
            words = nxt.group(1).split()
            return any(" ".join(words[:i]) in self.places for i in range(1, len(words) + 1))
        return True

    def detect(self, text: str) -> list[Entity]:
        nt = normalize(text) if self.normalize_text else None
        work = nt.text if nt else text
        cands: list[tuple[int, int, Rule]] = []
        for rule in self.rules:
            for m in rule.pattern.finditer(work):
                if rule.validator and not rule.validator(m, work):
                    continue
                s, e = _trim(work, *m.span(rule.group))
                if s >= e:
                    continue
                if rule.unit_guard and _UNIT_AFTER.match(work, e):
                    continue
                if rule.label == "AGE":
                    ty = _AGE_YEARS_ONLY.match(work, s, e)
                    if ty:
                        e = s + len(ty.group(1))
                if not self._accept_v4(rule.name, m, work, s, e):
                    continue
                if rule.label == "PERSON" and rule.name != "name_title_surname":
                    se = _clean_person(work, s, e)
                elif rule.name == "organization":   # suffix rule only; gazetteer names are taken as listed
                    se = _clean_org(work, s, e, self.org_min_words, self.org_generic_filter, self.org_common_words)
                else:
                    se = (s, e)
                if se is None:
                    continue
                if rule.name in ("org_field", "org_employer"):   # sentence full stop, not "Ltd." / "Pvt."
                    a, b = se
                    while b > a and work[b - 1] in ".,;" and not re.search(r"\b(?:Ltd|Pvt|Co|Inc|Corp)\.$", work[a:b]):
                        b -= 1
                    tail = re.search(r"(?:[ \t]+(?:of|and|for|&))+$", work[a:b])    # dangling connector
                    if tail:
                        b = a + tail.start()
                    se = (a, b)
                cands.append((se[0], se[1], rule))
        cands.sort(key=lambda c: (-(c[1] - c[0]), c[2].priority, c[0]))
        chosen: list[tuple[int, int, Rule]] = []
        for s, e, rule in cands:
            if all(e <= cs or s >= ce for cs, ce, _ in chosen):
                chosen.append((s, e, rule))
        out = []
        for s, e, rule in sorted(chosen, key=lambda c: c[0]):
            if nt is not None:
                s, e = nt.to_original(s, e)
            out.append(Entity(s, e, rule.label, source=self.source, score=1.0, fine_label=rule.name))
        return out
