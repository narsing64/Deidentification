"""Shared post-processing steps: Experiment 6 improvement candidates, identical for every system.

Each step is off by default and enabled in configs/detection.yaml (`postprocess:`) only after it
passed the pre-declared acceptance rule (docs/06_pipeline_changes.md).

* `min_model_score`        (B2) drop MODEL entities whose mean confidence is below a threshold
                                (regex entities are never affected)
* `time_only_date_to_time` (A6) a DATE span that contains only a time expression is a TIME, the same
                                content rule used to normalise the reference
* `strip_person_titles`    (A2) remove a leading Dr./Mr./Mrs./Ms./Miss/Prof. from PERSON spans
* `extend_person_names`    (A1) grow a PERSON span over directly adjacent capitalised name words or
                                initials on the same line (stop words, credentials, section words and
                                other entities are never absorbed; at most 4 name words)
Experiment 13 (v4, decided on DEV data only):
* `drop_non_phi`           drop spans that cannot be PHI by their form: eponyms ("Kernig's sign",
                                "Thomas splint"), degrees/credentials and "(Specialty)" after a degree, TIME
                                spans without a clock time ("18 hrs", "15 minutes"), DATE fractions that are
                                no calendar date ("15/15" GCS), short numbers as OTHER_PHI ("108"), and
                                ORGANIZATION spans made only of facility-type / ordinary words
* `drop_state_names`       drop LOCATION spans that are exactly an Indian state/UT (Safe Harbor allows states)
* `fix_person_initials`    keep the full stop of a final initial ("Anju N."), absorb a directly adjacent
                                single initial ("Ramesh V.", "N Kalaiselvi") and join "Mohd." + name
Experiment 14 (v5, decided on DEV data only):
* `trim_markup`            remove markdown/bracket characters at span edges ("Doe**" -> "Doe")
* `drop_non_phi` (extended) IDs without a digit or shaped like "45Y/F"; 5-digit codes as OTHER_PHI; ward/unit
                                acronyms (MICU); names right after a drug form ("Inj. Lasix") or a specimen
                                ("Blood Urea"); placeholders ("[Blank]")
* `drop_state_names` (extended) two-letter state codes ("GA", "MD"); city-states (Chandigarh, Delhi) are kept
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from functools import lru_cache

from deid.detectors.regex_detector import CREDENTIALS, IN_STATES_NAMES, NAME_STOP, ORG_GENERIC, _ORG_SUFFIX_RE
from deid.schema import Entity

_TIME_ONLY = re.compile(r"^(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?\s?[Mm]\.?)?|\d{1,2}\s?[AaPp]\.?[Mm]\.?)$")
_TITLE = re.compile(r"^(?:(?:Late|Shrimati|Kumari|Miss|MISS|Mrs|MRS|Mr|Ms|Mx|Dr|DR|Prof|PROF|Smt|SMT|Shri|SHRI|Sri|SRI|Kum|"
                    r"Er|Capt|Col|Maj|Brig|Sr|Sister|Staff\s+Nurse)\b\.?\s*(?:\(\s*\w+\.?\s*\)\s*)?)+")
_NAME_TOKEN = re.compile(r"[A-Z][a-zA-Z'\-]+|[A-Z]\.?")
_EXTRA_STOP = {
    "History", "Present", "Illness", "Brief", "Course", "Medications", "Medication", "Physical", "Exam",
    "Social", "Family", "Pertinent", "Results", "Diagnosis", "Diagnoses", "Plan", "Assessment", "Impression",
    "Instructions", "Followup", "Condition", "Disposition", "Procedure", "Major", "Surgical", "Invasive",
    "Medical", "Health", "Healthcare", "Institute", "Inc", "LLC", "Transfer", "Emergency", "ICU", "ED", "ER",
    "Notes", "Note", "Summary", "Report", "Lab", "Labs", "Blood", "CT", "MRI", "BP", "HR", "RR", "Temp",
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "I", "II", "III", "IV", "We", "You", "It", "If", "Per", "Please", "Pt", "Mrs", "Mr", "Ms", "Dr",
}
STOP_WORDS = set(NAME_STOP) | set(CREDENTIALS) | _EXTRA_STOP


def min_model_score(ents: Sequence[Entity], threshold: float | None, regex_source: str = "regex") -> list[Entity]:
    if threshold is None:
        return list(ents)
    return [e for e in ents if e.source == regex_source or e.score is None or e.score >= threshold]


def time_only_date_to_time(ents: Sequence[Entity], text: str) -> list[Entity]:
    out = []
    for e in ents:
        if e.label == "DATE" and _TIME_ONLY.match(text[e.start:e.end].strip()):
            e = e.with_(label="TIME", fine_label=f"{e.fine_label}->TIME")
        out.append(e)
    return out


def strip_person_titles(ents: Sequence[Entity], text: str) -> list[Entity]:
    out = []
    for e in ents:
        if e.label == "PERSON":
            m = _TITLE.match(text[e.start:e.end])
            if m and e.start + m.end() < e.end:
                e = e.with_(start=e.start + m.end())
            elif m and text[e.start + m.end():e.end].strip(" .") == "" and \
                    not re.fullmatch(r"(?i)kum(?:ari)?\.?", text[e.start:e.end].strip()):
                continue                                      # v4: the span is only a title ("DR")
        out.append(e)
    return out


def _is_name_token(tok: str) -> bool:
    core = tok.rstrip(".")
    return bool(_NAME_TOKEN.fullmatch(tok)) and core not in STOP_WORDS and not (len(core) > 1 and core.isupper())


def extend_person_names(ents: Sequence[Entity], text: str, max_words: int = 4) -> list[Entity]:
    ents = sorted(ents, key=lambda x: (x.start, x.end))
    out: list[Entity] = []
    for i, e in enumerate(ents):
        if e.label != "PERSON":
            out.append(e)
            continue
        lo = out[-1].end if out else 0
        hi = ents[i + 1].start if i + 1 < len(ents) else len(text)
        s, t = e.start, e.end
        n_words = len(text[s:t].split())
        # right: " Word" or " X." directly after, same line
        while n_words < max_words:
            m = re.match(r"[ ]([A-Z][a-zA-Z'\-]+|[A-Z]\.?)(?=[\s,;:)]|$)", text[t:hi])
            if not m or not _is_name_token(m.group(1)) or text[t + m.end():t + m.end() + 1] == ":":
                break
            t += m.end()
            n_words += 1
        # left: "Word " directly before, same line
        while n_words < max_words:
            m = re.search(r"(?:^|(?<=[\s(]))([A-Z][a-zA-Z'\-]+|[A-Z]\.)[ ]$", text[lo:s])
            if not m or not _is_name_token(m.group(1)):
                break
            s = lo + m.start(1)
            n_words += 1
        out.append(e.with_(start=s, end=t) if (s, t) != (e.start, e.end) else e)
    return out


# ---------------------------------------------------------------- v4 (Experiment 13)
_EPONYM_HEAD = re.compile(
    r"^(?:['’]s)?[ \t]+(?:sign|signs|point|test|catheter|tube|disease|syndrome|splint|viper|palsy|reflex|maneuver|"
    r"manoeuvre|incision|fracture|classification|scale|score|criteria|operation|procedure|phenomenon|lymphoma|ulcer|"
    r"node|nodes|space|line|lines|triangle|law|triad|bag|solution|stain|position|method|technique|repair|approach|"
    r"angle|membrane|gland|duct|canal|hernia|aneurysm|cyst|tumou?r|sarcoma|cast|traction|index|formula|sclerosis|"
    r"chorea|neuralgia|contracture|body|bodies|cells|nodules|spots|murmur|breathing|respiration|ratio|disc|"
    r"phenomenon|dilator|forceps|retractor|speculum|knot|stitch|suture|cannula|needle|mask|pouch|area)\b", re.I)
_DEGREES = {c.lower() for c in CREDENTIALS} | {"d.ortho", "ortho", "ms", "dm", "dnb", "mch", "m.ch", "mbbs", "md",
                                                 "dch", "dgo", "dmrd", "fmas", "frcs", "mrcp", "mrcog", "dpm"}
_DEGREE_BEFORE = re.compile(r"(?:\b(?:MBBS|MD|MS|DM|DNB|MCh|DCH|DGO|DMRD|DPM|D\.Ortho|FRCS|MRCP)\b\.?)[ \t]*\($")
_CLOCK = re.compile(r"\d{1,2}[:.]\d{2}|\d{3,4}[ \t]*(?:hrs?|hours|h)\b|\d[ \t]*[ap]\.?m\b|o['’][ \t]*clock|noon|midnight",
                    re.I)
_FRACTION = re.compile(r"^(\d{1,2})/(\d{1,2})$")
_AGE_SEX = re.compile(r"^\d{1,3}[ \t]*(?:Y|Yrs?|years?|M|D)?[ \t]*/[ \t]*[MFmf]$")
_WARD_UNITS = {"micu", "sicu", "icu", "nicu", "picu", "hdu", "ccu", "iccu", "rccu", "ot", "opd", "ipd", "er", "ed",
               "casualty", "emergency", "ward"}
_NOT_A_NAME_BEFORE = re.compile(r"(?:\b(?:Inj|Tab|Cap|Caps|Syp|Syr|Susp|Oint|Gel|Neb|Inh)\.?|"
                                r"\b(?:Blood|Serum|Urine|Plasma|CSF|Stool|Sputum)|\bS\.)[ \t]+$")
_MARKUP = "*_`[]"                      # not "#": "#802, 7th Cross" is a house number


def trim_markup(ents: Sequence[Entity], text: str) -> list[Entity]:
    out = []
    for e in ents:
        s, t = e.start, e.end
        while s < t and text[s] in _MARKUP:
            s += 1
        while t > s and text[t - 1] in _MARKUP:
            t -= 1
        if t > s:
            out.append(e.with_(start=s, end=t) if (s, t) != (e.start, e.end) else e)
    return out


@lru_cache(maxsize=4)
def _vocabulary(path: str | None) -> frozenset[str]:
    if not path:
        return frozenset()
    from deid.config import resolve_path
    p = resolve_path(path)
    return frozenset(p.read_text(encoding="utf-8").split()) if p.exists() else frozenset()


def drop_non_phi(ents: Sequence[Entity], text: str, vocab: frozenset[str] = frozenset()) -> list[Entity]:
    out = []
    for e in ents:
        v = text[e.start:e.end].strip()
        core = v.strip(".,;:()").lower()
        if e.label in ("PERSON", "LOCATION", "ORGANIZATION") and (core in _DEGREES or _DEGREE_BEFORE.search(text[max(0, e.start - 12):e.start])):
            continue                                          # "MD", "MS (Ortho)" are degrees, not places/people
        if e.label == "PERSON" and len(v.split()) == 1 and _EPONYM_HEAD.match(text[e.end:e.end + 20]):
            continue                                          # "Kernig's sign", "Thomas splint", "Russell's viper"
        if e.label == "TIME" and not _CLOCK.search(v):
            continue                                          # "18 hrs", "15 minutes", "hours"
        if e.label == "DATE":
            f = _FRACTION.match(v)
            if f:
                a, b = int(f.group(1)), int(f.group(2))
                if not ((1 <= a <= 12 and 1 <= b <= 31) or (1 <= b <= 12 and 1 <= a <= 31)):
                    continue                                  # "15/15" (GCS) is no calendar date
        if e.label == "OTHER_PHI" and re.fullmatch(r"\d{1,4}", v):
            continue                                          # "108" (ambulance); 5+ digits may be a record number
        if e.label == "ID" and (_AGE_SEX.match(v) or (not re.search(r"[\d@]", v) and "regex" not in (e.source or ""))):
            continue                                          # model "Mob", "45Y/F"; rule-found UPI/ABHA IDs are kept
        if v.isupper() and core in _WARD_UNITS:              # upper case only: "Ed" can be a name
            continue                                          # "MICU", "ICU" are units, not people or places
        if e.label == "PERSON" and _NOT_A_NAME_BEFORE.search(text[max(0, e.start - 12):e.start]):
            continue                                          # "Inj. Lasix", "Blood Urea"
        if (e.start > 0 and text[e.start - 1] == "[" and text[e.end:e.end + 1] == "]") or v.startswith("["):
            continue                                          # "[Blank]", "[Current Date]" placeholders
        if e.label == "ORGANIZATION" and "regex" not in (e.source or ""):
            words = [re.sub(r"['’]s$", "", w).strip(".,&()").lower() for w in v.split()]
            words = [w for w in words if w]
            suffix = _ORG_SUFFIX_RE.search(v)
            if words and not suffix and all(w in ORG_GENERIC or w in vocab for w in words):
                continue                                      # "ABDOMEN & PELVIS", "COURSE IN THE HOSPITAL"
            if words and suffix:
                head = [w for w in re.sub(r"['’]s", "", v[:suffix.start()]).lower().replace(",", " ").split() if w]
                if head and all(w.strip(".") in ORG_GENERIC for w in head):
                    continue                                  # "Civil Hospital", "Sub-District Hospital"
        out.append(e)
    return out


def drop_state_names(ents: Sequence[Entity], text: str) -> list[Entity]:
    states = {s.lower() for s in IN_STATES_NAMES} - {"chandigarh", "delhi", "puducherry"}   # also cities
    out = []
    for e in ents:
        v = text[e.start:e.end].strip(" ,.")
        if e.label == "LOCATION" and (v.lower() in states or re.fullmatch(r"[A-Z]{2}", v)):
            continue                                          # states and two-letter state codes are not PHI
        out.append(e)
    return out


_INITIAL_AFTER = re.compile(r"^[ ]([A-Z])\.(?=[\s,;:.)|]|$)")
_INITIAL_BEFORE = re.compile(r"(?:^|(?<=[\t:(|])|(?<=[:(|][ ])|(?<=[:(|][ ]{2})|(?<=\.[ ])|(?<=\n))([A-Z])\.?[ ]$")
_ABBREV_FIRST = re.compile(r"^(?:Mohd|Md|Sk|Ch|Smt|Syed|Sri)$", re.IGNORECASE)


def fix_person_initials(ents: Sequence[Entity], text: str) -> list[Entity]:
    ents = sorted(ents, key=lambda x: (x.start, x.end))
    out: list[Entity] = []
    for i, e in enumerate(ents):
        if e.label != "PERSON":
            out.append(e)
            continue
        s, t = e.start, e.end
        nxt = ents[i + 1].start if i + 1 < len(ents) else len(text)
        prv = out[-1].end if out else 0
        if t < len(text) and text[t] == "." and re.search(r"(?:^|[\s.])[A-Z]$", text[s:t]):
            t += 1                                            # "Anju N" + "." -> "Anju N."
        elif re.fullmatch(r"[A-Za-z'\- ]+", text[s:t] or "-") and t < nxt:
            m = _INITIAL_AFTER.match(text[t:nxt])
            if m:
                t += m.end()                                  # "Ramesh" + " V." -> "Ramesh V."
        m = _INITIAL_BEFORE.search(text[prv:s])
        if m and len(text[s:t].split()) <= 3:
            s = prv + m.start(1)                              # "N Kalaiselvi" / "K. Ramesh"
        if out and out[-1].label == "PERSON" and _ABBREV_FIRST.match(text[out[-1].start:out[-1].end]) \
                and re.fullmatch(r"\.?[ ]", text[out[-1].end:s]):
            prev = out.pop()                                  # "Mohd" + ". " + "Arif Shaikh"
            s = prev.start
        out.append(e.with_(start=s, end=t) if (s, t) != (e.start, e.end) else e)
    return out


def postprocess_model(ents: Sequence[Entity], text: str, cfg: dict[str, Any]) -> list[Entity]:
    pp = cfg.get("postprocess", {}) or {}
    return min_model_score(ents, pp.get("min_model_score"))


def postprocess_final(ents: Sequence[Entity], text: str, cfg: dict[str, Any]) -> list[Entity]:
    pp = cfg.get("postprocess", {}) or {}
    out = list(ents)
    if pp.get("time_only_date_to_time"):
        out = time_only_date_to_time(out, text)
    if pp.get("strip_person_titles"):
        out = strip_person_titles(out, text)
    if pp.get("extend_person_names"):
        out = extend_person_names(out, text)
    if pp.get("trim_markup"):
        out = trim_markup(out, text)
    if pp.get("drop_non_phi"):
        out = drop_non_phi(out, text, _vocabulary((cfg.get("regex") or {}).get("org_common_words")))
    if pp.get("drop_state_names"):
        out = drop_state_names(out, text)
    if pp.get("fix_person_initials"):
        out = fix_person_initials(out, text)
    return sorted(out, key=lambda x: (x.start, x.end))
