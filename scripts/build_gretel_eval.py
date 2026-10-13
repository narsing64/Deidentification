"""Convert the healthcare documents of the Gretel synthetic PII test split into this project's evaluation format.

Input : data/external/gretel_pii_en_v1/test.jsonl   (scripts/fetch_gretel_test.py)
Output: data/external/gretel_pii_en_v1/healthcare_eval.jsonl + conversion_report.json

Gretel lists entities as strings, without offsets. Every verbatim occurrence of an entity string is
annotated. Longer entities win over shorter ones they contain, and a first name directly followed by a
last name (one space) becomes one PERSON span, as in our reference convention. Entity types are mapped to
the 10 project categories. Types that are not HIPAA identifiers of a person (state, country, user name,
password, API key, CVV, PIN, bank routing/SWIFT) become *ignore regions*: predictions there are neither
rewarded nor penalised.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter

from deid.config import resolve_path

TYPE_MAP = {
    "first_name": "PERSON", "last_name": "PERSON", "name": "PERSON",
    "date": "DATE", "date_of_birth": "DATE", "date_time": "DATE",
    "time": "TIME",
    "phone_number": "PHONE", "email": "EMAIL",
    "medical_record_number": "ID", "ssn": "ID", "customer_id": "ID", "employee_id": "ID", "account_number": "ID",
    "certificate_license_number": "ID", "health_plan_beneficiary_number": "ID", "national_id": "ID",
    "tax_id": "ID", "unique_identifier": "ID", "credit_card_number": "ID",
    "street_address": "LOCATION", "address": "LOCATION", "city": "LOCATION", "postcode": "LOCATION",
    "coordinate": "LOCATION",
    "company_name": "ORGANIZATION",
    "url": "OTHER_PHI", "ipv4": "OTHER_PHI", "ipv6": "OTHER_PHI", "license_plate": "OTHER_PHI",
    "vehicle_identifier": "OTHER_PHI", "device_identifier": "OTHER_PHI", "biometric_identifier": "OTHER_PHI",
}
IGNORE_TYPES = {"state", "country", "user_name", "password", "api_key", "cvv", "pin", "swift_bic", "bank_routing_number"}
DOMAINS = ("healthcare", "healthcare-administration")


def locate(text: str, ents: list[dict]) -> tuple[list[dict], list[dict], Counter]:
    stats = Counter()
    cands = []          # (start, end, label or None for ignore, source type)
    for e in ents:
        val = e["entity"]
        types = e["types"]
        label = next((TYPE_MAP[t] for t in types if t in TYPE_MAP), None)
        ignore = label is None and any(t in IGNORE_TYPES for t in types)
        if label is None and not ignore:
            stats["unmapped_type:" + ",".join(types)] += 1
            continue
        found = False
        for m in re.finditer(re.escape(val), text):
            # whole-token occurrences only (avoid "Ann" inside "Annual")
            if (m.start() > 0 and text[m.start() - 1].isalnum() and val[0].isalnum()) or \
               (m.end() < len(text) and text[m.end()].isalnum() and val[-1].isalnum()):
                continue
            cands.append((m.start(), m.end(), label, types[0]))
            found = True
        stats["entities"] += 1
        stats["entities_not_found"] += 0 if found else 1
    # longest-first, non-overlapping
    cands.sort(key=lambda c: (-(c[1] - c[0]), c[0]))
    chosen = []
    for c in cands:
        if all(c[1] <= o[0] or o[1] <= c[0] for o in chosen):
            chosen.append(c)
        elif not any(c[0] == o[0] and c[1] == o[1] for o in chosen):
            stats["dropped_overlap"] += 1
    chosen.sort()
    spans, ignore = [], []
    for s, e, lab, _ in chosen:
        if lab is None:
            ignore.append({"start": s, "end": e, "reason": "non_hipaa_type"})
            continue
        if spans and lab == "PERSON" and spans[-1]["label"] == "PERSON" and text[spans[-1]["end"]:s] == " ":
            spans[-1]["end"] = e            # "First" + " " + "Last" -> one name
            stats["person_parts_merged"] += 1
            continue
        spans.append({"start": s, "end": e, "label": lab})
    return spans, ignore, stats


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    split = ap.parse_args(argv).split
    src = resolve_path(f"data/external/gretel_pii_en_v1/{split}.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    out, total = [], Counter()
    for r in rows:
        if r["domain"] not in DOMAINS:
            continue
        ents = ast.literal_eval(r["entities"])
        spans, ignore, st = locate(r["text"], ents)
        total.update(st)
        out.append({"note_id": "GR-" + r["uid"][:12], "split": split, "doc_type": r["document_type"],
                    "region": r["domain"], "text": r["text"], "spans": spans, "ignore": ignore})
    labels = Counter(s["label"] for o in out for s in o["spans"])
    rep = {"source": f"gretelai/gretel-pii-masking-en-v1 ({split} split, Apache-2.0)", "domains": list(DOMAINS),
           "documents": len(out), "spans": sum(labels.values()), "by_label": dict(labels.most_common()),
           "ignore_regions": sum(len(o["ignore"]) for o in out), "conversion": dict(total),
           "mean_chars": round(sum(len(o["text"]) for o in out) / max(1, len(out)))}
    dst = resolve_path("data/external/gretel_pii_en_v1/" + ("healthcare_eval.jsonl" if split == "test"
                                                             else f"healthcare_{split}.jsonl"))
    with open(dst, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    (dst.parent / f"conversion_report.{split}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
