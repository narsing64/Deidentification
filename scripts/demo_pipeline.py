"""Show every pipeline stage on a SYNTHETIC note (all names, numbers and places are invented).

Stages: regex detections -> model detections -> fused final entities -> transformed text under the
configured policy -> the same note under an alternative policy showing every action -> consistency
across two notes. Uses the deployed model (configs/models.yaml) and a fixed demo key.

    python scripts/demo_pipeline.py
"""
from __future__ import annotations

import copy
import sys

from deid.config import load_yaml
from deid.pipeline import DeidPipeline
from deid.policies.transform import PolicyEngine
from deid.postprocess import postprocess_model

DEMO_KEY = b"demo-key-for-illustration-only-000000"

NOTE = """DISCHARGE SUMMARY
Name: Jane Q. Roe          MRN: 00123456
Admission Date: 2101-03-04   Discharge Date: 2101-03-10
Attending: Dr. Max Mustermann

History of Present Illness:
Ms. Roe is a 67 year old woman transferred from Lakeside General Hospital on 03/04/2101 at 10:30 AM.
Her father, aged 93, lives at 12 Maple Grove, Springfield, New York - 62701.
Contact: (555) 123-4567, jane.roe@example.org. SSN: 123-45-6789.

Labs: 2101-03-05 06:20AM BLOOD WBC-8.5 Hgb-12.1 Plt-245
Medications: Metoprolol 25 mg PO BID; BP 120/80; follow up with Dr. Mustermann on March 20, 2101.
"""

NOTE_2 = "Follow-up visit 2101-03-20. Patient Jane Roe doing well. MRN 00123456."

NOTE_IN = """DISCHARGE SUMMARY - Medicover Hospitals, Hyderabad
Patient: Ramesh Kumar S/O Suresh Kumar      Age/Sex: 92 Y/M
UHID: MH/2024/00123   IP No: 24-56789   ABHA: 12-3456-7890-1234
Address: H.No 12-3-45, Banjara Hills, Hyderabad - 500034
Mob: +91 98765 43210, Landline: 040-23456789
DOA: 12-Mar-2024 at 1430 hrs    DOD: 18/03/2024 10.30 am
Consultant: Dr. Anil Sharma, MD. Referred from KIMS Hospital.
Native: Vill. Rampur, P.O. Sonari, Dist. Jamshedpur; Landmark: Opp. SBI Bank, Ameerpet, Hyderabad - 500016
Examined by: Dr. (Mrs.) Anita Rao, MBBS, DNB (Reg: TSMC/FMR/12345). Earlier treated at AIIMS in Mar'23.
LMP: 05-01-2024. Bill paid via UPI ramesh.k@okaxis. CGHS Ben ID: 12345678.
Findings s/o pneumonia; CT chest w/o contrast. Lt. knee pain. Adv. CBC. Paracetamol 650 mg TID. Review with Dr. Sharma after 1 week.
"""


def show(title: str, ents, text: str) -> None:
    print(f"\n=== {title} ({len(ents)}) ===")
    for e in ents:
        score = "" if e.score is None else f" score={e.score:.2f}"
        print(f"  {e.label:12s} {text[e.start:e.end]!r:42s} source={e.source}{score} [{e.fine_label}]")


def main() -> int:
    pipe = DeidPipeline("selected", key=DEMO_KEY)
    print("Model:", pipe.model_key, "| backend:", pipe.model.backend)

    show("1. Regex detections", pipe.regex.detect(NOTE), NOTE)
    model_raw = pipe.model.detect(NOTE)
    show("2. Model detections (after adapter clean-up and the 0.5 confidence filter)",
         postprocess_model(model_raw, NOTE, pipe.detection_cfg), NOTE)
    out = pipe.process("demo-1", NOTE, patient_id="P-demo")
    show("3. Final entities after fusion + post-processing", out.entities, NOTE)

    print("\n=== 4. De-identified text with the configured policy (configs/deid_policy.yaml) ===\n")
    print(out.text)
    print("Transformations:", ", ".join(f"{r.label}:{r.action}" for r in out.transform.records))
    print("Validator:", out.residual.to_dict()["review_required"], "items need review;",
          "regex residual:", out.residual.to_dict()["potential_residual_regex"])

    alt = copy.deepcopy(load_yaml("configs/deid_policy.yaml"))
    alt["retain_allowed"] = ["AGE"]
    alt["categories"].update({
        "PERSON": {"action": "placeholder"},
        "PHONE": {"action": "mask", "keep_last": 4},
        "ID": {"action": "mask", "keep_last": 2, "mask_char": "#"},
        "ORGANIZATION": {"action": "redact", "replacement": "[REDACTED]"},
        "LOCATION": {"action": "redact", "replacement": "[REDACTED]"},
        "DATE": {"action": "placeholder"},
        "AGE": {"action": "retain"},
    })
    res = PolicyEngine(alt, key=DEMO_KEY).transform(NOTE, out.entities, "demo-1")
    print("\n=== 5. Same note, alternative policy (placeholder / mask / redact / retain) ===\n")
    print(res.text)

    out2 = pipe.process("demo-2", NOTE_2, patient_id="P-demo")
    print("\n=== 6. Consistency: a second note about the same patient ===")
    print("  input :", NOTE_2)
    print("  output:", out2.text)
    print("  MRN -> same pseudonym as in note 1. Dates -> same patient shift (16-day admission -> follow-up")
    print("  interval preserved). Name -> SAME pseudonym as 'Jane Q. Roe' and 'Roe' in note 1 (name-variant linking).")

    out_in = pipe.process("demo-in", NOTE_IN, patient_id="P-demo-in")
    show("7. Indian-format synthetic note: final entities", out_in.entities, NOTE_IN)
    print("\n=== 8. Indian-format note, de-identified ===\n")
    print(out_in.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
