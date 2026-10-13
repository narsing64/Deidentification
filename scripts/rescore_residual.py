"""Re-compute residual PHI for SAVED system predictions with the current leftover classifier, so a before/after
comparison uses one measurement (Experiment 13). Offsets only are read; counts are printed.

    python scripts/rescore_residual.py --data data/synthetic_in/test.jsonl --pred results/exp12_external/indian_synth_test/predictions.system.jsonl
"""
from __future__ import annotations

import argparse
import json
import secrets
from collections import Counter

from deid.config import load_yaml, resolve_path
from deid.policies.transform import PolicyEngine
from deid.schema import Entity
from deid.validation.residual import reference_residual


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args(argv)
    preds = {r["note_id"]: [Entity.from_dict(e) for e in r["entities"]]
             for r in map(json.loads, open(resolve_path(args.pred), encoding="utf-8"))}
    engine = PolicyEngine(load_yaml("configs/deid_policy.yaml"), key=secrets.token_bytes(32))
    tot = Counter()
    for line in open(resolve_path(args.data), encoding="utf-8"):
        r = json.loads(line)
        gold = [Entity(s["start"], s["end"], s["label"]) for s in r["spans"]]
        res = engine.transform(r["text"], sorted(preds[r["note_id"]], key=lambda e: (e.start, e.end)), r["note_id"])
        for c in reference_residual(gold, res, r["text"])["by_category"].values():
            tot.update(c)
    n = tot["spans"]
    print(json.dumps({"spans": n, "fully_protected": tot["fully_protected"],
                      "identifying_left": tot["partially_exposed_identifying"],
                      "nonidentifying_left": tot["partially_exposed_nonidentifying"],
                      "fully_exposed": tot["fully_exposed"],
                      "rates": {k: round(100 * tot[v] / n, 2) for k, v in (("protected", "fully_protected"),
                                ("identifying_left", "partially_exposed_identifying"), ("fully_exposed", "fully_exposed"))}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
