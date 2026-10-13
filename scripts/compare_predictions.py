"""Label-free check: run the CURRENT deployed system on a dataset's texts and compare with saved predictions.
Reference spans are not read, so this does not use the test labels. If nothing differs, the saved scores still hold.

    python scripts/compare_predictions.py --data data/synthetic_in/test.jsonl --pred results/exp12_external/indian_synth_test_v5/predictions.system.jsonl
"""
from __future__ import annotations

import argparse
import json

from deid.config import load_yaml, resolve_path
from deid.fusion.span_merger import FusionConfig, fuse
from deid.pipeline import DeidPipeline
from deid.postprocess import postprocess_final, postprocess_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args(argv)
    saved = {r["note_id"]: {(e["start"], e["end"], e["label"]) for e in r["entities"]}
             for r in map(json.loads, open(resolve_path(args.pred), encoding="utf-8"))}
    cfg = load_yaml("configs/detection.yaml")
    pipe, fcfg = DeidPipeline("selected", deployment=True), FusionConfig.from_config(cfg)
    changed_notes = added = removed = 0
    for line in open(resolve_path(args.data), encoding="utf-8"):
        r = json.loads(line)
        text = r["text"]                       # only the text is used
        ents = postprocess_final(fuse(text, [pipe.regex.detect(text), postprocess_model(pipe.model.detect(text), text, cfg)],
                                      fcfg), text, cfg)
        now = {(e.start, e.end, e.label) for e in ents}
        if now != saved[r["note_id"]]:
            changed_notes += 1
            added += len(now - saved[r["note_id"]])
            removed += len(saved[r["note_id"]] - now)
    print(json.dumps({"notes": len(saved), "notes_with_changed_predictions": changed_notes,
                      "spans_added": added, "spans_removed": removed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
