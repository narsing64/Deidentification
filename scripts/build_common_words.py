"""Build the 'ordinary vocabulary' list used by the organisation rule's generic-word filter.

A word that appears in LOWER CASE in at least --min-docs different UNLABELLED notes is ordinary
vocabulary ("brief", "admission", "cardiology"). Brand / place names (Medicover, Apollo, Lakeside)
almost never appear in lower case. No annotations are used. Output (git-ignored, derived from the
data): data/processed/common_lowercase_words.txt

    python scripts/build_common_words.py --min-docs 5
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

from deid.config import load_yaml, resolve_path
from deid.data.loader import load_unlabelled_notes

_LOWER_WORD = re.compile(r"(?<![A-Za-z])[a-z][a-z'\-]+(?![A-Za-z])")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-docs", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    data = load_yaml("configs/benchmark.yaml")["data"]
    notes, _ = load_unlabelled_notes(resolve_path(data["unlabelled_tsv"]))
    df: Counter = Counter()
    for n in notes:
        df.update(set(_LOWER_WORD.findall(n.text)))
    words = sorted(w for w, c in df.items() if c >= args.min_docs)
    out = resolve_path(args.out or data["processed_dir"]) if args.out else resolve_path(data["processed_dir"]) / "common_lowercase_words.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"{len(words)} ordinary-vocabulary words (lower case in >= {args.min_docs} of {len(notes)} unlabelled notes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
