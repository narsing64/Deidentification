"""Markdown rendering of metric summaries (numbers only, never text)."""
from __future__ import annotations

from typing import Any, Sequence

from deid.evaluation.metrics import fmt


def _ci(summary: dict, kind: str, metric: str) -> str:
    ci = summary.get("bootstrap_ci", {}).get(kind, {}).get(metric)
    if not ci or ci[0] is None:
        return ""
    return f" [{ci[0]:.3f}, {ci[1]:.3f}]"


def headline_table(rows: Sequence[tuple[str, dict[str, Any]]], with_ci: bool = True) -> str:
    out = ["| System | Strict P | Strict R | Strict F1 | Agnostic P | Agnostic R | Agnostic F1 | Wrong-label (exact span) | Overlap R (diag.) | Char coverage (diag.) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        st, ag, ov = s["strict"], s["agnostic"], s["overlap_diagnostics"]
        r_ci = _ci(s, "strict", "recall") if with_ci else ""
        ar_ci = _ci(s, "agnostic", "recall") if with_ci else ""
        out.append(
            f"| {name} | {fmt(st['precision'], 3)} | {fmt(st['recall'], 3)}{r_ci} | {fmt(st['f1'], 3)} | "
            f"{fmt(ag['precision'], 3)} | {fmt(ag['recall'], 3)}{ar_ci} | {fmt(ag['f1'], 3)} | "
            f"{s['wrong_label_exact_span']} | {fmt(ov['gold_overlap_recall'], 3)} | {fmt(ov['gold_char_coverage'], 3)} |")
    return "\n".join(out)


def per_category_table(rows: Sequence[tuple[str, dict[str, Any]]], categories: Sequence[str]) -> str:
    head = "| Category | n (ref) | " + " | ".join(f"{name} P / R / F1" for name, _ in rows) + " |"
    out = [head, "|---|---|" + "---|" * len(rows)]
    for c in categories:
        n_ref = None
        cells = []
        for _, s in rows:
            pc = s["per_category"].get(c)
            if pc is None:
                cells.append("–")
                continue
            n_ref = pc["tp"] + pc["fn"]
            cells.append(f"{fmt(pc['precision'], 2)} / {fmt(pc['recall'], 2)} / {fmt(pc['f1'], 2)}")
        if n_ref is None and all(x == "–" for x in cells):
            continue
        out.append(f"| {c} | {n_ref if n_ref is not None else 0} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def error_table(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    out = ["| System | TP | FP | FN | FN missed | FN boundary | FN wrong-label | FP spurious | FP boundary | FP wrong-label | Ignored preds |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        st, fnb, fpb = s["strict"], s["fn_breakdown_strict"], s["fp_breakdown_strict"]
        out.append(f"| {name} | {st['tp']} | {st['fp']} | {st['fn']} | {fnb.get('missed', 0)} | {fnb.get('boundary', 0)} | "
                   f"{fnb.get('wrong_label', 0)} | {fpb.get('spurious', 0)} | {fpb.get('boundary', 0)} | "
                   f"{fpb.get('wrong_label', 0)} | {s['ignored_predictions']} |")
    return "\n".join(out)


def confusion_table(summary: dict[str, Any], categories: Sequence[str]) -> str:
    conf = summary["confusion_exact_span"]
    preds = sorted({k.split("->")[1] for k in conf})
    golds = [c for c in categories if any(k.startswith(c + "->") for k in conf)]
    if not golds:
        return "_no exact-span matches_"
    out = ["| reference \\ predicted | " + " | ".join(preds) + " |", "|---|" + "---|" * len(preds)]
    for g in golds:
        out.append(f"| {g} | " + " | ".join(str(conf.get(f"{g}->{p}", 0)) for p in preds) + " |")
    return "\n".join(out)


def resource_table(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    out = ["| System | Download (MB) | On disk (MB) | Peak RSS (MB) | Init (s) | Median latency (s/note) | p95 (s/note) | Notes/s | Chars/s |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows:
        lat = r.get("latency", {})
        def mb(x):
            return "n/a" if x is None else f"{x / 2**20:.0f}"
        out.append(f"| {name} | {mb(r.get('download_bytes'))} | {mb(r.get('disk_bytes'))} | {mb(r.get('peak_rss_bytes'))} | "
                   f"{fmt(r.get('init_s'), 2)} | {fmt(lat.get('median_s'), 3)} | {fmt(lat.get('p95_s'), 3)} | "
                   f"{fmt(lat.get('notes_per_s'), 2)} | {fmt(lat.get('chars_per_s'), 0)} |")
    return "\n".join(out)
