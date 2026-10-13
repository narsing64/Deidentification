"""Hugging Face token-classification adapter producing common `Entity` objects.

Handles, identically for every model:
* Long notes: the whole note is tokenised once (fast tokenizer, character offsets), then cut
  into sliding windows. Each token is predicted from the window where it lies in the central
  "core", so every prediction has `context_tokens` of context on both sides except at document
  edges. Nothing is truncated: every token belongs to exactly one core.
* Special tokens are added per window by the tokenizer (`build_inputs_with_special_tokens`) and
  dropped again before decoding.
* Sub-words: the label of a word is the label of its FIRST sub-token (the OpenMed training
  strategy, also standard for BERT NER). A word's span runs from its first to its last sub-token.
* Tag schemes: BIO, BILOU (obi) and IO (Stanford: bare type labels) are decoded leniently. An
  I-/L- tag that does not continue an entity of the same type starts a new one.
* Labels: model types are mapped to normalised categories via configs/labels.yaml. Types mapped to
  null are dropped. A model type missing from the map is an error at load time.
* Optional rejoin of adjacent pieces (first_name + last_name) for configured categories.
* Detection may run on normalised text; offsets are mapped back to the original text.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from deid.preprocessing.normalize import normalize
from deid.schema import Entity


class ModelLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str | None
    label_schema: str
    tag_scheme: str
    window_tokens: int
    context_tokens: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        return cls(d["key"], d["hf_id"], d.get("revision"), d["label_schema"], d["tag_scheme"].upper(),
                   int(d["window_tokens"]), int(d["context_tokens"]))


# ---------------------------------------------------------------- pure helpers (unit-tested)
def plan_windows(n_tokens: int, window: int, context: int) -> list[tuple[int, int, int, int]]:
    """Return (win_start, win_end, core_start, core_end) token ranges; cores tile [0, n) exactly."""
    if window <= 2 * context:
        raise ValueError("window must be larger than 2 * context")
    if n_tokens <= 0:
        return []
    if n_tokens <= window:
        return [(0, n_tokens, 0, n_tokens)]
    core = window - 2 * context
    out = []
    cs, ce = 0, min(n_tokens, window - context)
    while True:
        ws, we = max(0, cs - context), min(n_tokens, ce + context)
        if we - ws < window:  # use spare capacity at the document edges as extra context
            if ws == 0:
                we = min(n_tokens, window)
            else:
                ws = max(0, we - window)
        out.append((ws, we, cs, ce))
        if ce >= n_tokens:
            return out
        cs, ce = ce, min(n_tokens, ce + core)


def parse_label(label: str, scheme: str) -> tuple[str, str | None]:
    """'B-NAME' -> ('B', 'NAME'); 'O' -> ('O', None); IO scheme: 'NAME' -> ('I', 'NAME')."""
    if label in ("O", "0", "OUTSIDE"):
        return "O", None
    m = re.match(r"^([BILUES])[-_](.+)$", label)
    if m and scheme != "IO":
        prefix = {"E": "L", "S": "U"}.get(m.group(1), m.group(1))
        return prefix, m.group(2)
    return "I", label


def decode_tags(tags: Sequence[tuple[str, str | None]], scheme: str) -> list[tuple[int, int, str]]:
    """Word-level tags -> (first_word, last_word_inclusive, type). Lenient for all schemes."""
    ents: list[list] = []
    cur: list | None = None
    for i, (p, t) in enumerate(tags):
        if p == "O" or t is None:
            cur = None
            continue
        starts_new = (
            cur is None or cur[2] != t or cur[1] != i - 1
            or (scheme in ("BIO", "BILOU") and p in ("B", "U"))
        )
        if starts_new:
            cur = [i, i, t]
            ents.append(cur)
        else:
            cur[1] = i
        if scheme == "BILOU" and p in ("L", "U"):
            cur = None
    return [(a, b, t) for a, b, t in ents]


_GAP_OK = {
    "PERSON": re.compile(r"^[ \t]{0,2}(?:\.[ \t]{0,2})?$"),
    "LOCATION": re.compile(r"^[ \t,.\-]{0,4}$"),
}


def merge_adjacent(ents: list[Entity], text: str, categories: Sequence[str]) -> list[Entity]:
    """Rejoin same-category neighbours that (a) touch (no character in between: pieces of one
    whitespace token, e.g. "03" "/" "04" "/2101"), any category; or (b) are separated only by a tiny
    gap, for the configured categories (first + last name, street + city)."""
    out: list[Entity] = []
    for e in sorted(ents, key=lambda x: (x.start, x.end)):
        if out and e.label == out[-1].label and e.start >= out[-1].end:
            prev = out[-1]
            gap = text[prev.end:e.start]
            pat = _GAP_OK.get(e.label, re.compile(r"^[ \t]{0,2}$"))
            if gap == "" or (e.label in categories and pat.match(gap)):
                scores = [s for s in (prev.score, e.score) if s is not None]
                out[-1] = prev.with_(end=e.end, score=min(scores) if scores else None,
                                     fine_label=prev.fine_label if prev.fine_label == e.fine_label
                                     else f"{prev.fine_label}+{e.fine_label}")
                continue
        out.append(e)
    return out


_LEAD_PUNCT = "([{\"'"
_TRAIL_PUNCT = ")]}\"',;:.!?"


def text_words(text: str) -> list[tuple[int, int]]:
    """Word unit "text": whitespace-delimited chunks; leading ([{"' and trailing )]}"',;:.!? are
    peeled off as separate one-character words. Inner punctuation stays, so "2101-03-04",
    "06:20AM", "J.Roe@x.org" and "123-45-6789" are single words (Experiment 6, change C1)."""
    words: list[tuple[int, int]] = []
    for m in re.finditer(r"\S+", text):
        s, e = m.span()
        while s < e and text[s] in _LEAD_PUNCT:
            words.append((s, s + 1))
            s += 1
        trail = []
        while e > s and text[e - 1] in _TRAIL_PUNCT:
            trail.append((e - 1, e))
            e -= 1
        if s < e:
            words.append((s, e))
        words.extend(reversed(trail))
    return words


def scope_filter(ents: list[Entity], text: str, age_min: int) -> list[Entity]:
    """Shared AGE scope rule (Experiment 6, change C2): a 4-digit year 1800-2299 labelled AGE is a
    DATE; a numeric age below `age_min` is not an identifier under the policy and is dropped."""
    out = []
    for e in ents:
        if e.label == "AGE":
            digits = re.sub(r"\D", "", text[e.start:e.end])
            if len(digits) == 4 and 1800 <= int(digits) <= 2299:
                e = e.with_(label="DATE", fine_label=f"{e.fine_label}->DATE")
            elif digits and len(digits) <= 3 and int(digits) < age_min:
                continue
        out.append(e)
    return out


_KEEP_DOT = {"Dr", "Mr", "Mrs", "Ms", "Jr", "Sr", "St", "Inc", "Co", "Ltd", "Corp", "Prof"}
_PAIRS = {"(": ")", "[": "]", "{": "}"}


def trim_punctuation(text: str, s: int, e: int) -> tuple[int, int]:
    """Drop stray edge punctuation from a model span: separators, quotes, unmatched brackets and
    sentence-final periods (kept after initials / abbreviations such as "J." or "Inc.")."""
    changed = True
    while changed and s < e:
        changed = False
        c = text[s]
        if c in ",;:.!?\"'" or c in ")]}" or (c in _PAIRS and _PAIRS[c] not in text[s:e]):
            s, changed = s + 1, True
            continue
        c = text[e - 1]
        if c in ",;:!?\"'" or c in "([{" or (c in ")]}" and {v: k for k, v in _PAIRS.items()}[c] not in text[s:e]):
            e, changed = e - 1, True
        elif c == ".":
            word = re.findall(r"[A-Za-z]+$", text[s:e - 1])
            if not word or not (len(word[0]) == 1 and word[0].isupper() or word[0] in _KEEP_DOT):
                e, changed = e - 1, True
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


@dataclass
class WordScores:
    """Per-word model output (offsets in the detection text; no text stored): probability of O,
    arg-max label index/probability, best non-O label index/probability."""
    spans: list[tuple[int, int]]
    p_o: list[float]
    am: list[int]
    am_p: list[float]
    nb: list[int]
    nb_p: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {"spans": self.spans, "p_o": self.p_o, "am": self.am, "am_p": self.am_p, "nb": self.nb, "nb_p": self.nb_p}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WordScores":
        return cls([tuple(x) for x in d["spans"]], d["p_o"], d["am"], d["am_p"], d["nb"], d["nb_p"])


def entities_from_word_scores(text: str, ws: WordScores, *, tags: Sequence[tuple[str, str | None]], scheme: str,
                              label_map: dict[str, str | None], o_index: int | None, decision: str = "argmax",
                              o_threshold: float = 0.5, merge_categories: Sequence[str] = (),
                              age_min: int | None = 90, source: str = "model", normalize_text: bool = True) -> list[Entity]:
    """Decision rule + tag decoding + shared adapter post-processing; offsets refer to `text`.
    decision="argmax": label = arg-max. decision="o_threshold": a word is PHI if P(O) < o_threshold,
    labelled with its best non-O label (recall-oriented when o_threshold > 0.5)."""
    nt = normalize(text) if normalize_text else None
    work = nt.text if nt else text
    chosen: list[tuple[tuple[str, str | None], float]] = []
    for i in range(len(ws.spans)):
        if decision == "o_threshold" and o_index is not None:
            k, p = (ws.nb[i], ws.nb_p[i]) if ws.p_o[i] < o_threshold else (o_index, ws.p_o[i])
        else:
            k, p = ws.am[i], ws.am_p[i]
        chosen.append((tags[k], float(p)))
    ents: list[Entity] = []
    for a, b, typ in decode_tags([c[0] for c in chosen], scheme):
        label = label_map.get(typ)
        if label is None:
            continue
        s, e = ws.spans[a][0], ws.spans[b][1]
        if s >= e:
            continue
        score = float(np.mean([chosen[i][1] for i in range(a, b + 1)]))
        ents.append(Entity(s, e, label, source=source, score=round(score, 4), fine_label=typ))
    ents = merge_adjacent(ents, work, merge_categories)
    trimmed = []
    for ent in ents:
        s, e = trim_punctuation(work, ent.start, ent.end)
        if s < e and re.search(r"[^\W_]", work[s:e]):
            trimmed.append(ent.with_(start=s, end=e))
    ents = trimmed
    if age_min is not None:
        ents = scope_filter(ents, work, age_min)
    if nt is not None and not nt.is_identity_mapping:
        ents = [e.with_(start=x, end=y) for e in ents for x, y in [nt.to_original(e.start, e.end)]]
    return ents


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    return x / x.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- detector
class HFTokenClassifier:
    def __init__(self, spec: ModelSpec, label_map: dict[str, str | None], *, cache_dir: str | None = None,
                 threads: int | None = 4, batch_size: int = 8, decision: str = "argmax",
                 o_threshold: float = 0.5, merge_categories: Sequence[str] = ("PERSON", "LOCATION"),
                 normalize_text: bool = True, local_files_only: bool = True,
                 backend: str = "torch", onnx_path: str | None = None,
                 word_unit: str = "text", age_min: int | None = 90):
        """backend: "torch" (default) or "onnx" (ONNX Runtime session from `onnx_path`, e.g. an
        int8 dynamically quantised export produced by scripts/optimize_model.py)."""
        import torch
        from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        if threads:
            torch.set_num_threads(int(threads))
        self.spec = spec
        self.source = spec.key
        self.batch_size = batch_size
        self.decision = decision
        self.o_threshold = o_threshold
        self.merge_categories = tuple(merge_categories)
        self.normalize_text = normalize_text
        if word_unit not in ("text", "tokenizer"):
            raise ValueError("word_unit must be 'text' or 'tokenizer'")
        self.word_unit = word_unit
        self.age_min = age_min
        t0 = time.perf_counter()
        kw = dict(revision=spec.revision, cache_dir=cache_dir, local_files_only=local_files_only)
        self.backend = backend
        self.torch = torch
        self.model = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, use_fast=True, **kw)
            if backend == "torch":
                model, info = AutoModelForTokenClassification.from_pretrained(spec.hf_id, output_loading_info=True, **kw)
                config = model.config
            elif backend == "onnx":
                if not onnx_path:
                    raise ModelLoadError(f"{spec.key}: backend 'onnx' needs onnx_path")
                import onnxruntime as ort
                config = AutoConfig.from_pretrained(spec.hf_id, **kw)
                so = ort.SessionOptions()
                if threads:
                    so.intra_op_num_threads = int(threads)
                self.session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
                self.onnx_path = str(onnx_path)
                self.session_inputs = {i.name for i in self.session.get_inputs()}
                info = {}
            else:
                raise ModelLoadError(f"{spec.key}: unknown backend {backend!r}")
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a documented compatibility failure
            raise ModelLoadError(f"{spec.key}: {type(exc).__name__}: {exc}") from exc
        if not getattr(self.tokenizer, "is_fast", False):
            raise ModelLoadError(f"{spec.key}: a fast tokenizer (character offsets) is required")
        missing = [k for k in info.get("missing_keys", []) if "classifier" in k or "score" in k]
        if missing or info.get("mismatched_keys"):
            raise ModelLoadError(f"{spec.key}: classification head not loaded ({missing or info['mismatched_keys']})")
        self.loading_info = {k: len(v) for k, v in info.items()}
        if backend == "torch":
            self.model = model.eval()
        self.tokenizer.model_max_length = 10**9  # we window ourselves; silence length warnings
        self.n_special = len(self.tokenizer.build_inputs_with_special_tokens([]))
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        id2label = {int(k): v for k, v in config.id2label.items()}
        self.tags = [parse_label(id2label[i], spec.tag_scheme) for i in range(len(id2label))]
        types = {t for _, t in self.tags if t is not None}
        unmapped = sorted(t for t in types if t not in label_map)
        if unmapped:
            raise ModelLoadError(f"{spec.key}: model types missing from label map: {unmapped}")
        self.label_map = label_map
        self.o_index = next((i for i, (p, _) in enumerate(self.tags) if p == "O"), None)
        self.init_seconds = time.perf_counter() - t0

    # -- inference ---------------------------------------------------------------------------
    def _token_probs(self, ids: list[int]) -> np.ndarray:
        n = len(ids)
        inner = self.spec.window_tokens - self.n_special
        ctx = min(self.spec.context_tokens, (inner - 1) // 2)
        windows = plan_windows(n, inner, ctx)
        probs = np.zeros((n, len(self.tags)), dtype=np.float32)
        for b in range(0, len(windows), self.batch_size):
            batch = windows[b:b + self.batch_size]
            seqs = [self.tokenizer.build_inputs_with_special_tokens(ids[ws:we]) for ws, we, _, _ in batch]
            L = max(len(s) for s in seqs)
            input_ids = np.full((len(seqs), L), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(seqs), L), dtype=np.int64)
            for r, s in enumerate(seqs):
                input_ids[r, :len(s)] = s
                mask[r, :len(s)] = 1
            logits = self._forward(input_ids, mask)
            for r, (ws, we, cs, ce) in enumerate(batch):
                lead = self._leading_specials(seqs[r], ids[ws:we])
                win = _softmax(logits[r, lead:lead + (we - ws)].copy())
                probs[cs:ce] = win[cs - ws:ce - ws]
        return probs

    def _forward(self, input_ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            feed = {"input_ids": input_ids, "attention_mask": mask}
            if "token_type_ids" in self.session_inputs:
                feed["token_type_ids"] = np.zeros_like(input_ids)
            return self.session.run(["logits"], feed)[0].astype(np.float32)
        with self.torch.inference_mode():
            out = self.model(input_ids=self.torch.from_numpy(input_ids), attention_mask=self.torch.from_numpy(mask))
        return out.logits.float().numpy()

    def _leading_specials(self, seq: list[int], body: list[int]) -> int:
        for off in range(0, len(seq) - len(body) + 1):
            if seq[off:off + len(body)] == body:
                return off
        raise RuntimeError("could not locate window tokens inside special-token wrapper")

    @staticmethod
    def text_word_ids(offsets: Sequence[tuple[int, int]], words: Sequence[tuple[int, int]],
                      text: str) -> list[int | None]:
        """Assign each token to the text word containing its first NON-WHITESPACE character.
        Whitespace-only tokens (RoBERTa's newline/space tokens) belong to no word: otherwise a
        line-initial word would take its label from the newline token (handled since Exp. 6)."""
        import bisect
        starts = [w[0] for w in words]
        ids: list[int | None] = []
        for s, e in offsets:
            while s < e and text[s].isspace():
                s += 1
            if e <= s:
                ids.append(None)
                continue
            k = bisect.bisect_right(starts, s) - 1
            ids.append(k if k >= 0 and s < words[k][1] else None)
        return ids

    def score_words(self, text: str) -> WordScores | None:
        """Model pass only: per-word scores from the first sub-token of every word (work-text
        offsets). Decoding is separate (`entities_from_word_scores`), so decision rules can be
        evaluated offline from dumped scores without re-running the model."""
        if not text or not text.strip():
            return None
        nt = normalize(text) if self.normalize_text else None
        work = nt.text if nt else text
        enc = self.tokenizer(work, add_special_tokens=False, return_offsets_mapping=True,
                             return_attention_mask=False, return_token_type_ids=False, verbose=False)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        if not ids:
            return None
        probs = self._token_probs(ids)
        if self.word_unit == "text":
            tw = text_words(work)
            word_ids = self.text_word_ids(offsets, tw, work)
        else:
            word_ids = enc.word_ids()
        first_tok: dict[int, int] = {}
        last_tok: dict[int, int] = {}
        for t, w in enumerate(word_ids):
            if w is None:
                continue
            first_tok.setdefault(w, t)
            last_tok[w] = t
        words = sorted(first_tok)
        if self.word_unit == "text":
            spans = [tw[w] for w in words]
        else:
            spans = [(offsets[first_tok[w]][0], offsets[last_tok[w]][1]) for w in words]
        P = probs[[first_tok[w] for w in words]] if words else np.zeros((0, len(self.tags)), np.float32)
        am = P.argmax(axis=1) if len(P) else np.zeros(0, int)
        if self.o_index is not None and len(P):
            p_o = P[:, self.o_index]
            Q = P.copy()
            Q[:, self.o_index] = -1.0
            nb = Q.argmax(axis=1)
        else:
            p_o, nb = np.zeros(len(P)), am
        rows = np.arange(len(P))
        return WordScores(spans=[tuple(map(int, s)) for s in spans], p_o=p_o.round(5).tolist(),
                          am=am.tolist(), am_p=P[rows, am].round(5).tolist() if len(P) else [],
                          nb=nb.tolist(), nb_p=P[rows, nb].round(5).tolist() if len(P) else [])

    def decode_params(self) -> dict[str, Any]:
        return dict(tags=self.tags, scheme=self.spec.tag_scheme, label_map=self.label_map, o_index=self.o_index,
                    decision=self.decision, o_threshold=self.o_threshold, merge_categories=self.merge_categories,
                    age_min=self.age_min, source=self.source, normalize_text=self.normalize_text)

    def detect(self, text: str) -> list[Entity]:
        ws = self.score_words(text)
        if ws is None:
            return []
        return entities_from_word_scores(text, ws, **self.decode_params())

    def describe(self) -> dict[str, Any]:
        n_params = sum(p.numel() for p in self.model.parameters()) if self.model is not None else None
        return {"key": self.spec.key, "hf_id": self.spec.hf_id, "revision": self.spec.revision,
                "params": n_params, "backend": self.backend, "onnx_path": getattr(self, "onnx_path", None), "n_labels": len(self.tags), "tag_scheme": self.spec.tag_scheme,
                "window_tokens": self.spec.window_tokens, "context_tokens": self.spec.context_tokens,
                "word_unit": self.word_unit, "age_min": self.age_min,
                "special_tokens_per_window": self.n_special, "init_seconds": self.init_seconds,
                "loading_info": self.loading_info, "tokenizer": type(self.tokenizer).__name__}


def resolve_cache_dir(configured: str | None) -> str:
    """Explicit hub cache path. huggingface_hub derives it from HF_HOME, whereas transformers 4.x
    would prefer a (deprecated) TRANSFORMERS_CACHE variable pointing elsewhere; passing the path
    explicitly keeps downloads and loading in the same place."""
    if configured:
        return str(configured)
    from huggingface_hub import constants
    return str(constants.HF_HUB_CACHE)


def build_detector(model_key: str, models_cfg: dict[str, Any], labels_cfg: dict[str, Any],
                   detection_cfg: dict[str, Any], deployment: bool = False, **overrides: Any) -> HFTokenClassifier:
    """deployment=True applies the accepted Section-11 settings in models.yaml `deployment:`
    (e.g. ONNX int8 backend, threads) for that model; benchmarks use the plain settings."""
    spec_d = next((m for m in models_cfg["shortlist"] if m["key"] == model_key), None)
    if spec_d is None:
        fb = [f for f in (models_cfg.get("fallbacks") or {}).values() if f["key"] == model_key]
        if not fb:
            raise KeyError(f"unknown model key {model_key}")
        spec_d = fb[0]
    spec = ModelSpec.from_dict(spec_d)
    mc = detection_cfg.get("model", {}) or {}
    kwargs = dict(
        cache_dir=resolve_cache_dir(models_cfg.get("cache_dir")), threads=models_cfg.get("torch_threads", 4),
        batch_size=models_cfg.get("batch_size", 8), decision=mc.get("decision", "argmax"),
        o_threshold=float(mc.get("o_threshold", 0.5)),
        merge_categories=mc.get("merge_adjacent_categories") or (),
        normalize_text=bool(detection_cfg.get("normalize_for_detection", True)),
        word_unit=mc.get("word_unit", "text"),
        age_min=(detection_cfg.get("regex", {}) or {}).get("age_min", 90) if mc.get("age_scope", True) else None,
    )
    if deployment:
        dep = (models_cfg.get("deployment") or {}).get(model_key) or {}
        if dep.get("backend", "torch") == "onnx":
            from deid.config import resolve_path
            path = resolve_path(dep["onnx_path"])
            if not path.exists():
                raise ModelLoadError(f"{model_key}: deployment ONNX file missing ({dep['onnx_path']}); create it with "
                                     f"`python scripts/optimize_model.py --model {model_key} --configs onnx_int8:8:8`")
            kwargs.update(backend="onnx", onnx_path=str(path))
        if dep.get("threads"):
            kwargs["threads"] = int(dep["threads"])
        if dep.get("batch_size"):
            kwargs["batch_size"] = int(dep["batch_size"])
    kwargs.update(overrides)
    return HFTokenClassifier(spec, labels_cfg["models"][spec.label_schema], **kwargs)
