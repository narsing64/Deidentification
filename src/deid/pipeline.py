"""End-to-end pipeline: note -> (normalise) -> regex + model detection -> fusion -> policy -> validation.

The model is loaded once and reused for every note. Detection, fusion, transformation and validation
are separate components configured by configs/*.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deid.config import load_labels, load_yaml
from deid.detectors.regex_detector import RegexDetector
from deid.fusion.span_merger import FusionConfig, fuse
from deid.postprocess import postprocess_final, postprocess_model
from deid.policies.transform import PolicyEngine, TransformResult
from deid.schema import Entity
from deid.validation.residual import ResidualReport, ResidualValidator


@dataclass
class PipelineOutput:
    note_id: str
    text: str                       # de-identified text
    entities: list[Entity]          # detected spans in ORIGINAL offsets (never serialised with text)
    transform: TransformResult
    residual: ResidualReport


class DeidPipeline:
    def __init__(self, model_key: str | None = "selected", *, use_regex: bool = True, key: bytes | None = None,
                 deployment: bool = True,
                 detection_cfg: dict | None = None, policy_cfg: dict | None = None,
                 models_cfg: dict | None = None, labels_cfg: dict | None = None):
        self.detection_cfg = detection_cfg or load_yaml("configs/detection.yaml")
        self.models_cfg = models_cfg or load_yaml("configs/models.yaml")
        self.labels_cfg = labels_cfg or load_labels()
        self.policy = PolicyEngine(policy_cfg or load_yaml("configs/deid_policy.yaml"), key=key)
        self.regex = RegexDetector.from_config(self.detection_cfg) if use_regex else None
        if model_key == "selected":
            model_key = self.models_cfg.get("selected")
            if not model_key:
                raise ValueError("configs/models.yaml has no `selected` model yet; pass model_key explicitly")
        self.model_key = model_key
        self.model = None
        if model_key:
            from deid.detectors.hf_detector import build_detector
            self.model = build_detector(model_key, self.models_cfg, self.labels_cfg, self.detection_cfg,
                                        deployment=deployment)
        if self.model is None and self.regex is None:
            raise ValueError("at least one detector (regex or model) is required")
        self.fusion_cfg = FusionConfig.from_config(self.detection_cfg)
        self.validator = ResidualValidator(self.regex or RegexDetector.from_config(self.detection_cfg))

    def detect(self, text: str) -> list[Entity]:
        sources = []
        if self.regex is not None:
            sources.append(self.regex.detect(text))
        if self.model is not None:
            sources.append(postprocess_model(self.model.detect(text), text, self.detection_cfg))
        return postprocess_final(fuse(text, sources, self.fusion_cfg), text, self.detection_cfg)

    def process(self, note_id: str, text: str, patient_id: str | None = None) -> PipelineOutput:
        if not isinstance(text, str):
            raise TypeError("note text must be a string")
        ents = self.detect(text) if text.strip() else []
        result = self.policy.transform(text, ents, note_id, patient_id)
        residual = self.validator.validate(text, ents, result)
        return PipelineOutput(note_id, result.text, ents, result, residual)

    def describe(self) -> dict[str, Any]:
        return {"model": self.model_key, "regex": self.regex is not None,
                "ephemeral_key": self.policy.key_is_ephemeral,
                "model_info": self.model.describe() if self.model is not None else None}
