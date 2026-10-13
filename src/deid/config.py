"""YAML configuration loading with project-relative path resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path.name} must contain a mapping at top level")
    return data


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the project root (never machine-specific)."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_labels(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "labels.yaml")
