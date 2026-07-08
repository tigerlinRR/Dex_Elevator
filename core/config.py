"""Lightweight YAML config loading.

Configs live in ``configs/`` and are intentionally plain dicts so that a future
ROS 2 layer can map them onto parameters without a hard dependency here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_cameras() -> dict[str, Any]:
    return load_yaml("cameras.yaml")


def load_pipeline() -> dict[str, Any]:
    return load_yaml("pipeline.yaml")
