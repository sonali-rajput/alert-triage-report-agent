"""Loader for the YAML config files in config/."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

# Default to the config/ dir next to the repo root; overridable so the
# Docker images can bake configs anywhere.
CONFIG_DIR = Path(os.environ.get("TRIAGE_CONFIG_DIR", Path(__file__).resolve().parent.parent / "config"))


@functools.cache
def load_config(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def priority_matrix() -> dict[str, Any]:
    return load_config("priority_matrix")


def noise_filters() -> dict[str, Any]:
    return load_config("noise_filters")


def masking_patterns() -> dict[str, Any]:
    return load_config("masking_patterns")
