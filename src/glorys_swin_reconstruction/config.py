from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping.")
    for section in ("data", "model", "loss", "training"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Configuration requires a mapping section named {section!r}.")
    return cfg


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def require_data_paths(cfg: dict[str, Any]) -> None:
    data = cfg["data"]
    for key in ("surface_path", "label_path", "label_norm_path", "domain_mask_path"):
        value = data.get(key)
        if not value:
            raise ValueError(f"data.{key} is required.")
        if not Path(value).exists():
            raise FileNotFoundError(f"data.{key} does not exist: {value}")
