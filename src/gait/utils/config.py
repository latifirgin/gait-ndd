"""Yapılandırma ve yol yardımcıları."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """config.yaml'ı okur. ``GAIT_CONFIG`` ortam değişkeni varsayılanı geçersiz kılar."""
    path = Path(path or os.environ.get("GAIT_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path, root: Path = REPO_ROOT) -> Path:
    """Göreli yolları repo köküne göre çözer."""
    p = Path(path)
    return p if p.is_absolute() else root / p


def class_names(cfg: dict[str, Any]) -> list[str]:
    """Sınıf adları; sıra = tamsayı etiket."""
    return list(cfg["data"]["groups"].keys())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
