"""K3 (+K4) → K5 arayüzü: tek bir kaydın tahmini.

SÖZLEŞME
--------
predict(steps: pd.DataFrame) -> dict:
  {
    "label": str,                         # sınıf adı, örn. "hunt"
    "proba": dict[str, float],            # 4 sınıf, toplamı 1 (denek düzeyi = pencere ortalaması)
    "shap": pd.DataFrame,                 # sütunlar: feature, shap ; |shap| azalan, en fazla k satır
    "n_windows": int,
  }
``steps``: tek deneğin ham .ts tablosu (gait.data.loader.parse_ts çıktısı).
İç akış: clean_subject → filter → make_windows → extract_features → model.predict_proba
→ ortalama → argmax → SHAP (tahmin edilen sınıf için, pencere ortalaması).
Model dosyası: results/models/rf.joblib (fit_final_model ile üretilir).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

RESULT_KEYS = ("label", "proba", "shap", "n_windows")


def load_model(path: str | Path):
    raise NotImplementedError("K3: joblib.load")


def predict(steps: pd.DataFrame, model: Any = None, cfg: dict | None = None, top_k: int = 10) -> dict:
    raise NotImplementedError("K3: uygula (K4 SHAP fonksiyonlarını kullan)")
