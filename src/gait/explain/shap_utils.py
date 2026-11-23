"""K4 – SHAP açıklamaları. Kurulum: pip install -e ".[ml]"

RF  : shap.TreeExplainer (kesin, hızlı). Pipeline'dan 'clf' adımını al; girdi ölçeklenmiş X.
LSTM: shap.GradientExplainer ya da DeepExplainer; arka plan = eğitim fold'undan
      config.shap.background_size pencere. Çıktı (N, L, C) → kanal önemi için L ekseninde
      |SHAP| ortalaması.

Doktor için yorum: SHAP değeri, o özelliğin tahmini "ortalama hastadan" o sınıfa doğru ne
kadar ittiğidir (olasılık değil, model çıktısı ölçeğinde). Arka plan ve test kümesi
aynı fold'dan karıştırılmamalı.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def shap_values_tree(pipe, X: pd.DataFrame, background: pd.DataFrame | None = None) -> np.ndarray:
    """(N, F, K) SHAP değerleri (K = sınıf)."""
    raise NotImplementedError("K4: uygula")


def top_features(shap_row: np.ndarray, feature_names: list[str], class_idx: int, k: int) -> pd.DataFrame:
    """Tek bir örnek için seçilen sınıfa en çok katkı veren k özellik → [feature, shap]."""
    raise NotImplementedError("K4: uygula")


def plot_summary(shap_values: np.ndarray, X: pd.DataFrame, class_names: list[str], path: Any,
                 max_display: int) -> None:
    """Sınıf başına beeswarm/bar grafikleri, yüksek çözünürlük."""
    raise NotImplementedError("K4: uygula")
