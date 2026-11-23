"""K4 – LSTM (TensorFlow/Keras). Kurulum: pip install -e ".[dl]"

Girdi: windows.npz → X_filt (N, 32, 12) (config.lstm.input), y, groups.

Kurallar
--------
* Dış bölme folds.csv (iter_folds). Erken durdurma için doğrulama kümesi eğitim fold'undan
  DENEK bazında ayrılır (config.lstm.validation_subject_fraction) – pencere bazında değil!
* Kanal normalizasyonu (z-skor) yalnızca eğitim fold'u istatistikleriyle: μ_c, σ_c
  eğitim pencerelerinin (N_tr·L) örnekleri üzerinden; test aynı μ, σ ile dönüştürülür.
* Sınıf ağırlıkları: w_k = N / (K · N_k) (eğitim fold'undan).
* Çıktı katmanı softmax → predict_proba eşdeğeri; predictions_lstm.csv sözleşmesi K3 ile aynı.
* 64 denekte LSTM'in RF'yi geçmesi beklenmez; adil kıyas ve dürüst raporlama hedef.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


class ChannelScaler:
    """Eğitim fold'unda fit edilen kanal bazlı z-skor ölçekleyici (N, L, C)."""

    def fit(self, X: np.ndarray) -> ChannelScaler:
        raise NotImplementedError("K4: uygula")

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("K4: uygula")


def subject_validation_split(groups_train: np.ndarray, y_train: np.ndarray, fraction: float,
                             seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Eğitim fold'u içinde denek bazlı (katmanlı) train/val indeksleri."""
    raise NotImplementedError("K4: uygula")


def build_lstm(input_shape: tuple[int, int], n_classes: int, lcfg: Mapping[str, Any]):
    """Keras modeli: LSTM(units[0], return_sequences=True) → Dropout → LSTM(units[1]) →
    Dropout → Dense(n_classes, softmax). Kayıp: sparse_categorical_crossentropy."""
    raise NotImplementedError("K4: uygula")


def run_cv(windows: Mapping[str, np.ndarray], folds: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """predictions_lstm.csv içeriğini döndürür (make_predictions_frame şeması)."""
    raise NotImplementedError("K4: uygula")
