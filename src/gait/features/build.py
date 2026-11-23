"""K2 – features.parquet üretimi.

SÖZLEŞME (K3, K4 ve K5 buna güvenir):
  Sütunlar: subject_id (str), label (int), window_idx (int), ardından
            <kanal>__<özellik> (float64), örn. stride_l__cov, stride_l__spectral_entropy
            ve asimetri için asym__<sol_kanal>__<sağ_kanal>, örn. asym__stride_l__stride_r
  Satırlar: windows.npz ile aynı sıra ve sayı; window_idx = windows_meta.csv'deki window_idx
  NaN/inf yok. Etiketten türetilmiş hiçbir sütun yok.
  Özellikler pencere içinden hesaplanır (denekler/fold'lar arası bilgi kullanılmaz) →
  bu adımda sızıntı riski yoktur; ölçekleme K3'ün Pipeline'ında yapılır.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

ID_COLUMNS = ["subject_id", "label", "window_idx"]
SEP = "__"


def feature_name(channel: str, feature: str) -> str:
    return f"{channel}{SEP}{feature}"


def extract_features(X: np.ndarray, channels: Sequence[str], fcfg: Mapping[str, Any]) -> pd.DataFrame:
    """(N, L, C) → (N, F) yalnızca özellik sütunları (kimlik sütunu yok).

    K5'in demo için tek bir kaydı da bu fonksiyonla dönüştüreceğini unutma.
    """
    raise NotImplementedError("K2: time_domain + freq_domain + asimetri fonksiyonlarını birleştir")


def build_feature_table(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, window_idx: np.ndarray,
    channels: Sequence[str], fcfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Kimlik sütunları + extract_features çıktısı."""
    feats = extract_features(X, channels, fcfg)
    ids = pd.DataFrame({"subject_id": np.asarray(groups).astype(str),
                        "label": np.asarray(y, dtype=np.int64),
                        "window_idx": np.asarray(window_idx, dtype=np.int64)})
    return pd.concat([ids, feats.reset_index(drop=True)], axis=1)


def validate_feature_table(df: pd.DataFrame, n_rows: int | None = None) -> None:
    """Sözleşme denetimi; K3/K4 okurken de çağırır."""
    missing = [c for c in ID_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kimlik sütunları eksik: {missing}")
    feat = [c for c in df.columns if c not in ID_COLUMNS]
    if not feat:
        raise ValueError("Özellik sütunu yok.")
    bad = [c for c in feat if SEP not in c]
    if bad:
        raise ValueError(f"'<kanal>__<özellik>' adlandırmasına uymayan sütunlar: {bad[:5]}")
    vals = df[feat].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("Özelliklerde NaN/inf var.")
    if n_rows is not None and len(df) != n_rows:
        raise ValueError(f"{len(df)} satır var, {n_rows} bekleniyordu.")
    if df["window_idx"].duplicated().any():
        raise ValueError("window_idx tekrar ediyor.")
