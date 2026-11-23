"""Adım serilerinin temizlenmesi.

Sıra (denek başına):
1. İlk ``trim_start_s`` saniyeyi at (yürüyüşe başlama geçiş evresi).
2. Fizyolojik sınır: adım süresi kanallarından biri [min, max] dışındaysa adım geçersiz.
3. Denek içi IQR (k): yalnızca adım süresi kanallarında; sınır
   [Q1 − k·IQR, Q3 + k·IQR]. Q1/Q3, 2. adımdan geçen adımlarla hesaplanır.
4. Geçersiz adımın 12 kanalının TAMAMI NaN yapılır (olay bazlı: dönüş adımı tüm
   parametreleri bozar) ve denek içinde adım indeksine göre doğrusal enterpole edilir.

Neden k=3 ve yalnızca adım süresi? 12 kanala ayrı ayrı k=1,5 uygulamak adımların
~%6'sını siliyordu; hastalığın belirteci olan gerçek adım-adım değişkenliği (özellikle
HD'de) yok ediyordu.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def trim_start(df: pd.DataFrame, time_column: str, trim_s: float) -> pd.DataFrame:
    return df.loc[df[time_column] > trim_s].copy()


def physiological_mask(df: pd.DataFrame, cols: Sequence[str], lo: float, hi: float) -> np.ndarray:
    """True = geçerli adım."""
    vals = df[list(cols)].to_numpy()
    return np.all((vals >= lo) & (vals <= hi), axis=1)


def iqr_bounds(values: np.ndarray, k: float) -> tuple[float, float]:
    q1, q3 = np.nanpercentile(values, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def iqr_mask(df: pd.DataFrame, cols: Sequence[str], k: float, base_mask: np.ndarray | None = None) -> np.ndarray:
    """True = geçerli. Sınırlar yalnızca ``base_mask`` True olan adımlardan hesaplanır."""
    base = np.ones(len(df), dtype=bool) if base_mask is None else base_mask
    valid = np.ones(len(df), dtype=bool)
    for c in cols:
        v = df[c].to_numpy()
        lo, hi = iqr_bounds(v[base], k)
        valid &= (v >= lo) & (v <= hi)
    return valid


def interpolate_invalid(df: pd.DataFrame, channels: Sequence[str], valid: np.ndarray) -> pd.DataFrame:
    """Geçersiz adımların tüm kanallarını NaN yapıp adım indeksine göre doğrusal
    enterpole eder; baş/son uçlar en yakın geçerli değerle doldurulur."""
    out = df.copy()
    cols = list(channels)
    out.loc[~valid, cols] = np.nan
    out[cols] = out[cols].interpolate(method="linear", axis=0, limit_area="inside").ffill().bfill()
    return out


def clean_subject(
    df: pd.DataFrame, pre: Mapping[str, Any], time_column: str, channels: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Tek deneği temizler. Döndürür: (temiz tablo, rapor sözlüğü)."""
    n_raw = len(df)
    d = trim_start(df, time_column, pre["trim_start_s"]).reset_index(drop=True)
    stride_cols = pre["outlier_channels"]
    phys = physiological_mask(d, stride_cols, pre["stride_min_s"], pre["stride_max_s"])
    if phys.sum() == 0:
        raise ValueError("Fizyolojik sınırlar içinde hiç adım kalmadı.")
    iqr = iqr_mask(d, stride_cols, pre["iqr_k"], base_mask=phys)
    valid = phys & iqr
    d = interpolate_invalid(d, channels, valid)
    d["step"] = np.arange(len(d), dtype=np.int64)
    d["was_interpolated"] = ~valid
    report = {
        "n_raw": n_raw,
        "n_trimmed": n_raw - len(d),
        "n_physio_invalid": int((~phys).sum()),
        "n_iqr_invalid": int((phys & ~iqr).sum()),
        "n_after": len(d),
    }
    return d, report


def clean_all(
    steps: pd.DataFrame, pre: Mapping[str, Any], time_column: str, channels: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tüm denekler. Az adımı kalan denekler atılır (rapor 'dropped' sütununda)."""
    frames, reports = [], []
    for sid, part in steps.groupby("subject_id", sort=False):
        label = int(part["label"].iloc[0])
        body = part.drop(columns=["subject_id", "label"])
        clean, rep = clean_subject(body, pre, time_column, channels)
        rep.update(subject_id=sid, label=label, dropped=len(clean) < pre["min_valid_steps"])
        reports.append(rep)
        if not rep["dropped"]:
            clean.insert(0, "label", label)
            clean.insert(0, "subject_id", sid)
            frames.append(clean)
    report = pd.DataFrame(reports)
    report["pct_invalid"] = 100 * (report["n_physio_invalid"] + report["n_iqr_invalid"]) / report["n_after"]
    cols = ["subject_id", "label", "n_raw", "n_trimmed", "n_physio_invalid", "n_iqr_invalid",
            "n_after", "pct_invalid", "dropped"]
    return pd.concat(frames, ignore_index=True), report[cols]
