"""Sıfır fazlı Butterworth alçak geçiren filtre.

Örnekleme ekseni zaman değil ADIM İNDEKSİDİR (her adım bir örnek). Normalize
kesim ω_c ∈ (0, 1), Nyquist'e (0,5 döngü/adım) göredir: 0,35 → 0,175 döngü/adım,
yani ~5,7 adımdan kısa periyotlu dalgalanmalar bastırılır.

``filtfilt`` sinyali ileri ve geri süzer: faz gecikmesi sıfır, etkin derece 2×order,
genlik yanıtı |H(ω)|². Uyarı: filtre adım-adım değişkenliği (CoV) düşürür; bu yüzden
hem filtresiz (X_clean) hem filtreli (X_filt) seri saklanır.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def lowpass(x: np.ndarray, order: int, cutoff: float) -> np.ndarray:
    """x: (n,) ya da (n, c); eksen 0 boyunca süzer."""
    if not 0 < cutoff < 1:
        raise ValueError("Normalize kesim (0, 1) aralığında olmalı.")
    b, a = butter(order, cutoff, btype="low")
    padlen = 3 * max(len(a), len(b))
    if x.shape[0] <= padlen:
        raise ValueError(f"Filtre için seri çok kısa: {x.shape[0]} ≤ padlen={padlen}")
    return filtfilt(b, a, x, axis=0)


def filter_subjects(
    steps: pd.DataFrame, channels: Sequence[str], order: int, cutoff: float
) -> pd.DataFrame:
    """Her deneğin kanallarını ayrı ayrı süzer (denekler arası sınırda süzme yok)."""
    out = steps.copy()
    cols = list(channels)
    for _, idx in steps.groupby("subject_id", sort=False).groups.items():
        out.loc[idx, cols] = lowpass(steps.loc[idx, cols].to_numpy(dtype=float), order, cutoff)
    return out
