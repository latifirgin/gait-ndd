"""K2 – Frekans bölgesi özellikleri.

Örnekleme ekseni ADIM indeksidir: frekans birimi "döngü/adım", aralık [0, 0.5].

Ön işlem: x̃ = x − mean(x)  (config.features.detrend = constant). Aksi hâlde DC
bileşeni her pencerede baskın frekansı 0 yapar ve özellik bilgisiz olur.

Güç spektrumu (tek yanlı): P_k = |FFT(x̃)_k|² , k = 0..⌊L/2⌋ , f_k = k/L

dominant_freq : f_{k*}, k* = argmax_{k≥1} P_k        (DC hariç)
spectral_entropy : p_k = P_k / Σ_{j≥1} P_j ,
                   H = −Σ_{k≥1} p_k log2 p_k / log2(K)   ∈ [0, 1]  (K = terim sayısı)
                   Tek bir sinüs → H≈0 ; beyaz gürültü → H≈1. Σ P = 0 (sabit sinyal) → 0.
"""
from __future__ import annotations

import numpy as np


def power_spectrum(X: np.ndarray, detrend: str = "constant") -> tuple[np.ndarray, np.ndarray]:
    """(N, L, C) → (freqs (K,), P (N, K, C))"""
    raise NotImplementedError("K2: uygula (np.fft.rfft / rfftfreq)")


def dominant_frequency(X: np.ndarray, detrend: str = "constant") -> np.ndarray:
    """(N, L, C) → (N, C) [döngü/adım]"""
    raise NotImplementedError("K2: uygula")


def spectral_entropy(X: np.ndarray, detrend: str = "constant") -> np.ndarray:
    """(N, L, C) → (N, C), normalize [0, 1]"""
    raise NotImplementedError("K2: uygula")
