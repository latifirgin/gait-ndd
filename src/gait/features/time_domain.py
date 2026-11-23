"""K2 – Zaman bölgesi özellikleri.

Girdi: tek pencere tek kanal x ∈ R^L (L = 32 adım) ya da toplu X ∈ R^{N×L×C}.
Toplu sürümler eksen=1 (adım ekseni) boyunca hesaplar → (N, C).

Tanımlar
--------
mean : μ = (1/L) Σ x_i
std  : σ = sqrt( (1/(L−1)) Σ (x_i − μ)² )          (ddof=1, örneklem std)
var  : σ²
rms  : sqrt( (1/L) Σ x_i² )
cov  : 100 · σ / μ   [%]   (μ ≤ 0 ise NaN değil 0 döndür ve belgeye yaz)
asymmetry(L, R) : 100 · |μ_L − μ_R| / (0.5 · (μ_L + μ_R))   [%]   (pencere başına)

Dikkat: CoV, Hausdorff ve ark. çalışmalarında HD ve ALS'yi ayıran en güçlü
belirteçtir; X_filt üzerinde hesaplanırsa yapay olarak düşer → varsayılan X_clean.
"""
from __future__ import annotations

import numpy as np


def window_mean(X: np.ndarray) -> np.ndarray:
    """(N, L, C) → (N, C)"""
    raise NotImplementedError("K2: uygula")


def window_std(X: np.ndarray) -> np.ndarray:
    """(N, L, C) → (N, C), ddof=1"""
    raise NotImplementedError("K2: uygula")


def window_var(X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("K2: uygula")


def window_rms(X: np.ndarray) -> np.ndarray:
    raise NotImplementedError("K2: uygula")


def window_cov(X: np.ndarray) -> np.ndarray:
    """Değişim katsayısı [%]."""
    raise NotImplementedError("K2: uygula")


def asymmetry_index(X: np.ndarray, left: int, right: int) -> np.ndarray:
    """(N, L, C) ve kanal indeksleri → (N,) [%]"""
    raise NotImplementedError("K2: uygula")
