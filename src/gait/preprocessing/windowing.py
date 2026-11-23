"""Kayan pencere: L adım, örtüşme o → kaydırma s = round(L·(1−o)).

Denek başına pencere sayısı: ⌊(n − L)/s⌋ + 1 (n ≥ L ise), aksi hâlde 0.
Pencereler asla iki deneği kapsamaz: her denek ayrı işlenir.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def hop_size(length: int, overlap: float) -> int:
    if not 0 <= overlap < 1:
        raise ValueError("overlap [0, 1) aralığında olmalı.")
    hop = int(round(length * (1 - overlap)))
    if hop < 1:
        raise ValueError("Kaydırma en az 1 adım olmalı.")
    return hop


def window_starts(n: int, length: int, hop: int) -> np.ndarray:
    if n < length:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n - length + 1, hop, dtype=np.int64)


def make_windows(
    steps_clean: pd.DataFrame,
    steps_filt: pd.DataFrame,
    channels: Sequence[str],
    time_column: str,
    length: int,
    overlap: float,
) -> dict[str, np.ndarray | pd.DataFrame]:
    """Döndürür: X_clean, X_filt (N, L, C), y (N,), groups (N,), meta (DataFrame)."""
    if not steps_clean[["subject_id", "step"]].equals(steps_filt[["subject_id", "step"]]):
        raise ValueError("steps_clean ve steps_filt aynı satır düzenine sahip olmalı.")
    hop = hop_size(length, overlap)
    cols = list(channels)
    xc, xf, y, g, meta = [], [], [], [], []
    for sid, idx in steps_clean.groupby("subject_id", sort=False).groups.items():
        c = steps_clean.loc[idx]
        f = steps_filt.loc[idx]
        arr_c, arr_f = c[cols].to_numpy(float), f[cols].to_numpy(float)
        t = c[time_column].to_numpy(float)
        label = int(c["label"].iloc[0])
        for start in window_starts(len(c), length, hop):
            end = start + length
            xc.append(arr_c[start:end])
            xf.append(arr_f[start:end])
            y.append(label)
            g.append(sid)
            meta.append((sid, label, int(start), int(end), t[start], t[end - 1]))
    if not xc:
        raise ValueError("Hiç pencere üretilemedi (seriler pencere boyundan kısa).")
    meta_df = pd.DataFrame(meta, columns=["subject_id", "label", "start_step", "end_step", "t_start", "t_end"])
    meta_df.insert(0, "window_idx", np.arange(len(meta_df), dtype=np.int64))
    return {
        "X_clean": np.stack(xc).astype(np.float32),
        "X_filt": np.stack(xf).astype(np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "groups": np.asarray(g, dtype=str),
        "meta": meta_df,
    }
