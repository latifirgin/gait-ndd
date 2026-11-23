"""Gerçekçi parametrelerle sentetik gaitndd benzeri veri üretimi.

YALNIZCA geliştirme/test içindir (ağ yoksa boru hattı çalışsın diye). Rapordaki
hiçbir sonuç sentetik veriyle verilmez; windows.npz içindeki ``source`` alanı
bunu kayıt altına alır.

Model
-----
Sol adım süresi e_t için AR(1):  e_t = a·e_{t-1} + sqrt(1-a²)·z_t ,  z_t ~ N(0,1)
Sağ adım innovasyonu soldakiyle ρ korelasyonlu. Adım süresi T = μ_s (1 + CoV·e).
Salınım/duruş/çift destek, adım süresinin yüzdesi olarak üretilir; koridor dönüşü
benzeri aykırı adımlar tüm süreleri bir çarpanla uzatır.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gait.data.loader import write_ts


def _ar1(n: int, a: float, innovations: np.ndarray) -> np.ndarray:
    e = np.empty(n)
    e[0] = innovations[0]
    scale = np.sqrt(1 - a**2)
    for t in range(1, n):
        e[t] = a * e[t - 1] + scale * innovations[t]
    return e


def generate_subject(
    params: Mapping[str, float], syn: Mapping[str, Any], time_column: str,
    channels: Sequence[str], rng: np.random.Generator,
) -> pd.DataFrame:
    mu = params["stride_mean"] * (1 + syn["between_subject_sd"] * rng.standard_normal())
    n = int(syn["duration_s"] / mu)
    rho, a = syn["lr_correlation"], syn["ar_coef"]
    z_l = rng.standard_normal(n)
    z_r = rho * z_l + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    stride = {
        "l": mu * (1 + params["stride_cov"] * _ar1(n, a, z_l)),
        "r": mu * (1 + params["stride_cov"] * _ar1(n, a, z_r)),
    }
    out: dict[str, np.ndarray] = {}
    for side in ("l", "r"):
        swing_pct = params["swing_pct"] + rng.normal(0, syn["pct_noise_sd"], n)
        out[f"stride_{side}"] = stride[side]
        out[f"swing_pct_{side}"] = swing_pct
        out[f"stance_pct_{side}"] = 100 - swing_pct
    dsp = params["dsupport_pct"] + rng.normal(0, syn["pct_noise_sd"], n)
    out["dsupport_pct"] = dsp

    # koridor dönüşleri: tüm süreler birlikte uzar
    turn = rng.random(n) < syn["turn_outlier_rate"]
    lo, hi = syn["turn_outlier_scale"]
    factor = np.where(turn, rng.uniform(lo, hi, n), 1.0)
    for side in ("l", "r"):
        out[f"stride_{side}"] = out[f"stride_{side}"] * factor
        out[f"swing_{side}"] = out[f"stride_{side}"] * out[f"swing_pct_{side}"] / 100
        out[f"stance_{side}"] = out[f"stride_{side}"] - out[f"swing_{side}"]
    out["dsupport"] = 0.5 * (out["stride_l"] + out["stride_r"]) * dsp / 100

    df = pd.DataFrame({c: out[c] for c in channels})
    df.insert(0, time_column, np.cumsum(out["stride_l"]))
    return df


def generate_dataset(
    raw_dir: str | Path, cfg: Mapping[str, Any], seed: int,
) -> Path:
    """Sentetik .ts dosyalarını raw_dir'e yazar (gerçek veriyle aynı adlandırma)."""
    raw_dir = Path(raw_dir)
    data_cfg, syn = cfg["data"], cfg["synthetic"]
    rng = np.random.default_rng(seed)
    for prefix, n_subj in data_cfg["groups"].items():
        params = syn["classes"][prefix]
        for i in range(1, int(n_subj) + 1):
            df = generate_subject(params, syn, data_cfg["time_column"], data_cfg["channels"], rng)
            write_ts(df, raw_dir / f"{prefix}{i}.ts", data_cfg["time_column"], data_cfg["channels"])
    (raw_dir / "SYNTHETIC").write_text("Bu klasördeki veri sentetiktir; rapora girmez.\n", encoding="utf-8")
    return raw_dir
