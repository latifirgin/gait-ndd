"""Temizleme, filtreleme ve pencereleme testleri."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gait.preprocessing.cleaning import (
    clean_all,
    clean_subject,
    interpolate_invalid,
    iqr_mask,
    physiological_mask,
    trim_start,
)
from gait.preprocessing.filtering import filter_subjects, lowpass
from gait.preprocessing.windowing import hop_size, make_windows, window_starts


def _subject(cfg, n=200, seed=0):
    d = cfg["data"]
    rng = np.random.default_rng(seed)
    stride = 1.1 + 0.02 * rng.standard_normal(n)
    df = pd.DataFrame({c: 0.0 for c in d["channels"]}, index=range(n))
    df["stride_l"] = stride
    df["stride_r"] = stride + 0.01 * rng.standard_normal(n)
    for c in d["channels"]:
        if c not in ("stride_l", "stride_r"):
            df[c] = 10 + rng.standard_normal(n)
    df.insert(0, d["time_column"], np.cumsum(stride))
    return df


# ------------------------------------------------------------------ temizleme
def test_trim_start(cfg):
    df = _subject(cfg)
    t = cfg["data"]["time_column"]
    out = trim_start(df, t, 20)
    assert (out[t] > 20).all() and len(out) < len(df)


def test_physiological_mask():
    df = pd.DataFrame({"a": [0.4, 1.0, 3.5, 1.2], "b": [1.0, 1.0, 1.0, 0.49]})
    np.testing.assert_array_equal(physiological_mask(df, ["a", "b"], 0.5, 3.0), [False, True, False, False])


def test_iqr_only_flags_spike():
    v = np.r_[np.full(50, 1.0) + np.linspace(-0.05, 0.05, 50), 2.0]
    df = pd.DataFrame({"a": v})
    mask = iqr_mask(df, ["a"], k=3)
    assert mask[:-1].all() and not mask[-1]


def test_k3_preserves_more_steps_than_k15():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": 1.1 + 0.08 * rng.standard_t(4, 2000)})  # kalın kuyruklu gerçek değişkenlik
    assert iqr_mask(df, ["a"], 3).sum() > iqr_mask(df, ["a"], 1.5).sum()


def test_interpolate_invalid_whole_row_linear(cfg):
    df = pd.DataFrame({"a": [1.0, 9.0, 3.0], "b": [10.0, 99.0, 30.0]})
    out = interpolate_invalid(df, ["a", "b"], np.array([True, False, True]))
    assert out.loc[1, "a"] == pytest.approx(2.0) and out.loc[1, "b"] == pytest.approx(20.0)
    edge = interpolate_invalid(df, ["a", "b"], np.array([False, True, True]))
    assert edge.loc[0, "a"] == pytest.approx(9.0)  # uç: en yakın geçerli değer


def test_turn_step_invalidates_all_channels(cfg):
    pre, d = cfg["preprocessing"], cfg["data"]
    df = _subject(cfg)
    k = 100
    df.loc[k, ["stride_l", "stride_r"]] *= 1.8      # dönüş adımı
    df.loc[k, "swing_l"] = 500.0                   # diğer kanal da bozuk
    clean, rep = clean_subject(df, pre, d["time_column"], d["channels"])
    row = clean[clean["was_interpolated"]]
    assert len(row) == 1 and rep["n_iqr_invalid"] == 1
    assert row["swing_l"].iloc[0] < 20               # bozuk kanal da enterpole edildi
    assert not clean[d["channels"]].isna().any().any()


def test_clean_all_report(cfg, small_steps):
    d = cfg["data"]
    clean, rep = clean_all(small_steps, cfg["preprocessing"], d["time_column"], d["channels"])
    assert rep["subject_id"].nunique() == small_steps["subject_id"].nunique()
    assert (rep["pct_invalid"] < 10).all()            # silinen adım oranı makul
    assert (clean.groupby("subject_id")["step"].apply(lambda s: (s.diff().dropna() == 1).all())).all()


# ------------------------------------------------------------------ filtre
def test_lowpass_zero_phase_and_attenuation():
    n = np.arange(400)
    slow = np.sin(2 * np.pi * 0.02 * n)       # 50 adım periyot (geçmeli)
    fast = np.sin(2 * np.pi * 0.4 * n)        # 2,5 adım periyot (bastırılmalı)
    y = lowpass(slow + fast, order=2, cutoff=0.35)
    core = slice(50, 350)
    assert np.corrcoef(y[core], slow[core])[0, 1] > 0.99   # faz kayması yok
    assert np.std((y - slow)[core]) < 0.1 * np.std(fast)


def test_lowpass_reduces_cov_and_rejects_short():
    rng = np.random.default_rng(0)
    x = 1.1 + 0.05 * rng.standard_normal(300)
    y = lowpass(x, 2, 0.35)
    assert y.std() < x.std() and y.mean() == pytest.approx(x.mean(), rel=1e-3)
    with pytest.raises(ValueError):
        lowpass(x[:5], 2, 0.35)
    with pytest.raises(ValueError):
        lowpass(x, 2, 1.2)


def test_filter_subjects_independent(cfg, small_steps):
    d = cfg["data"]
    clean, _ = clean_all(small_steps, cfg["preprocessing"], d["time_column"], d["channels"])
    f_all = filter_subjects(clean, d["channels"], 2, 0.35)
    one = clean[clean.subject_id == clean.subject_id.iloc[0]]
    f_one = filter_subjects(one, d["channels"], 2, 0.35)
    np.testing.assert_allclose(f_all.loc[one.index, d["channels"]], f_one[d["channels"]])


# ------------------------------------------------------------------ pencere
def test_window_starts_formula():
    assert hop_size(32, 0.5) == 16
    np.testing.assert_array_equal(window_starts(80, 32, 16), [0, 16, 32, 48])
    assert window_starts(31, 32, 16).size == 0
    with pytest.raises(ValueError):
        hop_size(32, 1.0)


def test_make_windows_never_cross_subjects(cfg, small_steps):
    d, w = cfg["data"], cfg["windowing"]
    clean, _ = clean_all(small_steps, cfg["preprocessing"], d["time_column"], d["channels"])
    filt = filter_subjects(clean, d["channels"], 2, 0.35)
    out = make_windows(clean, filt, d["channels"], d["time_column"], w["length"], w["overlap"])
    n_ch = len(d["channels"])
    assert out["X_clean"].shape[1:] == (w["length"], n_ch) == out["X_filt"].shape[1:]
    meta = out["meta"]
    hop = hop_size(w["length"], w["overlap"])
    for sid, part in meta.groupby("subject_id"):
        n = (clean.subject_id == sid).sum()
        assert len(part) == (n - w["length"]) // hop + 1
        assert (part["end_step"] <= n).all()
    # örtüşme: ardışık pencerelerin ikinci yarısı = sonrakinin ilk yarısı
    i = 0
    assert meta.subject_id[i] == meta.subject_id[i + 1]
    np.testing.assert_array_equal(out["X_clean"][i, hop:], out["X_clean"][i + 1, : w["length"] - hop])
    assert set(out["groups"]) == set(clean.subject_id)
    assert (pd.Series(out["y"]).groupby(out["groups"]).nunique() == 1).all()
