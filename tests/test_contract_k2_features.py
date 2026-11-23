"""K2 kabul testleri – features (Adım 2). Uygulanana kadar SKIP."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import run_or_skip

from gait.features import freq_domain as fd
from gait.features import time_domain as td
from gait.features.build import ID_COLUMNS, build_feature_table, extract_features, validate_feature_table

pytestmark = pytest.mark.contract
L = 32


def _X(values):  # (L,) → (1, L, 1)
    return np.asarray(values, dtype=float).reshape(1, -1, 1)


def test_time_domain_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0] * 8)
    X = _X(x)
    assert run_or_skip(td.window_mean, X)[0, 0] == pytest.approx(2.5)
    assert run_or_skip(td.window_std, X)[0, 0] == pytest.approx(np.std(x, ddof=1))
    assert run_or_skip(td.window_var, X)[0, 0] == pytest.approx(np.var(x, ddof=1))
    assert run_or_skip(td.window_rms, X)[0, 0] == pytest.approx(np.sqrt(np.mean(x**2)))
    assert run_or_skip(td.window_cov, X)[0, 0] == pytest.approx(100 * np.std(x, ddof=1) / 2.5)


def test_time_domain_shapes():
    X = np.random.default_rng(0).uniform(0.5, 1.5, (7, L, 12))
    for fn in (td.window_mean, td.window_std, td.window_var, td.window_rms, td.window_cov):
        assert run_or_skip(fn, X).shape == (7, 12)


def test_asymmetry():
    X = np.ones((2, L, 2))
    X[1, :, 1] = 1.2
    a = run_or_skip(td.asymmetry_index, X, 0, 1)
    assert a[0] == pytest.approx(0.0)
    assert a[1] == pytest.approx(100 * 0.2 / 1.1)


def test_dominant_frequency_of_sine():
    n = np.arange(L)
    X = _X(5 + np.sin(2 * np.pi * 4 * n / L))   # 4 döngü / 32 adım, DC=5
    assert run_or_skip(fd.dominant_frequency, X)[0, 0] == pytest.approx(4 / L)


def test_spectral_entropy_ordering():
    n = np.arange(L)
    sine = run_or_skip(fd.spectral_entropy, _X(np.sin(2 * np.pi * 4 * n / L)))[0, 0]
    noise = np.random.default_rng(0).standard_normal((200, L, 1))
    h_noise = run_or_skip(fd.spectral_entropy, noise).mean()
    const = run_or_skip(fd.spectral_entropy, _X(np.ones(L)))[0, 0]
    assert 0 <= sine < 0.2 < 0.7 < h_noise <= 1
    assert const == pytest.approx(0.0)


def test_feature_table_contract(cfg, small_windows):
    w = small_windows
    df = run_or_skip(build_feature_table, w["X_clean"], w["y"], w["groups"], w["meta"]["window_idx"].to_numpy(),
                     cfg["data"]["channels"], cfg["features"])
    validate_feature_table(df, n_rows=len(w["y"]))
    assert list(df.columns[:3]) == ID_COLUMNS
    assert "stride_l__cov" in df.columns and "stride_l__spectral_entropy" in df.columns
    assert any(c.startswith("asym__") for c in df.columns)
    assert not any("label" in c for c in df.columns[3:])


def test_extract_features_deterministic_and_rowwise(cfg, small_windows):
    X = small_windows["X_clean"]
    a = run_or_skip(extract_features, X, cfg["data"]["channels"], cfg["features"])
    b = extract_features(X[:5], cfg["data"]["channels"], cfg["features"])
    np.testing.assert_allclose(a.iloc[:5].to_numpy(), b.to_numpy())   # pencereler birbirinden bağımsız
