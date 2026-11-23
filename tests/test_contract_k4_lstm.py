"""K4 kabul testleri – LSTM yardımcıları ve SHAP. Uygulanana kadar SKIP."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import run_or_skip

from gait.models import lstm

pytestmark = pytest.mark.contract


def test_channel_scaler_train_only():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(5, 2, (50, 32, 3))
    Xte = rng.normal(9, 2, (10, 32, 3))
    sc = run_or_skip(lstm.ChannelScaler().fit, Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    np.testing.assert_allclose(Ztr.reshape(-1, 3).mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(Ztr.reshape(-1, 3).std(0), 1, atol=1e-2)
    assert Zte.mean() > 1     # test, eğitim istatistikleriyle ölçeklendi (kendi ortalamasıyla değil)


def test_validation_split_is_subject_level(small_windows):
    w = small_windows
    tr, va = run_or_skip(lstm.subject_validation_split, w["groups"], w["y"], 0.2, 42)
    assert set(w["groups"][tr]).isdisjoint(w["groups"][va])
    assert len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == len(w["y"])


def test_build_lstm_output_shape(cfg):
    pytest.importorskip("tensorflow")
    model = run_or_skip(lstm.build_lstm, (32, 12), 4, cfg["lstm"])
    out = model.predict(np.zeros((2, 32, 12), dtype="float32"), verbose=0)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out.sum(1), 1, atol=1e-5)
