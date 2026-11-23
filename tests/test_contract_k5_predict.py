"""K3→K5 tahmin arayüzü kabul testi. Model dosyası/uygulama yoksa SKIP."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import run_or_skip

from gait.models import predict as pr
from gait.utils import class_names, resolve

pytestmark = pytest.mark.contract


def test_predict_contract(cfg, small_steps):
    path = resolve(cfg["classical"]["model_dir"]) / "rf.joblib"
    if not path.exists():
        pytest.skip("results/models/rf.joblib yok (K3: fit_final_model ile üret)")
    model = run_or_skip(pr.load_model, path)
    one = small_steps[small_steps.subject_id == small_steps.subject_id.iloc[0]]
    raw = one.drop(columns=["subject_id", "label", "step"])
    out = run_or_skip(pr.predict, raw, model=model, cfg=cfg, top_k=5)
    assert set(pr.RESULT_KEYS) <= set(out)
    assert out["label"] in class_names(cfg)
    assert set(out["proba"]) == set(class_names(cfg))
    assert sum(out["proba"].values()) == pytest.approx(1.0)
    assert list(out["shap"].columns) == ["feature", "shap"] and len(out["shap"]) <= 5
    assert out["n_windows"] >= 1
    assert np.all(np.diff(out["shap"]["shap"].abs().to_numpy()) <= 1e-12)
