"""K3 kabul testleri – klasik modeller (Adım 3). Uygulanana kadar SKIP."""
from __future__ import annotations

import copy

import pandas as pd
import pytest
from conftest import run_or_skip
from sklearn.preprocessing import StandardScaler

from gait.evaluation.metrics import check_proba, proba_columns
from gait.evaluation.splits import make_folds
from gait.models import classical

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("name", classical.MODEL_NAMES)
def test_pipeline_scaler_first(cfg, name):
    pipe = run_or_skip(classical.build_pipeline, name, cfg["classical"], 42)
    assert pipe.steps[0][0] == "scaler" and isinstance(pipe.steps[0][1], StandardScaler)
    assert pipe.steps[-1][0] == "clf" and hasattr(pipe, "predict_proba")


@pytest.mark.parametrize("name", classical.MODEL_NAMES)
def test_param_grid_prefixed(cfg, name):
    grid = run_or_skip(classical.param_grid, name, cfg["classical"])
    assert grid and all(k.startswith("clf__") for k in grid)


def _fake_features(small_windows):
    w = small_windows
    X = w["X_clean"]
    feats = pd.DataFrame({"stride_l__mean": X[:, :, 0].mean(1), "stride_l__std": X[:, :, 0].std(1),
                          "swing_pct_l__mean": X[:, :, 4].mean(1)})
    ids = pd.DataFrame({"subject_id": w["groups"], "label": w["y"], "window_idx": w["meta"]["window_idx"]})
    return pd.concat([ids, feats], axis=1)


def test_run_cv_contract(cfg, small_windows):
    c = copy.deepcopy(cfg)
    c["evaluation"]["inner_n_splits"] = 2
    c["classical"]["models"]["rf"] = {"n_estimators": [20], "max_depth": [3]}
    feats = _fake_features(small_windows)
    folds = make_folds(feats["label"], feats["subject_id"], n_splits=3, seed=42)
    preds, best = run_or_skip(classical.run_cv, feats, folds, "rf", c)
    assert len(preds) == len(feats) and preds["window_idx"].is_unique
    fold_of = dict(zip(folds.subject_id, folds.fold))
    assert (preds["subject_id"].map(fold_of) == preds["fold"]).all()   # her pencere kendi test fold'unda
    check_proba(preds[proba_columns(4)].to_numpy(), 4)
    assert len(best) == 3
    # sentetik veride şans düzeyinin üstünde olmalı
    assert (preds["y_true"] == preds["y_pred"]).mean() > 0.4
