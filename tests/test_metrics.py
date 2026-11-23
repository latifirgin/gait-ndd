"""Metrik, denek düzeyi birleştirme, grafik ve kıyas tablosu testleri."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from gait.evaluation.metrics import (  # noqa: E402
    METRIC_COLUMNS,
    aggregate_by_subject,
    check_proba,
    comparison_table,
    evaluate,
    evaluate_cv,
    flag_suspicious,
    load_metrics,
    make_predictions_frame,
    plot_confusion_matrix,
    plot_roc_curves,
    pooled_predictions,
    save_metrics_csv,
    table_to_markdown,
)

K = 4
NAMES = ["Kontrol", "Parkinson", "Huntington", "ALS"]


def _one_hot(y, k=K, eps=0.0):
    p = np.full((len(y), k), eps / (k - 1))
    p[np.arange(len(y)), y] = 1 - eps
    return p


def test_perfect_predictions():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    m = evaluate(y, y, _one_hot(y), K)
    assert all(v == pytest.approx(1.0) for v in m.values())


def test_hand_computed_values():
    # Sınıf 0: TP=2, FN=0, FP=1 → P=2/3, R=1,   F1=0.8
    # Sınıf 1: TP=1, FN=1, FP=0 → P=1,   R=0.5, F1=2/3
    # Sınıf 2: TP=1, FN=0, FP=0 → 1,1,1 ; Sınıf 3: TP=1 → 1,1,1
    y_true = np.array([0, 0, 1, 1, 2, 3])
    y_pred = np.array([0, 0, 1, 0, 2, 3])
    m = evaluate(y_true, y_pred, None, K)
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["precision"] == pytest.approx((2 / 3 + 1 + 1 + 1) / 4)
    assert m["recall"] == pytest.approx((1 + 0.5 + 1 + 1) / 4)
    assert m["macro_f1"] == pytest.approx((0.8 + 2 / 3 + 1 + 1) / 4)
    assert np.isnan(m["roc_auc"])


def test_never_predicted_class_penalised():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 2])  # ALS hiç tahmin edilmiyor
    m = evaluate(y_true, y_pred, None, K)
    assert m["recall"] == pytest.approx(0.75)


def test_check_proba_rejects_bad_input():
    with pytest.raises(ValueError, match="şekli"):
        check_proba(np.ones((3, 3)) / 3, K)
    with pytest.raises(ValueError, match="toplam"):
        check_proba(np.full((2, K), 0.5), K)
    with pytest.raises(ValueError, match="NaN"):
        check_proba(np.array([[np.nan, 0.5, 0.25, 0.25]]), K)


def test_random_scores_auc_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, K, 4000)
    p = rng.dirichlet(np.ones(K), size=len(y))
    m = evaluate(y, p.argmax(1), p, K)
    assert m["roc_auc"] == pytest.approx(0.5, abs=0.03)


def test_missing_class_in_fold_auc_warns_not_crashes():
    y = np.array([0, 0, 1, 1, 2, 2])  # ALS yok
    with pytest.warns(RuntimeWarning):
        m = evaluate(y, y, _one_hot(y, eps=0.1), K)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_aggregate_by_subject_soft_vote():
    y = np.array([1, 1, 1, 2, 2])
    g = np.array(["a", "a", "a", "b", "b"])
    p = np.array([[0, .6, .4, 0], [0, .2, .8, 0], [0, .7, .3, 0], [0, 0, 1, 0], [0, .5, .5, 0]])
    ids, ys, ps = aggregate_by_subject(y, p, g)
    assert ids.tolist() == ["a", "b"]
    assert ys.tolist() == [1, 2]
    np.testing.assert_allclose(ps[0], [0, 0.5, 0.5, 0])
    np.testing.assert_allclose(ps[1], [0, 0.25, 0.75, 0])


def _fake_predictions(n_folds=5, subj_per_class=3, win=6, seed=0, noise=0.6):
    rng = np.random.default_rng(seed)
    frames = []
    for f in range(n_folds):
        sid, wid, yt = [], [], []
        for c in range(K):
            for s in range(subj_per_class):
                sid += [f"f{f}_c{c}_s{s}"] * win
                wid += list(range(win))
                yt += [c] * win
        yt = np.array(yt)
        logits = np.eye(K)[yt] * 2 + rng.normal(0, noise * 2, (len(yt), K))
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        frames.append(make_predictions_frame(sid, wid, f, yt, p))
    return pd.concat(frames, ignore_index=True)


def test_evaluate_cv_contract_and_levels():
    preds = _fake_predictions()
    for level, expected_rows in (("window", 5), ("subject", 5)):
        m = evaluate_cv(preds, "rf", K, level=level)
        assert list(m.columns[: len(METRIC_COLUMNS)]) == METRIC_COLUMNS
        assert len(m) == expected_rows and (m["level"] == level).all()
        vals = m[["accuracy", "macro_f1", "roc_auc"]].to_numpy()
        assert ((vals >= 0) & (vals <= 1)).all()
    with pytest.raises(ValueError):
        evaluate_cv(preds, "rf", K, level="fold")


def test_save_load_and_comparison_table(tmp_path):
    preds = _fake_predictions()
    for model in ("rf", "svm"):
        m = pd.concat([evaluate_cv(preds, model, K, lv) for lv in ("window", "subject")])
        save_metrics_csv(m, tmp_path, model)
    # sözleşmedeki minimum sütunlarla (level olmadan) yazılmış dosya da okunmalı
    legacy = evaluate_cv(preds, "knn", K)[METRIC_COLUMNS]
    legacy.to_csv(tmp_path / "metrics_knn.csv", index=False)

    allm = load_metrics(tmp_path)
    numeric, pretty = comparison_table(allm)
    assert ("knn", "window") in numeric.index
    rf = allm[(allm.model == "rf") & (allm.level == "window")]
    assert numeric.loc[("rf", "window"), "accuracy_mean"] == pytest.approx(rf.accuracy.mean())
    assert numeric.loc[("rf", "window"), "accuracy_std"] == pytest.approx(rf.accuracy.std(ddof=1))
    assert "±" in pretty.iloc[0]["accuracy"]
    assert table_to_markdown(pretty).count("\n") == len(pretty) + 2


def test_load_metrics_rejects_bad_schema(tmp_path):
    pd.DataFrame({"model": ["x"], "fold": [0], "accuracy": [0.5]}).to_csv(tmp_path / "metrics_x.csv", index=False)
    with pytest.raises(ValueError, match="sözleşme"):
        load_metrics(tmp_path)


def test_flag_suspicious():
    m = pd.DataFrame({"model": ["a"] * 2, "level": ["window"] * 2, "accuracy": [0.97, 0.99]})
    assert flag_suspicious(m, 0.95) and not flag_suspicious(m, 0.995)


def test_plots_write_files(tmp_path):
    preds = _fake_predictions()
    yt, yp, pp = pooled_predictions(preds, K, level="window")
    fig, cm = plot_confusion_matrix(yt, yp, NAMES, path=tmp_path / "cm.png", dpi=50)
    plt.close(fig)
    assert cm.sum() == len(yt) and cm.shape == (K, K)
    np.testing.assert_array_equal(cm.sum(1), np.bincount(yt, minlength=K))
    fig, aucs = plot_roc_curves(yt, pp, NAMES, path=tmp_path / "roc.png", dpi=50)
    plt.close(fig)
    assert set(aucs) == set(NAMES) | {"macro"}
    assert (tmp_path / "cm.png").stat().st_size > 0 and (tmp_path / "roc.png").stat().st_size > 0
