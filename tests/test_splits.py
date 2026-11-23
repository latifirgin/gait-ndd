"""Kişi bazlı bölme ve sızıntı kontrol testleri."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from gait.evaluation.splits import (
    FOLD_COLUMNS,
    LeakageError,
    assert_no_leakage,
    assign_folds,
    encode_labels,
    get_inner_cv,
    iter_folds,
    load_folds,
    make_folds,
    save_folds,
    subject_table,
    validate_folds,
)

CLASS_COUNTS = [16, 15, 20, 13]  # Kontrol, PD, HD, ALS (gaitndd)
N_SPLITS, SEED = 5, 42


def _windows(class_counts=CLASS_COUNTS, seed=0):
    """Pencere düzeyinde sahte (y, groups): denek başına 8–20 pencere."""
    rng = np.random.default_rng(seed)
    y, g = [], []
    for label, n_subj in enumerate(class_counts):
        for s in range(n_subj):
            n_win = int(rng.integers(8, 21))
            y += [label] * n_win
            g += [f"c{label}_s{s:02d}"] * n_win
    return np.array(y), np.array(g)


@pytest.fixture(scope="module")
def data():
    y, g = _windows()
    return y, g, make_folds(y, g, N_SPLITS, SEED, min_test_subjects_per_class=2)


def test_schema_and_every_subject_once(data):
    y, g, folds = data
    assert list(folds.columns) == FOLD_COLUMNS
    assert folds["subject_id"].is_unique
    assert set(folds["subject_id"]) == set(g)
    assert folds["n_windows"].sum() == len(y)
    assert set(folds["fold"]) == set(range(N_SPLITS))


def test_iter_folds_disjoint_and_covering(data):
    y, g, folds = data
    seen_test = []
    for k, tr, te in iter_folds(g, folds):
        assert set(g[tr]).isdisjoint(set(g[te])), f"fold {k}: sızıntı"
        assert len(np.intersect1d(tr, te)) == 0
        assert len(tr) + len(te) == len(y)
        seen_test.append(te)
    all_test = np.sort(np.concatenate(seen_test))
    np.testing.assert_array_equal(all_test, np.arange(len(y)))  # her pencere tam 1 kez testte


def test_stratification_balanced(data):
    _, _, folds = data
    counts = folds.groupby(["fold", "label"]).size().unstack(fill_value=0)
    assert (counts >= 2).all().all()                       # her fold'da her sınıftan ≥2 denek
    assert (counts.max() - counts.min()).max() <= 1        # sınıf başına fold'lar arası fark ≤1


def test_deterministic_with_seed():
    y, g = _windows()
    a = make_folds(y, g, N_SPLITS, SEED)
    b = make_folds(y, g, N_SPLITS, SEED)
    c = make_folds(y, g, N_SPLITS, SEED + 1)
    pd.testing.assert_frame_equal(a, b)
    merged = a.merge(c, on="subject_id")
    assert (merged["fold_x"] != merged["fold_y"]).any()


def test_independent_of_window_order(data):
    y, g, folds = data
    perm = np.random.default_rng(1).permutation(len(y))
    shuffled = make_folds(y[perm], g[perm], N_SPLITS, SEED, 2)
    pd.testing.assert_frame_equal(folds, shuffled)


def test_inconsistent_subject_label_raises():
    y, g = _windows()
    y = y.copy()
    y[0] = (y[0] + 1) % 4
    with pytest.raises(ValueError, match="Birden fazla etiketi"):
        subject_table(y, g)


def test_too_few_subjects_raises():
    y, g = _windows(class_counts=[16, 15, 20, 3])
    with pytest.raises(ValueError, match="yetersiz"):
        make_folds(y, g, N_SPLITS, SEED)


def test_assert_no_leakage_detects_overlap():
    assert_no_leakage(["a", "b"], ["c"])
    with pytest.raises(LeakageError, match="b"):
        assert_no_leakage(["a", "b"], ["b", "c"])


def test_unknown_subject_raises(data):
    _, g, folds = data
    with pytest.raises(ValueError, match="bulunmayan"):
        assign_folds(np.append(g, "yeni_denek"), folds)


def test_validate_rejects_duplicate_subject(data):
    _, _, folds = data
    bad = pd.concat([folds, folds.iloc[[0]].assign(fold=(folds.iloc[0]["fold"] + 1) % N_SPLITS)])
    with pytest.raises(ValueError, match="Birden fazla fold"):
        validate_folds(bad, N_SPLITS)


def test_save_load_roundtrip(tmp_path, data):
    _, _, folds = data
    path = save_folds(folds, tmp_path / "folds.csv")
    loaded = load_folds(path, N_SPLITS)
    pd.testing.assert_frame_equal(folds.reset_index(drop=True), loaded[FOLD_COLUMNS])


def test_inner_cv_respects_groups(data):
    y, g, folds = data
    _, tr, _ = next(iter_folds(g, folds))
    for itr, ite in get_inner_cv(3, SEED).split(np.zeros(len(tr)), y[tr], groups=g[tr]):
        assert set(g[tr][itr]).isdisjoint(set(g[tr][ite]))


def test_encode_labels():
    names = ["control", "park", "hunt", "als"]
    np.testing.assert_array_equal(encode_labels(["Control", "ALS", "hunt"], names), [0, 3, 2])
    np.testing.assert_array_equal(encode_labels(np.array([0, 3]), names), [0, 3])
    with pytest.raises(ValueError):
        encode_labels(["mystery"], names)
    with pytest.raises(ValueError):
        encode_labels(np.array([4]), names)


def test_window_level_random_split_would_leak():
    """Belgeleme testi: pencere bazlı rastgele KFold neredeyse her deneği iki tarafa böler.
    Bu yüzden projede yalnızca folds.csv kullanılır."""
    y, g = _windows()
    tr, te = next(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(y))
    with pytest.raises(LeakageError):
        assert_no_leakage(g[tr], g[te])
