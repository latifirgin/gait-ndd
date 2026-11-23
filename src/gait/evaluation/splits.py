"""Kişi (denek) bazlı çapraz doğrulama bölmeleri ve sızıntı kontrolleri.

Tasarım kararları
-----------------
* Bölme DENEK düzeyinde yapılır: her denek tek bir satırla temsil edilir ve
  ``StratifiedGroupKFold`` (gruplar = denek kimliği) ile fold'lara atanır. Böylece
  katmanlama, pencere sayısına göre değil denek sayısına göre yapılır. ALS/HD
  deneklerinde adım süresi farklı olduğundan denek başına pencere sayısı sınıfa
  göre değişir; pencere düzeyinde katmanlama çok adımlı deneklere fazla ağırlık
  verirdi.
* Sonuç tek bir ``folds.csv`` dosyasıdır (subject_id, label, fold, n_windows).
  K3 (klasik ML) ve K4 (LSTM) aynı dosyayı kullanır; bu da modellerin aynı test
  deneklerinde kıyaslanmasını garanti eder.
* ``iter_folds`` her fold için eğitim/test indekslerini üretir ve her adımda
  ``assert_no_leakage`` çağırır.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

FOLD_COLUMNS: list[str] = ["subject_id", "label", "fold", "n_windows"]


class LeakageError(AssertionError):
    """Aynı denek hem eğitim hem test kümesinde bulunduğunda fırlatılır."""


# ---------------------------------------------------------------------------
# Etiket yardımcıları
# ---------------------------------------------------------------------------
def encode_labels(y: Sequence, class_names: Sequence[str]) -> np.ndarray:
    """y'yi 0..K-1 tamsayılarına çevirir.

    Tamsayı ise aralığı denetlenir; metin ise ``class_names`` ile büyük/küçük harf
    duyarsız eşlenir. Bilinmeyen etiket ValueError üretir.
    """
    y_arr = np.asarray(y)
    n_classes = len(class_names)
    if np.issubdtype(y_arr.dtype, np.integer):
        if y_arr.size and (y_arr.min() < 0 or y_arr.max() >= n_classes):
            raise ValueError(
                f"Tamsayı etiketler 0..{n_classes - 1} aralığında olmalı; "
                f"bulunan aralık: {y_arr.min()}..{y_arr.max()}"
            )
        return y_arr.astype(np.int64)
    lookup = {name.lower(): i for i, name in enumerate(class_names)}
    out = np.empty(y_arr.shape[0], dtype=np.int64)
    for i, label in enumerate(y_arr):
        key = str(label).strip().lower()
        if key not in lookup:
            raise ValueError(f"Bilinmeyen etiket '{label}'. Beklenen: {list(class_names)}")
        out[i] = lookup[key]
    return out


def subject_table(y: Sequence[int], groups: Sequence) -> pd.DataFrame:
    """Pencere düzeyindeki (y, groups) dizilerinden denek tablosu çıkarır.

    Her denek tek etikete sahip olmalıdır; aksi hâlde veri hatası vardır.
    """
    y_arr = np.asarray(y)
    g_arr = np.asarray(groups).astype(str)
    if y_arr.shape[0] != g_arr.shape[0]:
        raise ValueError(f"y ({y_arr.shape[0]}) ve groups ({g_arr.shape[0]}) uzunlukları farklı.")
    df = pd.DataFrame({"subject_id": g_arr, "label": y_arr})
    n_labels = df.groupby("subject_id")["label"].nunique()
    bad = n_labels[n_labels > 1]
    if not bad.empty:
        raise ValueError(f"Birden fazla etiketi olan denekler: {sorted(bad.index.tolist())}")
    tbl = (
        df.groupby("subject_id", sort=True)
        .agg(label=("label", "first"), n_windows=("label", "size"))
        .reset_index()
    )
    tbl["label"] = tbl["label"].astype(np.int64)
    return tbl


# ---------------------------------------------------------------------------
# Fold üretimi
# ---------------------------------------------------------------------------
def make_folds(
    y: Sequence[int],
    groups: Sequence,
    n_splits: int,
    seed: int,
    min_test_subjects_per_class: int = 1,
) -> pd.DataFrame:
    """Her deneği tam olarak bir test fold'una atar.

    Parametreler
    ------------
    y, groups : pencere düzeyinde tamsayı etiketler ve denek kimlikleri.
    n_splits  : fold sayısı.
    seed      : karıştırma tohumu (tekrar üretilebilirlik).
    min_test_subjects_per_class : her test fold'unda her sınıftan en az bu kadar
        denek bulunmalı; sağlanamazsa ValueError.

    Döndürür
    --------
    DataFrame[subject_id, label, fold, n_windows]
    """
    tbl = subject_table(y, groups)
    class_counts = tbl["label"].value_counts()
    too_small = class_counts[class_counts < n_splits * min_test_subjects_per_class]
    if not too_small.empty:
        raise ValueError(
            f"{n_splits} fold ve fold başına en az {min_test_subjects_per_class} denek için "
            f"yetersiz denek sayısı olan sınıflar: {too_small.to_dict()}"
        )

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = np.full(len(tbl), -1, dtype=np.int64)
    dummy_x = np.zeros((len(tbl), 1))
    for k, (_, test_idx) in enumerate(
        sgkf.split(dummy_x, tbl["label"].to_numpy(), groups=tbl["subject_id"].to_numpy())
    ):
        fold_of[test_idx] = k
    tbl["fold"] = fold_of
    tbl = tbl[FOLD_COLUMNS].sort_values(["fold", "label", "subject_id"]).reset_index(drop=True)
    validate_folds(tbl, n_splits=n_splits, min_test_subjects_per_class=min_test_subjects_per_class)
    return tbl


def validate_folds(
    folds: pd.DataFrame,
    n_splits: int,
    n_classes: int | None = None,
    min_test_subjects_per_class: int = 1,
) -> None:
    """folds tablosunun şemasını ve bütünlüğünü denetler; hata varsa ValueError."""
    missing = set(FOLD_COLUMNS) - set(folds.columns)
    if missing:
        raise ValueError(f"folds tablosunda eksik sütunlar: {sorted(missing)}")
    dup = folds["subject_id"][folds["subject_id"].duplicated()]
    if not dup.empty:
        raise ValueError(f"Birden fazla fold'a atanmış denekler: {sorted(dup.unique().tolist())}")
    if folds["fold"].isna().any() or (folds["fold"] < 0).any():
        raise ValueError("Fold'a atanmamış denek(ler) var.")
    found = set(folds["fold"].unique().tolist())
    if found != set(range(n_splits)):
        raise ValueError(f"Beklenen fold'lar {list(range(n_splits))}, bulunan {sorted(found)}")
    classes = range(n_classes) if n_classes is not None else sorted(folds["label"].unique())
    counts = folds.groupby(["fold", "label"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(classes), fill_value=0)
    lacking = counts[counts.lt(min_test_subjects_per_class).any(axis=1)]
    if not lacking.empty:
        raise ValueError(
            "Bazı test fold'larında sınıf başına denek sayısı "
            f"{min_test_subjects_per_class}'den az:\n{lacking}"
        )


def fold_summary(folds: pd.DataFrame, class_names: Sequence[str] | None = None) -> pd.DataFrame:
    """Fold × sınıf denek sayısı tablosu (+ toplam denek ve pencere sayısı)."""
    counts = folds.groupby(["fold", "label"]).size().unstack(fill_value=0)
    if class_names is not None:
        counts = counts.reindex(columns=range(len(class_names)), fill_value=0)
        counts.columns = list(class_names)
    counts["n_subjects"] = folds.groupby("fold").size()
    counts["n_windows"] = folds.groupby("fold")["n_windows"].sum()
    return counts


# ---------------------------------------------------------------------------
# Sızıntı kontrolleri ve fold yineleyicisi
# ---------------------------------------------------------------------------
def assert_no_leakage(train_groups: Sequence, test_groups: Sequence) -> None:
    """Eğitim ve test denek kümeleri kesişiyorsa LeakageError fırlatır."""
    overlap = set(np.asarray(train_groups).astype(str)) & set(np.asarray(test_groups).astype(str))
    if overlap:
        raise LeakageError(f"Veri sızıntısı: {len(overlap)} denek hem eğitimde hem testte: {sorted(overlap)}")


def assign_folds(groups: Sequence, folds: pd.DataFrame) -> np.ndarray:
    """Her pencere/satır için fold numarasını döndürür. folds.csv'de olmayan
    denek varsa ValueError (veri ile bölme dosyası uyumsuz)."""
    g_arr = np.asarray(groups).astype(str)
    mapping = dict(zip(folds["subject_id"].astype(str), folds["fold"].astype(np.int64)))
    unknown = sorted(set(g_arr) - set(mapping))
    if unknown:
        raise ValueError(
            f"folds.csv'de bulunmayan denekler: {unknown}. "
            "windows.npz/features.parquet değişti mi? folds.csv'yi yeniden üretin."
        )
    return np.array([mapping[g] for g in g_arr], dtype=np.int64)


def iter_folds(
    groups: Sequence, folds: pd.DataFrame
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """(fold, train_idx, test_idx) üretir. Her adımda sızıntı denetlenir.

    ``groups`` herhangi bir veri kaynağının (windows.npz 'groups' ya da
    features.parquet 'subject_id') satır sırasına göre denek kimlikleridir.
    """
    g_arr = np.asarray(groups).astype(str)
    fold_ids = assign_folds(g_arr, folds)
    for k in sorted(np.unique(fold_ids)):
        test_idx = np.flatnonzero(fold_ids == k)
        train_idx = np.flatnonzero(fold_ids != k)
        assert_no_leakage(g_arr[train_idx], g_arr[test_idx])
        yield int(k), train_idx, test_idx


def get_inner_cv(n_splits: int, seed: int) -> StratifiedGroupKFold:
    """İç CV (hiperparametre araması) için bölücü.

    Kullanım: ``GridSearchCV(pipe, grid, cv=get_inner_cv(...)).fit(X_tr, y_tr, groups=g_tr)``
    ``groups`` verilmezse iç CV pencere bazlı bölünür ve sızıntı olur; mutlaka verin.
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# ---------------------------------------------------------------------------
# G/Ç
# ---------------------------------------------------------------------------
def save_folds(folds: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    folds[FOLD_COLUMNS].to_csv(path, index=False)
    return path


def load_folds(path: str | Path, n_splits: int | None = None) -> pd.DataFrame:
    folds = pd.read_csv(path, dtype={"subject_id": str})
    if n_splits is None:
        n_splits = int(folds["fold"].max()) + 1
    validate_folds(folds, n_splits=n_splits)
    return folds
