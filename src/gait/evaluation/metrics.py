"""Değerlendirme metrikleri, denek düzeyinde birleştirme, grafikler ve kıyas tablosu.

Sözleşmeler
-----------
* metrics_<model>.csv : model, fold, accuracy, macro_f1, precision, recall, roc_auc
  (+ önerilen ek sütun ``level`` ∈ {window, subject}; okuyucu yoksa 'window' varsayar)
* predictions_<model>.csv (önerilen, K3/K4 yazar): subject_id, window_idx, fold,
  y_true, y_pred, p_0..p_{K-1}  → birleşik karmaşıklık matrisi ve ROC için gerekir.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

METRIC_NAMES: list[str] = ["accuracy", "macro_f1", "precision", "recall", "roc_auc"]
METRIC_COLUMNS: list[str] = ["model", "fold", *METRIC_NAMES]
LEVELS = ("window", "subject")
PROBA_DECIMALS = 12  # sayısal kararlılık sabiti (hiperparametre değil)


# ---------------------------------------------------------------------------
# Doğrulama yardımcıları
# ---------------------------------------------------------------------------
def proba_columns(n_classes: int) -> list[str]:
    return [f"p_{k}" for k in range(n_classes)]


def check_proba(proba: np.ndarray, n_classes: int, atol: float = 1e-6) -> np.ndarray:
    """Olasılık matrisini denetler: (N, K), sonlu, [0,1], satır toplamı 1."""
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != n_classes:
        raise ValueError(f"proba şekli (N, {n_classes}) olmalı; bulunan {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("proba NaN/inf içeriyor.")
    if (p < -atol).any() or (p > 1 + atol).any():
        raise ValueError("proba [0, 1] aralığı dışında değerler içeriyor.")
    if not np.allclose(p.sum(axis=1), 1.0, atol=atol):
        raise ValueError("proba satır toplamları 1 değil (softmax/predict_proba çıktısı mı?).")
    return p


# ---------------------------------------------------------------------------
# Metrikler
# ---------------------------------------------------------------------------
def multiclass_roc_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """Makro One-vs-Rest ROC-AUC.

    Tüm sınıflar y_true'da varsa sklearn kullanılır. Bir sınıf yoksa (küçük test
    kümesi) yalnızca hem pozitif hem negatif örneği olan sınıflar üzerinden
    ortalama alınır ve uyarı verilir. Tek sınıf varsa NaN.
    """
    y_true = np.asarray(y_true)
    present = np.unique(y_true)
    if present.size < 2:
        return float("nan")
    if present.size == n_classes:
        return float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=np.arange(n_classes))
        )
    warnings.warn(
        f"y_true'da yalnızca {present.tolist()} sınıfları var; ROC-AUC bu sınıflar üzerinden hesaplandı.",
        RuntimeWarning,
        stacklevel=2,
    )
    aucs = [roc_auc_score((y_true == k).astype(int), proba[:, k]) for k in present]
    return float(np.mean(aucs))


def evaluate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: np.ndarray | None,
    n_classes: int,
    atol: float = 1e-6,
) -> dict[str, float]:
    """Accuracy, Macro-F1, Macro-Precision, Macro-Recall ve Makro OvR ROC-AUC.

    Makro ortalama tüm K sınıf üzerinden alınır (sınıf dengesizliğinde her sınıf
    eşit ağırlıklı). Tahmin edilmeyen bir sınıfın precision'ı 0 sayılır.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} ve y_pred {y_pred.shape} şekilleri farklı.")
    if y_true.size == 0:
        raise ValueError("Boş girdi.")
    labels = np.arange(n_classes)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float("nan"),
    }
    if y_proba is not None:
        p = check_proba(y_proba, n_classes, atol=atol)
        if p.shape[0] != y_true.shape[0]:
            raise ValueError("y_proba satır sayısı y_true ile aynı olmalı.")
        out["roc_auc"] = multiclass_roc_auc(y_true, p, n_classes)
    return out


def aggregate_by_subject(
    y_true: Sequence[int], proba: np.ndarray, groups: Sequence
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pencere olasılıklarını denek düzeyinde ortalar (yumuşak oylama).

    p_s = (1/|W_s|) Σ_{w∈W_s} p_w ,  ŷ_s = argmax_k p_s[k]

    Döndürür: (subject_ids, y_true_subject, proba_subject)
    """
    df = pd.DataFrame(np.asarray(proba, dtype=float))
    df["subject_id"] = np.asarray(groups).astype(str)
    df["y"] = np.asarray(y_true, dtype=np.int64)
    if (df.groupby("subject_id")["y"].nunique() > 1).any():
        raise ValueError("Bir deneğin pencereleri farklı etiketlere sahip.")
    agg = df.groupby("subject_id", sort=True).mean(numeric_only=True)
    p_cols = [c for c in agg.columns if c != "y"]
    # Yuvarlama: ortalamadaki kayan nokta gürültüsü (1e-17) eşit skorların
    # sırasını bozup ROC-AUC'yi yapay olarak değiştirmesin.
    proba_s = np.round(agg[p_cols].to_numpy(), PROBA_DECIMALS)
    return agg.index.to_numpy(), agg["y"].round().astype(np.int64).to_numpy(), proba_s


# ---------------------------------------------------------------------------
# Fold-dışı (out-of-fold) tahmin tablosu ve CV değerlendirmesi
# ---------------------------------------------------------------------------
def make_predictions_frame(
    subject_id: Sequence,
    window_idx: Sequence[int],
    fold: Sequence[int] | int,
    y_true: Sequence[int],
    proba: np.ndarray,
) -> pd.DataFrame:
    """K3/K4'ün her fold sonunda biriktireceği tahmin tablosu."""
    p = np.asarray(proba, dtype=float)
    n = p.shape[0]
    df = pd.DataFrame(
        {
            "subject_id": np.asarray(subject_id).astype(str),
            "window_idx": np.asarray(window_idx, dtype=np.int64),
            "fold": np.broadcast_to(np.asarray(fold, dtype=np.int64), (n,)),
            "y_true": np.asarray(y_true, dtype=np.int64),
            "y_pred": p.argmax(axis=1),
        }
    )
    for k in range(p.shape[1]):
        df[f"p_{k}"] = p[:, k]
    return df


def evaluate_cv(
    predictions: pd.DataFrame,
    model: str,
    n_classes: int,
    level: str = "window",
    atol: float = 1e-6,
) -> pd.DataFrame:
    """Fold başına metrikler. level='subject' ise önce denek düzeyine birleştirilir."""
    if level not in LEVELS:
        raise ValueError(f"level {LEVELS} içinden olmalı.")
    p_cols = proba_columns(n_classes)
    rows = []
    for k, part in predictions.groupby("fold", sort=True):
        y = part["y_true"].to_numpy()
        p = part[p_cols].to_numpy()
        if level == "subject":
            _, y, p = aggregate_by_subject(y, p, part["subject_id"])
        m = evaluate(y, p.argmax(axis=1), p, n_classes, atol=atol)
        rows.append({"model": model, "fold": int(k), **m, "level": level})
    return pd.DataFrame(rows, columns=[*METRIC_COLUMNS, "level"])


def pooled_predictions(
    predictions: pd.DataFrame, n_classes: int, level: str = "window"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tüm fold'ların test tahminlerini birleştirir → (y_true, y_pred, proba)."""
    p_cols = proba_columns(n_classes)
    y = predictions["y_true"].to_numpy()
    p = predictions[p_cols].to_numpy()
    if level == "subject":
        _, y, p = aggregate_by_subject(y, p, predictions["subject_id"])
    return y, p.argmax(axis=1), p


def flag_suspicious(metrics: pd.DataFrame, threshold: float) -> list[str]:
    """Kural 4: ortalama doğruluk eşik üstündeyse uyarı mesajları döndürür."""
    msgs = []
    for (model, level), part in metrics.groupby(["model", "level"]):
        acc = part["accuracy"].mean()
        if acc >= threshold:
            msgs.append(
                f"[ŞÜPHELİ] {model} ({level}) ortalama doğruluk {acc:.3f} ≥ {threshold:.2f}: "
                "önce sızıntı arayın (folds.csv dışı bölme, Pipeline dışı ölçekleme, groups'suz iç CV)."
            )
    return msgs


# ---------------------------------------------------------------------------
# Kayıt / okuma
# ---------------------------------------------------------------------------
def save_metrics_csv(metrics: pd.DataFrame, metrics_dir: str | Path, model: str) -> Path:
    path = Path(metrics_dir) / f"metrics_{model}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = set(METRIC_COLUMNS) - set(metrics.columns)
    if missing:
        raise ValueError(f"metrics tablosunda eksik sütunlar: {sorted(missing)}")
    cols = [*METRIC_COLUMNS, *(["level"] if "level" in metrics.columns else [])]
    metrics[cols].to_csv(path, index=False)
    return path


def load_metrics(metrics_dir: str | Path) -> pd.DataFrame:
    """metrics_*.csv dosyalarını okur ve sözleşme şemasını denetler."""
    files = sorted(Path(metrics_dir).glob("metrics_*.csv"))
    if not files:
        raise FileNotFoundError(f"{metrics_dir} altında metrics_*.csv yok.")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        missing = set(METRIC_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{f.name} sözleşmeye uymuyor; eksik: {sorted(missing)}")
        if "level" not in df.columns:
            df["level"] = "window"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def comparison_table(metrics: pd.DataFrame, decimals: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model × düzey bazında ortalama ve standart sapma.

    Döndürür: (sayısal tablo [<metrik>_mean, <metrik>_std], biçimli tablo "ort ± std").
    std, fold'lar arası örneklem std'sidir (ddof=1).
    """
    grouped = metrics.groupby(["model", "level"], sort=True)[METRIC_NAMES]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    numeric = pd.concat([mean, std], axis=1)
    pretty = pd.DataFrame(index=numeric.index)
    for m in METRIC_NAMES:
        pretty[m] = [
            f"{a:.{decimals}f} ± {b:.{decimals}f}"
            for a, b in zip(numeric[f"{m}_mean"], numeric[f"{m}_std"].fillna(0.0))
        ]
    pretty["n_folds"] = grouped.size()
    return numeric, pretty


def table_to_markdown(df: pd.DataFrame) -> str:
    """Ek bağımlılık (tabulate) olmadan DataFrame → Markdown tablo (rapor için)."""
    flat = df.reset_index()
    header = "| " + " | ".join(map(str, flat.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in flat.columns) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in flat.itertuples(index=False)]
    return "\n".join([header, sep, *body]) + "\n"


# ---------------------------------------------------------------------------
# Grafikler
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, path: str | Path | None, dpi: int) -> None:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    normalize: str | None = "true",
    title: str | None = None,
    path: str | Path | None = None,
    dpi: int = 300,
    figsize: Sequence[float] = (6.5, 5.5),
) -> tuple[plt.Figure, np.ndarray]:
    """Karmaşıklık matrisi. Hücrelerde normalize oran ve ham sayı birlikte yazılır.

    normalize='true' → satır bazlı (her satır o sınıfın duyarlılık dağılımı).
    Döndürür: (fig, ham sayım matrisi)
    """
    k = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(k))
    if normalize == "true":
        denom = cm.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        denom = cm.sum(axis=0, keepdims=True)
    elif normalize is None:
        denom = np.ones_like(cm)
    else:
        raise ValueError("normalize ∈ {'true', 'pred', None}")
    with np.errstate(invalid="ignore", divide="ignore"):
        shown = np.where(denom > 0, cm / denom, 0.0)

    fig, ax = plt.subplots(figsize=tuple(figsize))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(k), yticks=np.arange(k),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Tahmin edilen sınıf", ylabel="Gerçek sınıf",
        title=title or "Karmaşıklık matrisi",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = shown.max() / 2 if shown.max() > 0 else 0.5
    for i in range(k):
        for j in range(k):
            txt = f"{shown[i, j]:.2f}\n(n={cm[i, j]})" if normalize else f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if shown[i, j] > thresh else "black")
    fig.tight_layout()
    _save(fig, path, dpi)
    return fig, cm


def plot_roc_curves(
    y_true: Sequence[int],
    proba: np.ndarray,
    class_names: Sequence[str],
    title: str | None = None,
    path: str | Path | None = None,
    dpi: int = 300,
    figsize: Sequence[float] = (6.5, 5.5),
) -> tuple[plt.Figure, dict[str, float]]:
    """Her sınıf için One-vs-Rest ROC eğrisi + makro ortalama eğri.

    Makro eğri: tüm sınıfların FPR noktalarının birleşimi üzerinde TPR'lerin
    doğrusal enterpolasyonla ortalaması. Döndürür: (fig, {sınıf: AUC, 'macro': AUC})
    """
    y_true = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    k = len(class_names)
    fig, ax = plt.subplots(figsize=tuple(figsize))
    aucs: dict[str, float] = {}
    curves = []
    for c in range(k):
        y_bin = (y_true == c).astype(int)
        if y_bin.min() == y_bin.max():  # sınıf yok ya da hepsi o sınıf
            continue
        fpr, tpr, _ = roc_curve(y_bin, p[:, c])
        aucs[class_names[c]] = float(auc(fpr, tpr))
        curves.append((fpr, tpr))
        ax.plot(fpr, tpr, lw=1.6, label=f"{class_names[c]} (AUC = {aucs[class_names[c]]:.3f})")
    if curves:
        grid = np.unique(np.concatenate([f for f, _ in curves]))
        mean_tpr = np.mean([np.interp(grid, f, t) for f, t in curves], axis=0)
        aucs["macro"] = float(auc(grid, mean_tpr))
        ax.plot(grid, mean_tpr, "k--", lw=2.2, label=f"Makro ortalama (AUC = {aucs['macro']:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls=":", label="Şans")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Yanlış pozitif oranı (1 − özgüllük)",
           ylabel="Doğru pozitif oranı (duyarlılık)", title=title or "ROC eğrileri (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path, dpi)
    return fig, aucs
