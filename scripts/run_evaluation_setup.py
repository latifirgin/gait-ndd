"""K1 – 24 Kas teslimatı: folds.csv üretimi + değerlendirme altyapısının uçtan uca denenmesi.

Ne yapar?
1. data/processed/windows.npz'yi okur (y, groups).
2. Denek bazlı StratifiedGroupKFold ile folds.csv üretir ve doğrular.
3. Şans düzeyi referansı olarak DummyClassifier'ı aynı fold'larla çalıştırır
   (K3/K4 için örnek CV döngüsü) → metrics_dummy.csv, predictions_dummy.csv.
4. Pencere + denek düzeyinde karmaşıklık matrisi ve ROC grafiklerini, kıyas tablosunu yazar.

Kullanım:
    python scripts/run_evaluation_setup.py [--config config/config.yaml] [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gait.evaluation.metrics import (  # noqa: E402
    comparison_table,
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
from gait.evaluation.splits import (  # noqa: E402
    encode_labels,
    fold_summary,
    iter_folds,
    load_folds,
    make_folds,
    save_folds,
)
from gait.utils import load_config, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="Var olan folds.csv'nin üzerine yaz.")
    args = ap.parse_args()

    ev = load_config(args.config)["evaluation"]
    paths = {k: resolve(v) for k, v in ev["paths"].items()}
    classes = ev["classes"]
    names = ev["class_display_names"]
    n_classes = len(classes)
    fig_cfg = ev["figures"]
    ext = fig_cfg["format"]

    # 1) Veri
    data = np.load(paths["windows"], allow_pickle=True)
    y = encode_labels(data["y"], classes)
    groups = np.asarray(data["groups"]).astype(str)
    source = str(data["source"]) if "source" in data.files else "bilinmiyor"
    print(f"[veri] kaynak={source}, {len(y)} pencere, {len(np.unique(groups))} denek")
    if source != "physionet":
        print("⚠ UYARI: veri gerçek PhysioNet verisi değil. Bu folds.csv'yi COMMIT ETMEYİN; "
              "gerçek veriyle yeniden üretin (--force).")

    # 2) Fold'lar (var olan dosya korunur: tüm ekip aynı bölmeyi kullanmalı)
    if paths["folds"].exists() and not args.force:
        folds = load_folds(paths["folds"], n_splits=ev["n_splits"])
        print(f"[folds] mevcut dosya kullanıldı: {paths['folds']}")
    else:
        folds = make_folds(y, groups, ev["n_splits"], ev["seed"], ev["min_test_subjects_per_class"])
        save_folds(folds, paths["folds"])
        print(f"[folds] yazıldı: {paths['folds']}")
    print(fold_summary(folds, names).to_string())

    # 3) Şans düzeyi referansı – K3/K4 için örnek döngü
    x_dummy = np.zeros((len(y), 1))
    window_idx = np.arange(len(y))
    preds = []
    for k, tr, te in iter_folds(groups, folds):
        clf = DummyClassifier(strategy=ev["dummy_strategy"], random_state=ev["seed"])
        clf.fit(x_dummy[tr], y[tr])
        proba = np.zeros((len(te), n_classes))
        proba[:, clf.classes_] = clf.predict_proba(x_dummy[te])  # eksik sınıf sütunlarını hizala
        preds.append(make_predictions_frame(groups[te], window_idx[te], k, y[te], proba))
    preds = pd.concat(preds, ignore_index=True)
    paths["predictions_dir"].mkdir(parents=True, exist_ok=True)
    preds.to_csv(paths["predictions_dir"] / "predictions_dummy.csv", index=False)

    metrics = pd.concat(
        [evaluate_cv(preds, "dummy", n_classes, level=lv, atol=ev["proba_atol"]) for lv in ("window", "subject")],
        ignore_index=True,
    )
    save_metrics_csv(metrics, paths["metrics_dir"], "dummy")

    # 4) Grafikler (birleşik fold-dışı tahminler)
    for lv in ("window", "subject"):
        yt, yp, pp = pooled_predictions(preds, n_classes, level=lv)
        fig, _ = plot_confusion_matrix(
            yt, yp, names, normalize=fig_cfg["confusion_normalize"],
            title=f"Dummy – {lv} düzeyi", dpi=fig_cfg["dpi"], figsize=fig_cfg["figsize"],
            path=paths["figures_dir"] / f"confusion_dummy_{lv}.{ext}",
        )
        plt.close(fig)
        fig, _ = plot_roc_curves(
            yt, pp, names, title=f"Dummy – {lv} düzeyi", dpi=fig_cfg["dpi"], figsize=fig_cfg["figsize"],
            path=paths["figures_dir"] / f"roc_dummy_{lv}.{ext}",
        )
        plt.close(fig)

    # 5) Kıyas tablosu (results/metrics altındaki tüm modeller)
    all_metrics = load_metrics(paths["metrics_dir"])
    numeric, pretty = comparison_table(all_metrics)
    paths["tables_dir"].mkdir(parents=True, exist_ok=True)
    numeric.to_csv(paths["tables_dir"] / "comparison.csv")
    (paths["tables_dir"] / "comparison.md").write_text(table_to_markdown(pretty), encoding="utf-8")
    print("\n" + pretty.to_string())
    for msg in flag_suspicious(all_metrics, ev["suspicious_accuracy"]):
        print(msg)


if __name__ == "__main__":
    main()
