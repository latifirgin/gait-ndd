"""Adım 3 (K3): features.parquet + folds.csv → RF/SVM/KNN CV sonuçları.

Kullanım: python scripts/run_step3.py [--models rf svm knn]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gait.evaluation.metrics import evaluate_cv, save_metrics_csv  # noqa: E402
from gait.evaluation.splits import load_folds  # noqa: E402
from gait.features.build import validate_feature_table  # noqa: E402
from gait.models.classical import MODEL_NAMES, run_cv  # noqa: E402
from gait.utils import load_config, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    args = ap.parse_args()
    cfg = load_config()
    ev = cfg["evaluation"]
    # GAIT_FEATURES ile geçici tablo (features_dev.parquet) kullanılabilir
    feats = pd.read_parquet(resolve(os.environ.get("GAIT_FEATURES", cfg["features"]["output_file"])))
    validate_feature_table(feats)
    folds = load_folds(resolve(ev["paths"]["folds"]), ev["n_splits"])
    n_classes = len(ev["classes"])
    for name in args.models:
        preds, best = run_cv(feats, folds, name, cfg)
        pdir = resolve(ev["paths"]["predictions_dir"])
        pdir.mkdir(parents=True, exist_ok=True)
        preds.to_csv(pdir / f"predictions_{name}.csv", index=False)
        (pdir / f"best_params_{name}.json").write_text(json.dumps(best, indent=2, default=str))
        m = pd.concat([evaluate_cv(preds, name, n_classes, lv) for lv in ("window", "subject")])
        save_metrics_csv(m, resolve(ev["paths"]["metrics_dir"]), name)
        print(m.groupby("level")[["accuracy", "macro_f1", "roc_auc"]].mean().round(3).assign(model=name))
    # TODO(K3): RF feature importance grafiği + fit_final_model → results/models/rf.joblib


if __name__ == "__main__":
    main()
