"""Adım 4 (K4): windows.npz + folds.csv → LSTM CV sonuçları + SHAP grafikleri.

Kurulum: pip install -e ".[dl,ml]"   Kullanım: python scripts/run_step4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gait.evaluation.metrics import evaluate_cv, save_metrics_csv  # noqa: E402
from gait.evaluation.splits import load_folds  # noqa: E402
from gait.models.lstm import run_cv  # noqa: E402
from gait.utils import load_config, resolve  # noqa: E402


def main() -> None:
    cfg = load_config()
    ev, out = cfg["evaluation"], cfg["output"]
    w = dict(np.load(resolve(out["processed_dir"]) / out["windows_file"]))
    folds = load_folds(resolve(ev["paths"]["folds"]), ev["n_splits"])
    preds = run_cv(w, folds, cfg)
    pdir = resolve(ev["paths"]["predictions_dir"])
    pdir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(pdir / "predictions_lstm.csv", index=False)
    m = pd.concat([evaluate_cv(preds, "lstm", len(ev["classes"]), lv) for lv in ("window", "subject")])
    save_metrics_csv(m, resolve(ev["paths"]["metrics_dir"]), "lstm")
    # TODO(K4): RF ve LSTM için SHAP grafikleri → results/figures/shap_*.png


if __name__ == "__main__":
    main()
