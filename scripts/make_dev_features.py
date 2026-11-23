"""GEÇİCİ: K2'nin features.parquet'i hazır olana kadar K3/K5'in bloke olmaması için
sözleşmeye uyan basit bir özellik tablosu (yalnızca kanal ortalaması ve std) üretir.

Çıktı: data/processed/features_dev.parquet  (features.parquet DEĞİL – üzerine yazmaz)
K3 geliştirirken: GAIT_FEATURES=data/processed/features_dev.parquet python scripts/run_step3.py
Rapora bu tabloyla elde edilen sonuç GİRMEZ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gait.utils import load_config, resolve  # noqa: E402


def main() -> None:
    cfg = load_config()
    out = cfg["output"]
    proc = resolve(out["processed_dir"])
    w = np.load(proc / out["windows_file"])
    meta = pd.read_csv(proc / out["windows_meta_file"])
    X = w["X_clean"]
    cols = {"subject_id": w["groups"], "label": w["y"], "window_idx": meta["window_idx"]}
    for j, ch in enumerate(w["channels"]):
        cols[f"{ch}__mean"] = X[:, :, j].mean(1)
        cols[f"{ch}__std"] = X[:, :, j].std(1, ddof=1)
    df = pd.DataFrame(cols)
    df.to_parquet(proc / "features_dev.parquet", index=False)
    print(f"features_dev.parquet: {df.shape}")


if __name__ == "__main__":
    main()
