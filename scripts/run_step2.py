"""Adım 2 (K2): windows.npz → data/processed/features.parquet

Kullanım: python scripts/run_step2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gait.features.build import build_feature_table, validate_feature_table  # noqa: E402
from gait.utils import load_config, resolve  # noqa: E402


def main() -> None:
    cfg = load_config()
    out = cfg["output"]
    proc = resolve(out["processed_dir"])
    w = np.load(proc / out["windows_file"], allow_pickle=False)
    meta = pd.read_csv(proc / out["windows_meta_file"])
    fcfg = cfg["features"]
    df = build_feature_table(w[fcfg["input"]], w["y"], w["groups"], meta["window_idx"].to_numpy(),
                             list(w["channels"]), fcfg)
    validate_feature_table(df, n_rows=len(w["y"]))
    path = resolve(fcfg["output_file"])
    df.to_parquet(path, index=False)
    print(f"{path} yazıldı: {df.shape[0]} pencere × {df.shape[1] - 3} özellik (kaynak={w['source']})")


if __name__ == "__main__":
    main()
