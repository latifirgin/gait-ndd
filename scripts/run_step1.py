"""Adım 1: veri temini → temizleme → filtreleme → pencereleme.

Çıktılar (data/processed/):
  windows.npz        X_clean, X_filt (N, 32, 12), y, groups, channels, class_names, source
  windows_meta.csv   window_idx, subject_id, label, start_step, end_step, t_start, t_end
  steps_clean.parquet  temizlenmiş adım tablosu (K2/K5 için)
  cleaning_report.csv  denek başına silinen/enterpole edilen adım sayıları

Kullanım:
  python scripts/run_step1.py               # gerçek veri; indirilemezse sentetik (config izin verirse)
  python scripts/run_step1.py --synthetic   # doğrudan sentetik (çevrimdışı geliştirme)
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gait.data.download import DownloadError, download_gaitndd  # noqa: E402
from gait.data.loader import load_raw_dir  # noqa: E402
from gait.data.synthetic import generate_dataset  # noqa: E402
from gait.preprocessing.cleaning import clean_all  # noqa: E402
from gait.preprocessing.filtering import filter_subjects  # noqa: E402
from gait.preprocessing.windowing import make_windows  # noqa: E402
from gait.utils import class_names, load_config, resolve, set_seed  # noqa: E402

log = logging.getLogger("step1")


def get_raw_dir(cfg: dict, synthetic: bool, force_download: bool) -> tuple[Path, str]:
    d = cfg["data"]
    if not synthetic:
        try:
            raw = download_gaitndd(
                resolve(d["raw_dir"]), d["groups"], d["base_urls"], d["time_column"], d["channels"],
                timeout=d["timeout_s"], retries=d["retries"], force=force_download,
            )
            return raw, "physionet"
        except DownloadError as e:
            if not d["allow_synthetic_fallback"]:
                raise
            log.warning("PhysioNet indirilemedi → SENTETİK veri kullanılacak.\n%s", str(e).splitlines()[0])
    syn_dir = resolve(cfg["synthetic"]["synthetic_dir"])
    shutil.rmtree(syn_dir, ignore_errors=True)
    generate_dataset(syn_dir, cfg, seed=cfg["project"]["seed"])
    return syn_dir, "synthetic"


def plot_example(steps_raw, steps_clean, steps_filt, cfg, path: Path) -> None:
    d, pre = cfg["data"], cfg["preprocessing"]
    sid = steps_clean.loc[steps_clean["label"] == steps_clean["label"].max(), "subject_id"].iloc[0]
    ch = pre["outlier_channels"][0]
    raw = steps_raw[steps_raw.subject_id == sid]
    cl = steps_clean[steps_clean.subject_id == sid]
    fl = steps_filt[steps_filt.subject_id == sid]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(raw[d["time_column"]], raw[ch], color="0.7", lw=1, label="ham")
    ax.plot(cl[d["time_column"]], cl[ch], lw=1.2, label="temiz (IQR + enterpolasyon)")
    ax.plot(fl[d["time_column"]], fl[ch], lw=1.6, label="filtreli")
    interp = cl[cl["was_interpolated"]]
    ax.scatter(interp[d["time_column"]], interp[ch], s=18, color="crimson", zorder=3, label="enterpole adım")
    ax.axvline(pre["trim_start_s"], ls="--", color="k", lw=0.8)
    ax.set(xlabel="Geçen süre (s)", ylabel=f"{ch} (s)", title=f"Ön işleme örneği – {sid}")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg["evaluation"]["figures"]["dpi"])
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    d, pre, win, out = cfg["data"], cfg["preprocessing"], cfg["windowing"], cfg["output"]
    names = class_names(cfg)

    raw_dir, source = get_raw_dir(cfg, args.synthetic, args.force_download)
    steps_raw = load_raw_dir(raw_dir, names, d["time_column"], d["channels"])
    log.info("kaynak=%s, %d denek, %d adım", source, steps_raw.subject_id.nunique(), len(steps_raw))

    steps_clean, report = clean_all(steps_raw, pre, d["time_column"], d["channels"])
    steps_filt = filter_subjects(steps_clean, d["channels"], pre["filter"]["order"], pre["filter"]["cutoff"])
    w = make_windows(steps_clean, steps_filt, d["channels"], d["time_column"], win["length"], win["overlap"])

    proc = resolve(out["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        proc / out["windows_file"],
        X_clean=w["X_clean"], X_filt=w["X_filt"], y=w["y"], groups=w["groups"],
        channels=np.array(d["channels"]), class_names=np.array(names), source=np.array(source),
    )
    w["meta"].to_csv(proc / out["windows_meta_file"], index=False)
    steps_clean.to_parquet(proc / out["step_table_file"], index=False)
    report.to_csv(proc / "cleaning_report.csv", index=False)
    plot_example(steps_raw, steps_clean, steps_filt, cfg, resolve("results/figures/step1_preprocessing_example.png"))

    # Özet
    meta = w["meta"]
    summary = meta.groupby("label").agg(denek=("subject_id", "nunique"), pencere=("window_idx", "size"))
    summary.index = [names[i] for i in summary.index]
    print("\n=== Adım 1 özeti ===")
    print(f"Kaynak: {source.upper()}" + ("  ⚠ SENTETİK – sonuçlar rapora girmez!" if source == "synthetic" else ""))
    print(summary.to_string())
    print(f"X shape: {w['X_clean'].shape}  |  geçersiz adım oranı (medyan): "
          f"%{report['pct_invalid'].median():.2f}  |  atılan denek: {int(report['dropped'].sum())}")


if __name__ == "__main__":
    main()
