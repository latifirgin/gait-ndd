""".ts dosyalarını okuma ve tüm denekleri tek bir uzun tabloda birleştirme.

gaitndd .ts formatı: her satır bir adım (stride), boşluk/sekme ayrılmış 13 sayısal
sütun: geçen süre + 12 kanal (bkz. config.data.channels).
"""
from __future__ import annotations

import io
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

N_COLUMNS = 13
_RECORD_RE = re.compile(r"^([a-z]+)(\d+)$")


def column_names(time_column: str, channels: Sequence[str]) -> list[str]:
    cols = [time_column, *channels]
    if len(cols) != N_COLUMNS:
        raise ValueError(f"{N_COLUMNS} sütun bekleniyor (zaman + 12 kanal); config'de {len(cols)} var.")
    return cols


def parse_ts(source: str | Path, time_column: str, channels: Sequence[str]) -> pd.DataFrame:
    """Bir .ts dosyasını (yol) ya da metnini okur.

    Sayısal olmayan satırlar atılır; 13 sütun yoksa ValueError.
    """
    text = Path(source).read_text(encoding="utf-8", errors="replace") if _is_path(source) else str(source)
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue  # başlık/yorum satırı
        if len(values) != N_COLUMNS:
            raise ValueError(f"Satırda {len(values)} sütun var, {N_COLUMNS} bekleniyordu: '{line[:80]}'")
        rows.append(values)
    if not rows:
        raise ValueError("Sayısal satır bulunamadı (dosya HTML/boş olabilir).")
    return pd.DataFrame(np.asarray(rows), columns=column_names(time_column, channels))


def _is_path(source: str | Path) -> bool:
    if isinstance(source, Path):
        return True
    return "\n" not in source and len(source) < 4096 and Path(source).exists()


def label_from_record(record: str, class_names: Sequence[str]) -> int:
    """'hunt12' → class_names.index('hunt')."""
    m = _RECORD_RE.match(record.lower())
    if not m or m.group(1) not in class_names:
        raise ValueError(f"Kayıt adından sınıf çıkarılamadı: '{record}' (beklenen önekler: {list(class_names)})")
    return list(class_names).index(m.group(1))


def load_raw_dir(
    raw_dir: str | Path, class_names: Sequence[str], time_column: str, channels: Sequence[str]
) -> pd.DataFrame:
    """Klasördeki tüm <önek><n>.ts dosyalarını uzun tabloya çevirir.

    Sütunlar: subject_id, label, step, <time_column>, <12 kanal>
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.ts"), key=lambda p: _sort_key(p.stem))
    if not files:
        raise FileNotFoundError(f"{raw_dir} içinde .ts dosyası yok.")
    frames = []
    for f in files:
        df = parse_ts(f, time_column, channels)
        df.insert(0, "step", np.arange(len(df), dtype=np.int64))
        df.insert(0, "label", label_from_record(f.stem, class_names))
        df.insert(0, "subject_id", f.stem.lower())
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _sort_key(stem: str) -> tuple[str, int]:
    m = _RECORD_RE.match(stem.lower())
    return (m.group(1), int(m.group(2))) if m else (stem, 0)


def write_ts(df: pd.DataFrame, path: str | Path, time_column: str, channels: Sequence[str]) -> Path:
    """DataFrame'i gaitndd .ts formatında yazar (sentetik veri için)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    np.savetxt(buf, df[column_names(time_column, channels)].to_numpy(), fmt="%.4f", delimiter="\t")
    path.write_text(buf.getvalue(), encoding="utf-8")
    return path
