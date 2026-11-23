"""PhysioNet gaitndd v1.0.0 .ts dosyalarını indirir (yalnızca eksik olanları)."""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import requests

from gait.data.loader import parse_ts

log = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    pass


def record_names(groups: Mapping[str, int]) -> list[str]:
    """{'als': 13, ...} → ['als1', ..., 'als13', ...]"""
    return [f"{prefix}{i}" for prefix, n in groups.items() for i in range(1, int(n) + 1)]


def _fetch(record: str, base_urls: Iterable[str], timeout: float, retries: int) -> str:
    errors = []
    for base in base_urls:
        url = f"{base.rstrip('/')}/{record}.ts"
        for attempt in range(retries + 1):
            try:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                return r.text
            except requests.RequestException as e:  # ağ/HTTP hatası
                errors.append(f"{url} (deneme {attempt + 1}): {e}")
    raise DownloadError(f"{record}.ts indirilemedi:\n  " + "\n  ".join(errors))


def download_gaitndd(
    raw_dir: str | Path,
    groups: Mapping[str, int],
    base_urls: Sequence[str],
    time_column: str,
    channels: Sequence[str],
    timeout: float = 30,
    retries: int = 2,
    force: bool = False,
) -> Path:
    """Tüm kayıtları raw_dir'e indirir. İçerik .ts formatında değilse kaydedilmez."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for rec in record_names(groups):
        target = raw_dir / f"{rec}.ts"
        if target.exists() and target.stat().st_size > 0 and not force:
            continue
        text = _fetch(rec, base_urls, timeout, retries)
        parse_ts(text, time_column, channels)  # biçim doğrulaması (HTML hata sayfasını kaydetme)
        target.write_text(text, encoding="utf-8")
        log.info("indirildi: %s", target.name)
    return raw_dir
