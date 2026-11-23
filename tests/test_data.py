"""Veri okuma, indirme yardımcıları ve sentetik üretim testleri."""
from __future__ import annotations

import numpy as np
import pytest

from gait.data import download as dl
from gait.data.download import DownloadError, record_names
from gait.data.loader import label_from_record, parse_ts
from gait.utils import class_names

ROW = "20.0 1.10 1.12 0.40 0.41 36.4 36.6 0.70 0.71 63.6 63.4 0.28 25.5"


def test_parse_ts_text_skips_headers(cfg):
    d = cfg["data"]
    df = parse_ts(f"# başlık\n{ROW}\n{ROW}\n", d["time_column"], d["channels"])
    assert df.shape == (2, 13)
    assert df[d["time_column"]].iloc[0] == pytest.approx(20.0)


def test_parse_ts_rejects_wrong_column_count(cfg):
    d = cfg["data"]
    with pytest.raises(ValueError, match="sütun"):
        parse_ts("1 2 3\n", d["time_column"], d["channels"])
    with pytest.raises(ValueError, match="Sayısal satır"):
        parse_ts("<html>Not found</html>\n", d["time_column"], d["channels"])


def test_label_from_record(cfg):
    names = class_names(cfg)
    assert label_from_record("control3", names) == names.index("control")
    assert label_from_record("ALS13", names) == names.index("als")
    with pytest.raises(ValueError):
        label_from_record("patient1", names)


def test_record_names_match_gaitndd(cfg):
    recs = record_names(cfg["data"]["groups"])
    assert len(recs) == 64 and len(set(recs)) == 64
    assert {"als1", "als13", "control16", "hunt20", "park15"} <= set(recs)


def test_download_raises_without_network(cfg, tmp_path, monkeypatch):
    class Boom:
        def raise_for_status(self):
            raise dl.requests.HTTPError("403")

    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: Boom())
    d = cfg["data"]
    with pytest.raises(DownloadError):
        dl.download_gaitndd(tmp_path, {"als": 1}, ["https://example.invalid/"], d["time_column"], d["channels"], retries=0)


def test_download_saves_valid_file(cfg, tmp_path, monkeypatch):
    class Ok:
        text = f"{ROW}\n{ROW}\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: Ok())
    d = cfg["data"]
    dl.download_gaitndd(tmp_path, {"als": 2}, ["https://x/"], d["time_column"], d["channels"])
    assert sorted(p.name for p in tmp_path.glob("*.ts")) == ["als1.ts", "als2.ts"]


def test_synthetic_dataset_is_physiological(cfg, small_steps):
    d = cfg["data"]
    assert small_steps["subject_id"].nunique() == 24
    assert set(small_steps["label"]) == {0, 1, 2, 3}
    s = small_steps
    np.testing.assert_allclose(s["swing_l"] + s["stance_l"], s["stride_l"], rtol=1e-3)
    np.testing.assert_allclose(s["swing_pct_l"] + s["stance_pct_l"], 100, atol=1e-2)
    assert s.groupby("subject_id")[d["time_column"]].apply(lambda t: t.is_monotonic_increasing).all()
    med = s.groupby("subject_id")["stride_l"].median()
    assert med.between(0.8, 2.0).all()


def test_synthetic_class_variability_order(small_steps):
    """HD'nin adım değişkenliği kontrolden yüksek olmalı (literatürle uyumlu)."""
    cov = small_steps.groupby(["label", "subject_id"])["stride_l"].apply(
        lambda x: x.clip(upper=x.quantile(0.95)).std() / x.mean()).groupby("label").median()
    assert cov[2] > cov[0]
