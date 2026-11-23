"""Ortak test fikstürleri. Testler ağ kullanmaz ve data/ klasörüne yazmaz."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from gait.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def small_steps(cfg, tmp_path_factory):
    """Her sınıftan 6 sentetik denek → uzun adım tablosu (ağ yok)."""
    import copy

    from gait.data.loader import load_raw_dir
    from gait.data.synthetic import generate_dataset
    from gait.utils import class_names

    c = copy.deepcopy(cfg)
    c["data"]["groups"] = {k: 6 for k in c["data"]["groups"]}
    raw = tmp_path_factory.mktemp("raw")
    generate_dataset(raw, c, seed=0)
    return load_raw_dir(raw, class_names(c), c["data"]["time_column"], c["data"]["channels"])


def run_or_skip(fn, *args, **kwargs):
    """Ekip üyesi fonksiyonu henüz uygulamadıysa testi SKIP eder (CI kırmızıya dönmez).
    Uygulandığı anda test gerçekten çalışır ve sözleşmeyi zorunlu kılar."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"Henüz uygulanmadı: {fn.__module__}.{fn.__name__} ({e})")


@pytest.fixture(scope="session")
def small_windows(cfg, small_steps):
    """24 denekli küçük pencere seti (X_clean, X_filt, y, groups, meta)."""
    from gait.preprocessing.cleaning import clean_all
    from gait.preprocessing.filtering import filter_subjects
    from gait.preprocessing.windowing import make_windows

    d, pre, w = cfg["data"], cfg["preprocessing"], cfg["windowing"]
    clean, _ = clean_all(small_steps, pre, d["time_column"], d["channels"])
    filt = filter_subjects(clean, d["channels"], pre["filter"]["order"], pre["filter"]["cutoff"])
    return make_windows(clean, filt, d["channels"], d["time_column"], w["length"], w["overlap"])
