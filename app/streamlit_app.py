"""K5 – Klinik karar destek demosu.

Çalıştır:  pip install -e ".[app,ml]"  &&  streamlit run app/streamlit_app.py

Akış: .ts yükle → ham sinyal grafiği → predict() → tahmin + sınıf olasılıkları + SHAP.
predict() sözleşmesi: src/gait/models/predict.py
"""
from __future__ import annotations

import streamlit as st

from gait.data.loader import parse_ts
from gait.models import predict as pr
from gait.utils import load_config, resolve

cfg = load_config()
d = cfg["data"]
NAMES = dict(zip(cfg["evaluation"]["classes"], cfg["evaluation"]["class_display_names"]))

st.set_page_config(page_title="Yürüyüş Analizi – Karar Destek", layout="wide")
st.title("Yürüyüş analizi ile nörodejeneratif hastalık ön değerlendirmesi")
st.caption("Araştırma prototipidir; tanı aracı değildir. Karar her zaman klinisyene aittir.")

up = st.file_uploader("Yürüyüş kaydı (.ts, PhysioNet gaitndd formatı)", type=["ts", "txt"])
if up is None:
    st.info("Başlamak için bir .ts dosyası yükleyin.")
    st.stop()

try:
    steps = parse_ts(up.getvalue().decode("utf-8", errors="replace"), d["time_column"], d["channels"])
except ValueError as e:
    st.error(f"Dosya okunamadı: {e}")
    st.stop()

st.subheader("Kayıt")
st.write(f"{len(steps)} adım, {steps[d['time_column']].iloc[-1]:.0f} s")
st.line_chart(steps.set_index(d["time_column"])[["stride_l", "stride_r"]])  # TODO(K5): matplotlib/plotly + birimler

try:
    model = pr.load_model(resolve(cfg["classical"]["model_dir"]) / "rf.joblib")
    res = pr.predict(steps, model=model, cfg=cfg)
except NotImplementedError:
    st.warning("Model arayüzü henüz hazır değil (K3: predict.py). Arayüz iskeleti çalışıyor.")
    st.stop()
except FileNotFoundError:
    st.warning("Model dosyası yok: results/models/rf.joblib (K3: fit_final_model).")
    st.stop()

# TODO(K5): sonuç kartı, olasılık çubuk grafiği, SHAP şelale/çubuk grafiği, açıklama metni
c1, c2 = st.columns(2)
c1.metric("Tahmin", NAMES[res["label"]], f"{100 * res['proba'][res['label']]:.0f}% olasılık")
c2.bar_chart({NAMES[k]: v for k, v in res["proba"].items()})
st.subheader("Kararı en çok etkileyen özellikler (SHAP)")
st.dataframe(res["shap"])
