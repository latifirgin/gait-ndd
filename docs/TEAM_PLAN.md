# Ekip planı (2 kişi)

**Ekip:** Latif (lider) + Serhat · **Ara entegrasyon** 1 Ara · **Kod dondurma** 6 Ara · **Teslim** 12 Ara 2026

Proje 5 kişilik planlanmıştı; 25 Kas itibarıyla 2 kişiyle yürüyor. K1–K5 görevlerinin hepsi korunuyor,
iki kişiye dağıtılıyor. Bonus işler (Optuna, 1D-CNN, X_clean/X_filt ablation, LSTM için SHAP) kapsamdan çıkarıldı.

| Görev | Eski kod | Sahibi | Son tarih | Issue |
|---|---|---|---|---|
| Veri boru hattı, folds.csv, metrikler, kıyas tablosu | K1 | Latif | 26 Kas | [K1](tasks/K1.md) |
| Özellik çıkarımı (features.parquet) | K2 | Latif | 29 Kas | [K2](tasks/K2.md) |
| RF / SVM / KNN + predict() + rf.joblib | K3 | Serhat | 3 Ara | [K3](tasks/K3.md) |
| LSTM + SHAP (RF için) | K4 | Serhat | 5 Ara | [K4](tasks/K4.md) |
| Streamlit demo | K5 | Latif | 6 Ara | [K5](tasks/K5.md) |
| Sunum slaytları | – | Latif + Serhat | 9 Ara | – |
| Rapor (editör Latif; model + SHAP bölümleri Serhat) | – | Latif + Serhat | 11 Ara | – |

## Bağımlılıklar

```
Latif: windows.npz + folds.csv (26 Kas) ──> Latif: features.parquet (29 Kas) ──┐
                                                                               ├──> Serhat: RF/SVM/KNN + predict() (3 Ara) ──> Latif: demo (6 Ara)
Serhat: features_dev.parquet ile RF geliştirme (25–28 Kas) ────────────────────┘                │
Serhat: LSTM + SHAP (3–5 Ara) ─────────────────────────────────────────────────────────────────┴──> Latif: kıyas tablosu + rapor
```

- Serhat, Latif'in özellik tablosunu beklemeden `scripts/make_dev_features.py` ile hemen başlar.
- Latif, `predict()` hazır olana kadar demoyu sözleşmeye uyan sahte bir `predict` ile geliştirir.
- LSTM 5 Ara'ya yetişmezse ana sonuç RF olur; LSTM denendiği hâliyle dürüstçe raporlanır.

## Takvim

| Tarih | Kilometre taşı |
|---|---|
| 26 Kas | Gerçek veriyle `folds.csv` main'de (Latif) |
| 29 Kas | `features.parquet` main'de (Latif) |
| 1 Ara | **Ara entegrasyon:** RF uçtan uca, ilk kıyas tablosu |
| 3 Ara | RF/SVM/KNN + `predict()` + `rf.joblib` (Serhat) |
| 5 Ara | LSTM + SHAP (Serhat) |
| 6 Ara | **Kod dondurma:** demo main'de (Latif); sonrasında yalnızca hata düzeltme |
| 9 Ara | Sunum, prova |
| 11 Ara | Rapor son hâli |
| 12 Ara | Teslim: repo etiketi `v1.0` |

## Çalışma düzeni
- Her PR'ı diğer kişi onaylar: Latif Serhat'ınkini, Serhat Latif'inkini.
- Haftada 2 kısa görüşme (15 dk): ne bitti, ne bloke, sözleşme değişti mi.
