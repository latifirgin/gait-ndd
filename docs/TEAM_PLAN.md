# Ekip planı

**Başlangıç** 23 Kas 2026 · **Ara entegrasyon** 1 Ara · **Kod dondurma** 6 Ara · **Teslim** 12 Ara 2026

| Kişi | İsim | Rol | Teslimat | Tarih | Görev |
|---|---|---|---|---|---|
| K1 | Latif | Lider, veri, değerlendirme, rapor editörü | `windows.npz`, `folds.csv`, `evaluate()`, grafikler, kıyas tablosu | 24 Kas | [K1](tasks/K1.md) |
| K2 | Süleyman | Özellik mühendisi | `features.parquet`, `docs/features.md` | 30 Kas | [K2](tasks/K2.md) |
| K3 | Serhat | Klasik ML | RF/SVM/KNN metrikleri, önem grafiği, `predict()` | 5 Ara | [K3](tasks/K3.md) |
| K4 | Ahmet | Derin öğrenme + XAI | LSTM metrikleri, SHAP grafikleri | 5 Ara | [K4](tasks/K4.md) |
| K5 | Halil | Demo + sunum | Streamlit (6 Ara), slaytlar (9 Ara) | 6 / 9 Ara | [K5](tasks/K5.md) |

## Bağımlılık grafiği

```
K1: windows.npz ──┬──> K2: features.parquet ──> K3: RF/SVM/KNN ──> predict() ──> K5: demo
                  │                                   │
K1: folds.csv ────┼───────────────────────────────────┤
                  └──> K4: LSTM ──────────────────────┴──> SHAP ──> K5
K3, K4: metrics_*.csv ──> K1: kıyas tablosu ──> rapor + sunum
```

Kimse beklemesin:
- **K2** sentetik ya da gerçek `windows.npz` ile hemen başlar (`run_step1.py`).
- **K3** `scripts/make_dev_features.py` ile geçici özellik tablosu kullanır.
- **K4** LSTM için yalnızca `windows.npz` + `folds.csv`'ye ihtiyaç duyar; RF-SHAP'ı K3'ün modeliyle sonra bağlar.
- **K5** `predict()` sözleşmesine uyan sahte bir fonksiyonla arayüzü geliştirir.

## Takvim

| Tarih | Kilometre taşı |
|---|---|
| 24 Kas | Repo hazır, herkes kurulumu yaptı ve `pytest` yeşil; gerçek veriyle `folds.csv` main'de |
| 30 Kas | K2 PR'ı merge: `features.parquet` |
| 1 Ara | **Ara entegrasyon:** RF uçtan uca, ilk kıyas tablosu (K1 toplantı) |
| 5 Ara | K3 + K4 PR'ları merge: tüm metrikler + SHAP |
| 6 Ara | **Kod dondurma:** K5 demo merge; sonrasında yalnızca hata düzeltme |
| 9 Ara | Slaytlar, prova |
| 12 Ara | Teslim: repo etiketi `v1.0`, rapor, sunum |

## Toplantı
Haftada 2 kısa senkron (15 dk): ne bitti / ne bloke / sözleşme değişikliği var mı.
