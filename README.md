# Yürüyüş Analizi ile Nörodejeneratif Hastalık Sınıflandırma

Ayakkabı altı kuvvet sensörlerinden çıkarılan adım-adım yürüyüş serilerinden bir kişinin
**Sağlıklı / Parkinson / Huntington / ALS** olduğunu tahmin eden ve kararını **SHAP** ile
açıklayan klinik karar destek prototipi.

> Araştırma prototipidir; tanı aracı değildir.

## Veri

[PhysioNet Gait in Neurodegenerative Disease Database v1.0.0](https://physionet.org/content/gaitndd/1.0.0/):
64 denek (16 Kontrol, 15 Parkinson, 20 Huntington, 13 ALS), ~5 dk yürüyüş. `.ts` dosyalarında her
satır bir adım; 13 sütun (geçen süre + 12 kanal: sol/sağ adım, salınım, duruş süreleri ve yüzdeleri, çift destek).
Veri repoya **girmez**; `run_step1.py` indirir. İnternet yoksa **sentetik** veri üretilir
(yalnızca geliştirme içindir; `windows.npz` içindeki `source` alanı bunu kaydeder, sonuçlar rapora girmez).

## Kurulum

```bash
git clone https://github.com/latifirgin/gait-ndd.git
cd gait-ndd
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"            # temel + test araçları
# K3/K4: pip install -e ".[dev,ml]"   K4: + ".[dl]"   K5: + ".[app]"
pytest -q                          # hepsi geçmeli (uygulanmamış görevler SKIP görünür)
```

## Boru hattı

| Adım | Komut | Çıktı | Sahibi |
|---|---|---|---|
| 1. Veri + ön işleme | `python scripts/run_step1.py` | `data/processed/windows.npz`, `windows_meta.csv` | K1 |
| Bölme + değerlendirme | `python scripts/run_evaluation_setup.py` | `data/processed/folds.csv`, şans referansı | K1 |
| 2. Özellikler | `python scripts/run_step2.py` | `data/processed/features.parquet` | K2 |
| 3. Klasik modeller | `python scripts/run_step3.py` | `results/metrics/metrics_{rf,svm,knn}.csv` | K3 |
| 4. LSTM + SHAP | `python scripts/run_step4.py` | `results/metrics/metrics_lstm.csv`, SHAP figürleri | K4 |
| Demo | `streamlit run app/streamlit_app.py` | – | K5 |

Ön işleme: ilk 20 s kırpma → fizyolojik sınır (adım 0,5–3 s) → denek içi IQR (k=3, yalnızca adım süresi
kanalları; aykırı adımın tüm kanalları geçersiz) → doğrusal enterpolasyon → sıfır fazlı Butterworth
(derece 2, kesim 0,35, eksen = adım indeksi) → 32 adımlık pencere, %50 örtüşme. Filtresiz (`X_clean`) ve
filtreli (`X_filt`) seriler birlikte saklanır, çünkü filtre CoV'yi düşürür.

## Değerlendirme protokolü

Kişi bazlı 5 katlı `StratifiedGroupKFold` (seed=42) → tek bir `folds.csv`; **aynı denek asla hem eğitimde
hem testte olmaz**. Birincil metrik: denek düzeyi Macro-F1. Ayrıntı: [docs/evaluation.md](docs/evaluation.md).

## Repo yapısı

```
config/config.yaml        tüm parametreler (kodda sabit sayı yok)
src/gait/
  data/                   indirme, .ts okuma, sentetik yedek
  preprocessing/          temizleme, filtre, pencereleme
  evaluation/             folds, metrikler, grafikler          (K1)
  features/               zaman/frekans özellikleri            (K2)
  models/                 classical (K3), lstm (K4), predict (K3→K5)
  explain/                SHAP                                  (K4)
scripts/                  run_step*.py, setup_github.sh
tests/                    birim testleri + test_contract_k*.py (ekip kabul testleri)
app/                      Streamlit demo                        (K5)
docs/                     TEAM_PLAN, evaluation, features, tasks/
```

## Ekip

Görevler, takvim ve sözleşmeler: [docs/TEAM_PLAN.md](docs/TEAM_PLAN.md) · Çalışma kuralları: [CONTRIBUTING.md](CONTRIBUTING.md)
