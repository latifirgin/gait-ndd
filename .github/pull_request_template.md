## Ne değişti?
<!-- 1-3 cümle. İlgili issue: Closes #... -->

## Kontrol listesi
- [ ] `pytest -q` yerelde geçiyor, `ruff check src scripts tests app` temiz
- [ ] Yeni her fonksiyon için test var
- [ ] Parametreler `config/config.yaml`'da (kodda sabit sayı yok)
- [ ] Bölme yalnızca `folds.csv` + `iter_folds` ile; iç CV'de `groups=` verildi
- [ ] Ölçekleme / özellik seçimi Pipeline içinde (yalnızca eğitim fold'una fit)
- [ ] Sözleşme dosyalarının (features.parquet, metrics_*.csv, predictions_*.csv, predict()) şeması değişmedi ya da değişiklik ekipte onaylandı
- [ ] Sonuçlar **gerçek PhysioNet** verisiyle (windows.npz `source=physionet`)

## Sonuçlar (varsa)
| model | düzey | accuracy | macro_f1 | roc_auc |
|---|---|---|---|---|
