# Değerlendirme Protokolü (K1)

## Bölme
- `data/processed/folds.csv` → `subject_id, label, fold, n_windows` (64 satır, 5 fold, seed=42).
- Denek düzeyinde `StratifiedGroupKFold`; her denek **tam olarak bir** test fold'unda.
- Dosya bir kez üretilir ve repoya commit edilir. `--force` olmadan yeniden yazılmaz.
- Rastgele bölme, `train_test_split`, `KFold`, `cross_val_score(cv=5)` **yasak**.

## K3/K4 için CV döngüsü şablonu
```python
from gait.evaluation.splits import load_folds, iter_folds, get_inner_cv
from gait.evaluation.metrics import make_predictions_frame, evaluate_cv, save_metrics_csv

folds = load_folds("data/processed/folds.csv", n_splits=5)
preds = []
for k, tr, te in iter_folds(df["subject_id"], folds):          # sızıntı burada otomatik denetlenir
    search = GridSearchCV(pipe, grid, cv=get_inner_cv(3, 42), scoring="f1_macro")
    search.fit(X[tr], y[tr], groups=df["subject_id"].values[tr])  # groups ZORUNLU
    proba = search.predict_proba(X[te])
    preds.append(make_predictions_frame(df["subject_id"].values[te], df["window_idx"].values[te], k, y[te], proba))
preds = pd.concat(preds)
preds.to_csv("results/predictions/predictions_rf.csv", index=False)
metrics = pd.concat([evaluate_cv(preds, "rf", 4, lv) for lv in ("window", "subject")])
save_metrics_csv(metrics, "results/metrics", "rf")
```

## Dosya sözleşmeleri
| Dosya | Sütunlar | Yazan |
|---|---|---|
| `metrics_<model>.csv` | model, fold, accuracy, macro_f1, precision, recall, roc_auc, **level** | K3, K4 |
| `predictions_<model>.csv` | subject_id, window_idx, fold, y_true, y_pred, p_0..p_3 | K3, K4 |

`level` (window/subject) ve `predictions_<model>.csv`, birleşik karmaşıklık matrisi/ROC ve denek düzeyi metrikler için sözleşmenin parçasıdır.

## Raporlama kuralları
- Birincil metrik: **denek düzeyi Macro-F1** (pencere olasılıklarının ortalaması → argmax). Klinik karar kişi başınadır.
- Tablolar: 5 fold ortalama ± std (ddof=1). Şans referansı: `dummy` satırı.
- Ortalama doğruluk ≥ 0,95 → `flag_suspicious` uyarısı → önce sızıntı aranır.
