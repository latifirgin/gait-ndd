# Katkı kuralları

## Değişmez kurallar
1. **Veri sızıntısı yasak.** Rastgele bölme yok; yalnızca `data/processed/folds.csv` + `gait.evaluation.splits.iter_folds`.
   İç CV'de `GridSearchCV(cv=get_inner_cv(...)).fit(X, y, groups=...)`.
2. Ölçekleme / özellik seçimi yalnızca eğitim fold'una fit edilir (sklearn `Pipeline` içinde).
3. Parametreler `config/config.yaml`'da; kodda sabit sayı yok. **Her yeni fonksiyona test.**
4. 64 denekte %95+ doğruluk şüphelidir: önce sızıntı aranır.
5. `main` korumalı: her iş ayrı dalda → PR → 1 onay → CI yeşil → merge.

## Günlük akış

```bash
git switch main && git pull                 # güncel main
git switch -c feat/features                 # kendi dalın (ilk seferde)
# ... çalış ...
pytest -q && ruff check src scripts tests app
git add -A && git commit -m "features: spektral entropi eklendi"
git push -u origin feat/features            # sonra GitHub'da PR aç
```

Uzun süren dalı haftada en az iki kez güncelle: `git switch feat/xxx && git merge main`.

**Dal adları:** `feat/<konu>`, `fix/<konu>`, `docs/<konu>`.
**Commit mesajı:** `<alan>: <ne yapıldı>` (ör. `classical: RF iç CV groups ile`).
**PR:** küçük tut (< ~400 satır), şablondaki kontrol listesini doldur, issue'yu bağla (`Closes #3`).

## Sözleşmeler (değiştirmek için ekipte onay + K1)

| Dosya / fonksiyon | Şema | Üreten → Tüketen |
|---|---|---|
| `windows.npz` | `X_clean`, `X_filt` (N,32,12), `y`, `groups`, `channels`, `class_names`, `source` | K1 → K2, K4 |
| `folds.csv` | `subject_id, label, fold, n_windows` | K1 → K3, K4 |
| `features.parquet` | `subject_id, label, window_idx, <kanal>__<özellik>…` | K2 → K3, K5 |
| `predictions_<model>.csv` | `subject_id, window_idx, fold, y_true, y_pred, p_0..p_3` | K3, K4 → K1 |
| `metrics_<model>.csv` | `model, fold, accuracy, macro_f1, precision, recall, roc_auc, level` | K3, K4 → K1 |
| `predict(steps)` | `{label, proba, shap, n_windows}` | K3 (+K4) → K5 |

Sözleşmeler `tests/test_contract_k*.py` ile zorunlu kılınır. Fonksiyonun `NotImplementedError` fırlattığı
sürece ilgili test **SKIP** olur; uyguladığın anda gerçekten çalışır.

## Veri ve sonuçlar
- `data/` commit edilmez (istisna: `folds.csv`). Herkes `python scripts/run_step1.py` ile üretir.
- `results/metrics`, `results/predictions`, `results/tables`, `results/figures` commit edilir; `results/models` edilmez.
- Rapora/sunuma giren her sayı **gerçek veriyle** (`source=physionet`) üretilmiş olmalı.
