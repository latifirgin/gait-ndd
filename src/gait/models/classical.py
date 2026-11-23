"""K3 – Klasik modeller: Random Forest (öncelikli), SVM, KNN.

Kurallar
--------
* Bölme yalnızca folds.csv ile (gait.evaluation.splits.iter_folds).
* StandardScaler ve (varsa) özellik seçimi Pipeline İÇİNDE → yalnızca eğitim fold'una fit.
* Hiperparametre araması iç CV ile: GridSearchCV(cv=get_inner_cv(...)).fit(..., groups=...)
  ``groups`` verilmezse iç CV pencere bazlı bölünür → sızıntı.
* Sınıf dengesizliği: class_weight='balanced' (RF, SVM); KNN'de yok → yorumla.
* SVM'de olasılık için probability=True (Platt ölçekleme, iç 5-katlı CV yapar; yavaş).

Çıktılar: results/predictions/predictions_<model>.csv, results/metrics/metrics_<model>.csv,
results/figures/rf_feature_importance.png, results/models/<model>.joblib (commit edilmez)
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

MODEL_NAMES = ("rf", "svm", "knn")


def build_pipeline(name: str, ccfg: Mapping[str, Any], seed: int) -> Pipeline:
    """Pipeline([('scaler', StandardScaler()), ('clf', <model>)])

    'scaler' adımı ilk sırada olmalı (sözleşme testi denetler). RF ölçeklemeye ihtiyaç
    duymaz ama tek tip Pipeline K5'in predict() kodunu basitleştirir.
    """
    raise NotImplementedError("K3: uygula")


def param_grid(name: str, ccfg: Mapping[str, Any]) -> dict[str, list]:
    """config.classical.models.<name> → {'clf__<param>': [...]}"""
    raise NotImplementedError("K3: uygula")


def run_cv(
    features: pd.DataFrame, folds: pd.DataFrame, name: str, cfg: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Dış CV döngüsü.

    Döndürür:
      predictions : gait.evaluation.metrics.make_predictions_frame çıktılarının birleşimi
                    (her pencere tam bir kez, kendi test fold'unda)
      best_params : fold başına seçilen hiperparametreler (rapor için)
    """
    raise NotImplementedError("K3: docs/evaluation.md'deki şablonu kullan")


def fit_final_model(features: pd.DataFrame, name: str, best_params: Mapping[str, Any],
                    cfg: Mapping[str, Any]) -> Pipeline:
    """Demo için TÜM veriyle eğitilen nihai model (CV skoru bundan raporlanmaz!)."""
    raise NotImplementedError("K3: uygula")


def feature_importance(pipe: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """RF Gini önemleri → DataFrame[feature, importance] azalan sırada.

    Not: Gini önemi korelasyonlu özelliklerde yanlıdır; sklearn.inspection.permutation_importance
    ile (test fold'unda) karşılaştırmak rapora değer katar.
    """
    raise NotImplementedError("K3: uygula")
