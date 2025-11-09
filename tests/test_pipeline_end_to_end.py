# tests/test_pipeline_end_to_end.py

import sys
from pathlib import Path
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Asegurar raíz del proyecto en el path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from german_credit_ml.modeling.train import (
    DataModule,
    PreprocessorFactory,
    TrainConfig,
    Evaluator,
)
from german_credit_ml.modeling.evaluate import evaluate_model
from german_credit_ml.modeling.predict import run_inference


def _build_synthetic_credit_csv(path: Path, n_rows: int = 80):
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "status": rng.integers(1, 5, size=n_rows),
            "duration": rng.integers(6, 60, size=n_rows),
            "credit_history": rng.integers(1, 5, size=n_rows),
            "purpose": rng.integers(1, 5, size=n_rows),
            "amount": rng.integers(500, 10000, size=n_rows),
            "savings": rng.integers(1, 5, size=n_rows),
            "employment_duration": rng.integers(1, 5, size=n_rows),
            "installment_rate": rng.integers(1, 5, size=n_rows),
            "personal_status_sex": rng.integers(1, 5, size=n_rows),
            "other_debtors": rng.integers(1, 3, size=n_rows),
            "present_residence": rng.integers(1, 5, size=n_rows),
            "property": rng.integers(1, 4, size=n_rows),
            "age": rng.integers(18, 70, size=n_rows),
            "other_installment_plans": rng.integers(1, 4, size=n_rows),
            "housing": rng.integers(1, 4, size=n_rows),
            "number_credits": rng.integers(1, 4, size=n_rows),
            "job": rng.integers(1, 4, size=n_rows),
            "people_liable": rng.integers(1, 3, size=n_rows),
            "telephone": rng.integers(1, 3, size=n_rows),
            "foreign_worker": rng.integers(1, 3, size=n_rows),
            "credit_risk": rng.integers(0, 2, size=n_rows),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_full_pipeline_train_evaluate_predict(tmp_path: Path):
    # 1) Datos sintéticos de entrenamiento/evaluación
    train_csv = tmp_path / "train_credit.csv"
    _build_synthetic_credit_csv(train_csv, n_rows=80)

    # 2) Cargar datos con el DataModule de tu pipeline
    dm = DataModule(train_csv, target="credit_risk")
    X, y = dm.load()

    cfg = TrainConfig(test_size=0.25, random_state=42)
    X_train, X_test, y_train, y_test = dm.split(X, y, cfg)

    # 3) Preprocesador usando PreprocessorFactory
    preprocessor = PreprocessorFactory.build(X_train)

    # 4) Modelo XGBoost dentro de un Pipeline idéntico al de train.py
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                XGBClassifier(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=1,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    # 5) Guardar modelo y métricas como haría el entrenamiento
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = Evaluator.compute_metrics(y_test, y_pred, y_proba)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    # Aserciones de entrenamiento
    assert model_path.exists()
    assert metrics_path.exists()
    assert "accuracy_test" in metrics

    # 6) EVALUACIÓN completa con evaluate_model (usa el modelo guardado)
    eval_output_dir = tmp_path / "eval_artifacts"
    metrics_eval = evaluate_model(
        model_path=model_path,
        test_csv=train_csv,
        target_col="credit_risk",
        output_dir=eval_output_dir,
    )

    # Aserciones sobre evaluación
    assert "accuracy_test" in metrics_eval
    assert 0.0 <= metrics_eval["accuracy_test"] <= 1.0
    assert (eval_output_dir / "confusion_matrix.png").exists()
    assert (eval_output_dir / "roc_curve.png").exists()

    # 7) PREDICCIÓN sobre nuevo lote (sin columna credit_risk)
    df_new = pd.read_csv(train_csv).drop(columns=["credit_risk"]).head(10)
    new_csv = tmp_path / "nuevo_lote.csv"
    df_new.to_csv(new_csv, index=False)

    preds_csv = tmp_path / "predicciones.csv"
    preds_df = run_inference(
        input_data=new_csv,
        model_path=model_path,
        output_path=preds_csv,
    )

    # Aserciones finales: predicciones completas
    assert preds_csv.exists()
    assert "prediction" in preds_df.columns
    assert len(preds_df) == 10
