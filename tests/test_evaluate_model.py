# tests/test_evaluate_model.py

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from german_credit_ml.modeling import evaluate as eval_module


class DummyShapInterpreter:
    """Reemplazo ligero para evitar cálculo real de SHAP en tests."""
    @staticmethod
    def explain(pipeline, X_test, output_dir: Path):
        # Simular creación de algún artefacto
        output_dir.mkdir(parents=True, exist_ok=True)
        dummy_file = output_dir / "shap_dummy.txt"
        dummy_file.write_text("ok")
        return [dummy_file], None


def test_evaluate_model_creates_metrics_and_artifacts(tmp_path: Path, monkeypatch):
    # 1) Parchear ShapInterpreter por una versión dummy
    monkeypatch.setattr(eval_module, "ShapInterpreter", DummyShapInterpreter)

    # 2) Crear datos de prueba
    n = 50
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "credit_risk": np.random.randint(0, 2, size=n),
        }
    )
    X = df[["feature1", "feature2"]]
    y = df["credit_risk"]

    # 3) Construir un pipeline simple compatible con evaluate_model
    pipe = Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(X, y)

    # 4) Guardar modelo y CSV de test
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)

    test_csv = tmp_path / "test.csv"
    df.to_csv(test_csv, index=False)

    output_dir = tmp_path / "eval_artifacts"

    # 5) Ejecutar evaluate_model
    metrics = eval_module.evaluate_model(
        model_path=model_path,
        test_csv=test_csv,
        target_col="credit_risk",
        output_dir=output_dir,
    )

    # 6) Validaciones
    assert "accuracy_test" in metrics
    assert 0.0 <= metrics["accuracy_test"] <= 1.0

    # Confusion matrix y curva ROC deben existir
    assert (output_dir / "confusion_matrix.png").exists()
    assert (output_dir / "roc_curve.png").exists()

    # Y nuestro artefacto dummy de SHAP también
    assert (output_dir / "shap_dummy.txt").exists()
