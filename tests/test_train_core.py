# tests/test_train_core.py

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

# Asegurar que la raíz del proyecto está en el path (igual que en test_preprocess.py)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from german_credit_ml.modeling.train import (
    DataModule,
    PreprocessorFactory,
    TrainConfig,
    Evaluator,
)


def _build_dummy_credit_csv(path: Path, n_rows: int = 50):
    """Genera un CSV sintético con las columnas esperadas para entrenamiento."""
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
            # target
            "credit_risk": rng.integers(0, 2, size=n_rows),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_datamodule_load_and_split(tmp_path: Path):
    """Verifica que DataModule carga y separa correctamente X e y."""
    csv_path = tmp_path / "dummy_credit.csv"
    _build_dummy_credit_csv(csv_path, n_rows=100)

    dm = DataModule(csv_path, target="credit_risk")
    X, y = dm.load()

    # X e y con tipos esperados
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    # La columna objetivo no debe estar en X
    assert "credit_risk" not in X.columns
    # Longitudes consistentes
    assert len(X) == len(y)

    # División train/test con estratificación
    cfg = TrainConfig(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = dm.split(X, y, cfg)

    # Tamaños coherentes
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]

    # Ambas particiones deben contener ambas clases (estratify=y)
    assert set(y_train.unique()) == set(y.unique())
    assert set(y_test.unique()) == set(y.unique())


def test_preprocessor_factory_builds_valid_transformer(tmp_path: Path):
    """Verifica que PreprocessorFactory.build devuelve un ColumnTransformer funcional."""
    csv_path = tmp_path / "dummy_credit.csv"
    _build_dummy_credit_csv(csv_path, n_rows=30)

    dm = DataModule(csv_path, target="credit_risk")
    X, y = dm.load()

    cfg = TrainConfig()
    X_train, X_test, y_train, y_test = dm.split(X, y, cfg)

    preprocessor = PreprocessorFactory.build(X_train)

    assert isinstance(preprocessor, ColumnTransformer)

    X_train_trans = preprocessor.fit_transform(X_train)
    # Debe devolver un array numpy con el mismo número de filas
    assert isinstance(X_train_trans, np.ndarray)
    assert X_train_trans.shape[0] == X_train.shape[0]


def test_evaluator_compute_metrics_perfect_predictions():
    """Métricas deben ser perfectas cuando las predicciones son correctas."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = Evaluator.compute_metrics(y_true, y_pred, y_proba)

    # Todas las claves importantes deben estar presentes
    for key in [
        "f1_score_test",
        "accuracy_test",
        "precision_test",
        "recall_test",
        "auc_test",
        "bad_rate_test",
    ]:
        assert key in metrics

    # En predicciones perfectas, accuracy y F1 deben ser 1.0
    assert metrics["accuracy_test"] == 1.0
    assert metrics["f1_score_test"] == 1.0


def test_evaluator_handles_single_class_y_true():
    """Cuando y_true tiene una sola clase, AUC debe fijarse en 0.5 (comportamiento definido en Evaluator)."""
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.95, 0.85])

    metrics = Evaluator.compute_metrics(y_true, y_pred, y_proba)

    # Por implementación, AUC se pone en 0.5 si sólo hay una clase en y_true
    assert metrics["auc_test"] == 0.5
