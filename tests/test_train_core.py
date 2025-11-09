# tests/test_train_core.py

from pathlib import Path

import numpy as np
import pandas as pd

from german_credit_ml.modeling.train import (
    DataModule,
    PreprocessorFactory,
    TrainConfig,
    Evaluator,
)


def _build_dummy_credit_csv(path: Path, n_rows: int = 50):
    df = pd.DataFrame(
        {
            "status": np.random.randint(1, 4, size=n_rows),
            "amount": np.random.randint(100, 1000, size=n_rows),
            "housing": np.random.choice(["own", "rent"], size=n_rows),
            "duration": np.random.randint(6, 60, size=n_rows),
            "age": np.random.randint(18, 70, size=n_rows),
            "credit_risk": np.random.randint(0, 2, size=n_rows),
        }
    )
    df.to_csv(path, index=False)


def test_datamodule_load_and_split(tmp_path: Path):
    csv_path = tmp_path / "dummy_credit.csv"
    _build_dummy_credit_csv(csv_path, n_rows=100)

    dm = DataModule(csv_path, target="credit_risk")
    X, y = dm.load()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "credit_risk" not in X.columns
    assert len(X) == len(y)

    cfg = TrainConfig(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = dm.split(X, y, cfg)

    # 80/20 aproximadamente
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
    # Estratificación: ambas particiones deben contener ambas clases
    assert set(y_train.unique()) == set(y.unique())
    assert set(y_test.unique()) == set(y.unique())


def test_preprocessor_factory_builds_valid_transformer(tmp_path: Path):
    csv_path = tmp_path / "dummy_credit.csv"
    _build_dummy_credit_csv(csv_path, n_rows=30)

    dm = DataModule(csv_path, target="credit_risk")
    X, y = dm.load()
    cfg = TrainConfig()
    X_train, X_test, y_train, y_test = dm.split(X, y, cfg)

    pre = PreprocessorFactory.build(X_train)

    from sklearn.compose import ColumnTransformer

    assert isinstance(pre, ColumnTransformer)

    X_train_trans = pre.fit_transform(X_train)
    assert isinstance(X_train_trans, np.ndarray)
    assert X_train_trans.shape[0] == X_train.shape[0]


def test_evaluator_compute_metrics_perfect_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = Evaluator.compute_metrics(y_true, y_pred, y_proba)

    for key in [
        "f1_score_test",
        "accuracy_test",
        "precision_test",
        "recall_test",
        "auc_test",
        "bad_rate_test",
    ]:
        assert key in metrics

    assert metrics["accuracy_test"] == 1.0
    assert metrics["f1_score_test"] == 1.0


def test_evaluator_handles_single_class_y_true():
    # Todos 1 en y_true → AUC se fija en 0.5 según implementación
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.95, 0.85])

    metrics = Evaluator.compute_metrics(y_true, y_pred, y_proba)

    assert metrics["auc_test"] == 0.5
