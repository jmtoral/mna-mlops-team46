# tests/test_preprocess.py

from pathlib import Path

import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

# Asegurar que la raíz del proyecto está en el path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from german_credit_ml.modeling import preprocess


def test_describe_preprocessing_counts_columns_correctly():
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0],
            "num2": [10.0, 20.0, 30.0],
            "cat1": ["a", "b", "a"],
        }
    )

    summary = preprocess.describe_preprocessing(df)

    assert summary["n_num_cols"] == 2
    assert summary["n_cat_cols"] == 1
    assert "num1" in summary["num_cols"]
    assert "cat1" in summary["cat_cols"]


def test_build_preprocessor_returns_column_transformer():
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0],
            "num2": [10.0, 20.0, 30.0],
            "cat1": ["a", "b", "a"],
        }
    )

    pre = preprocess.build_preprocessor(df)

    assert isinstance(pre, ColumnTransformer)

    X_trans = pre.fit_transform(df)
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == df.shape[0]


def test_load_data_reads_csv(tmp_path: Path):
    csv_path = tmp_path / "dummy.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(csv_path, index=False)

    loaded = preprocess.load_data(csv_path)

    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape == (2, 2)


def test_preprocess_data_returns_numpy_array():
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan],
            "cat1": ["x", None, "z"],
        }
    )

    X_trans = preprocess.preprocess_data(df)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == df.shape[0]
