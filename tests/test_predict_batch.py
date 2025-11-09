# tests/test_predict_batch.py

import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

# asegurar raíz proyecto
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from german_credit_ml.modeling.predict import run_inference


def test_run_inference_creates_predictions(tmp_path: Path):
    # 1) datos dummy con mismas columnas que el modelo
    columns = [
        "status","duration","credit_history","purpose","amount","savings",
        "employment_duration","installment_rate","personal_status_sex",
        "other_debtors","present_residence","property","age",
        "other_installment_plans","housing","number_credits","job",
        "people_liable","telephone","foreign_worker",
    ]

    df = pd.DataFrame(
        [
            [2,24,3,2,3500,4,3,3,2,1,2,3,45,3,1,1,2,1,1,1],
            [1,12,2,1,1500,2,2,2,3,1,1,2,30,3,2,2,3,1,2,1],
        ],
        columns=columns,
    )
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)

    # 2) modelo dummy compatible
    X = df[columns]
    y = [0, 1]
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]
    )
    pipe.fit(X, y)
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)

    # 3) ejecutar inferencia
    output_csv = tmp_path / "preds.csv"
    out_df = run_inference(input_csv, model_path, output_csv)

    # 4) validaciones
    assert output_csv.exists()
    assert "prediction" in out_df.columns
    assert len(out_df) == len(df)
