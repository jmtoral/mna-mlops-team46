import pandas as pd
from german_credit_ml.modeling.preprocess import build_preprocessor

def test_build_preprocessor_runs():
    df = pd.DataFrame({
        "edad": [21, 35, None],
        "genero": ["M", "F", None],
        "monto": [1000, 5000, 2000],
    })
    prep = build_preprocessor(df)
    Xt = prep.fit_transform(df)
    assert Xt.shape[0] == 3
