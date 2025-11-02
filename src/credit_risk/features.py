import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = list(X.columns)
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler(with_mean=False))])
    pre = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return pre
