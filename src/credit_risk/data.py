from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
