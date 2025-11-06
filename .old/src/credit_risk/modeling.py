from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def make_estimator(model_cfg: dict):
    if model_cfg["type"] == "RandomForestClassifier":
        return RandomForestClassifier(
            class_weight=model_cfg.get("class_weight"),
            n_jobs=model_cfg.get("n_jobs", -1),
            random_state=42,
        )
    raise ValueError(f"Modelo no soportado: {model_cfg['type']}")


def train_and_search(
    preprocessor, model_cfg: dict, param_grid: dict, X_train, y_train, cv: int
) -> GridSearchCV:
    base_est = make_estimator(model_cfg)
    pipe = Pipeline([("pre", preprocessor), ("clf", base_est)])
    gs = GridSearchCV(
        pipe,
        param_grid={f"clf__{k}": v for k, v in param_grid.items()},
        cv=cv,
        n_jobs=-1,
        scoring="f1",
        refit=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs


def evaluate(gs: GridSearchCV, X_test, y_test):
    from .metrics import classification_report_dict

    y_pred = gs.predict(X_test)
    y_proba = None
    if hasattr(gs, "predict_proba"):
        y_proba = gs.predict_proba(X_test)[:, 1]
    return classification_report_dict(y_test, y_pred, y_proba)
