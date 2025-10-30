from dataclasses import dataclass

import mlflow
import mlflow.sklearn

from .config import Cfg
from .data import load_data, split_data
from .features import build_preprocessor
from .modeling import evaluate, train_and_search
from .viz import log_confusion_matrix, log_pr_roc


@dataclass
class Trainer:
    cfg: Cfg

    def _setup_mlflow(self):
        if self.cfg.mlflow.tracking_uri:
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment)

    def run(self):
        self._setup_mlflow()

        X, y = load_data(self.cfg.data.path, self.cfg.data.target)
        X_train, X_test, y_train, y_test = split_data(
            X, y, self.cfg.data.test_size, self.cfg.data.random_state
        )
        pre = build_preprocessor(X_train)

        with mlflow.start_run(run_name=self.cfg.mlflow.run_name) as run:
            mlflow.set_tags(
                {
                    "stage": "modeling",
                    "framework": "sklearn",
                    "model_family": self.cfg.model.type,
                }
            )

            gs = train_and_search(
                pre,
                self.cfg.model.__dict__,
                self.cfg.tuning.param_grid,
                X_train,
                y_train,
                self.cfg.tuning.cv,
            )

            best_params = {
                k.replace("clf__", ""): v for k, v in gs.best_params_.items()
            }
            for k, v in best_params.items():
                mlflow.log_param(k, v)

            y_pred = gs.predict(X_test)
            y_proba = None
            if hasattr(gs, "predict_proba"):
                y_proba = gs.predict_proba(X_test)[:, 1]

            metrics = evaluate(gs, X_test, y_test)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            log_confusion_matrix(mlflow, y_test, y_pred)
            log_pr_roc(mlflow, y_test, y_proba)

            mlflow.sklearn.log_model(gs.best_estimator_, artifact_path="model")

            if self.cfg.mlflow.register_model:
                mv = mlflow.register_model(
                    f"runs:/{run.info.run_id}/model", self.cfg.mlflow.register_model
                )
                if self.cfg.mlflow.register_alias:
                    client = mlflow.tracking.MlflowClient()
                    client.set_registered_model_alias(
                        name=self.cfg.mlflow.register_model,
                        alias=self.cfg.mlflow.register_alias,
                        version=mv.version,
                    )
