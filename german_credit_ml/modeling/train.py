from __future__ import annotations
import argparse, json, pickle, warnings, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow, mlflow.sklearn, mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass(frozen=True)
class Paths:
    input_data: Path
    model_output: Path
    metrics_output: Path
    plots_output: Path


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    xgb_params: Dict = None
    experiment_name: str = "German Credit XGBoost"

    def xgb(self) -> Dict:
        base = dict(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=self.random_state
        )
        if self.xgb_params:
            base.update(self.xgb_params)
        return base


class DataModule:
    def __init__(self, csv_path: Path, target: str = "credit_risk"):
        self.csv_path = csv_path
        self.target = target
        self._train = self._test = None

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(self.csv_path)
        X, y = df.drop(columns=self.target), df[self.target]
        return X, y

    def split(self, X: pd.DataFrame, y: pd.Series, cfg: TrainConfig):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test


class PreprocessorFactory:
    @staticmethod
    def build(X_train: pd.DataFrame) -> ColumnTransformer:
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X_train.columns if c not in num_cols]

        num = SimpleImputer(strategy="median")
        cat = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        return ColumnTransformer(
            transformers=[("num", num, num_cols), ("cat", cat, cat_cols)],
            remainder="drop"
        )


class Evaluator:
    @staticmethod
    def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
        report = classification_report(y_true, y_pred, output_dict=True)
        return {
            "f1_score_test": report["1"]["f1-score"],
            "accuracy_test": report["accuracy"],
            "auc_test": roc_auc_score(y_true, y_proba),
            "bad_rate_test": float(np.mean(y_pred == 0)),
        }

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, outpath: Path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Malo", "Bueno"], yticklabels=["Malo", "Bueno"])
        plt.title("Matriz de Confusión"); plt.ylabel("Verdadero"); plt.xlabel("Predicho")
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath); plt.close()

    @staticmethod
    def plot_roc(y_true, y_proba, outpath: Path, auc_val: float):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc_val:.2f})")
        plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curva ROC"); plt.legend(loc="lower right")
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath); plt.close()


class ShapInterpreter:
    @staticmethod
    def explain(pipeline: Pipeline, X_test: pd.DataFrame, plots_dir: Path):
        pre = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["clf"]

        X_proc = pre.transform(X_test)
        try:
            feature_names = pre.get_feature_names_out()
        except Exception:
            feature_names = [f"f{i}" for i in range(X_proc.shape[1])]
        X_proc_df = pd.DataFrame(X_proc, columns=feature_names)

        try:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_proc)
        except Exception:
            explainer = shap.Explainer(clf)
            shap_values = explainer(X_proc).values

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Importancia media absoluta
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        vals = np.abs(shap_df.values).mean(0)
        shap_importance = (
            pd.DataFrame({"feature": feature_names, "importance": vals})
            .sort_values("importance", ascending=False)
        )

        # Plots
        bar_path = plots_dir / "shap_importance_plot.png"
        shap.summary_plot(shap_values, X_proc_df, plot_type="bar", show=False)
        plt.title("Importancia de Features (SHAP | media abs)"); plt.tight_layout()
        bar_path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(bar_path); plt.close()

        swarm_path = plots_dir / "shap_summary_plot.png"
        shap.summary_plot(shap_values, X_proc_df, show=False)
        plt.tight_layout(); plt.savefig(swarm_path); plt.close()

        return [bar_path, swarm_path], shap_importance

class MlflowLogger:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def start_run(self) -> None:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run = mlflow.start_run(run_name=f"run_{now}")
        print("TRACKING_URI:", mlflow.get_tracking_uri())
        print("ARTIFACT_URI:", mlflow.get_artifact_uri())

    def log_params_metrics(self, model: Pipeline, metrics: Dict[str, float]):
        clf = model.named_steps["clf"]
        mlflow.log_params(clf.get_params())
        mlflow.log_metrics(metrics)

    def log_models(self, model: Pipeline):
        clf = model.named_steps["clf"]
        mlflow.sklearn.log_model(model, artifact_path="sklearn-pipeline")
        mlflow.xgboost.log_model(clf, artifact_path="xgboost-model")

    def log_artifacts(self, artifact_paths: List[Path], subdir: str = "plots"):
        for p in artifact_paths:
            mlflow.log_artifact(str(p), subdir)

    def end_run(self):
        mlflow.end_run()


class Trainer:
    def __init__(self, paths: Paths, cfg: TrainConfig):
        self.paths = paths
        self.cfg = cfg
        self.mlflog = MlflowLogger(cfg.experiment_name)

    def build_model(self, pre: ColumnTransformer) -> Pipeline:
        xgb_clf = xgb.XGBClassifier(**self.cfg.xgb())
        return Pipeline(steps=[("preprocessor", pre), ("clf", xgb_clf)])

    def run(self):
        self.mlflog.start_run()
        print("\n[INFO] Cargando datos…")
        data = DataModule(self.paths.input_data)
        X, y = data.load()
        X_tr, X_te, y_tr, y_te = data.split(X, y, self.cfg)

        print("[INFO] Construyendo preprocesamiento…")
        pre = PreprocessorFactory.build(X_tr)

        print("[INFO] Entrenando modelo…")
        model = self.build_model(pre)
        model.fit(X_tr, y_tr)

        print("[INFO] Evaluando…")
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        metrics = Evaluator.compute_metrics(y_te, y_pred, y_proba)
        for k, v in metrics.items():
            print(f"  -> {k}: {v:.4f}")

        print("[INFO] Graficando…")
        plots_dir = self.paths.plots_output
        cm_path = plots_dir / "confusion_matrix.png"
        roc_path = plots_dir / "roc_curve.png"
        Evaluator.plot_confusion_matrix(y_te, y_pred, cm_path)
        Evaluator.plot_roc(y_te, y_proba, roc_path, metrics["auc_test"])

        print("[INFO] SHAP…")
        shap_paths, shap_importance = ShapInterpreter.explain(model, X_te, plots_dir)

        TOP_N = 10
        print(f"\n  -> Top {TOP_N} features más importantes (SHAP | media abs):")
        print(shap_importance.head(TOP_N).to_string(index=False))

        # (opcional) guardarlo/loguearlo
        (shap_importance.head(TOP_N)
        .to_csv(plots_dir / "shap_top_features.csv", index=False))
        self.mlflog.log_artifacts([plots_dir / "shap_top_features.csv"], subdir="plots")
        try:
            mlflow.log_text(shap_importance.head(TOP_N).to_string(index=False),
                            "plots/shap_top_features.txt")
        except Exception:
            pass


        print("[INFO] Guardando artefactos locales…")
        with open(self.paths.model_output, "wb") as f:
            pickle.dump(model, f)
        with open(self.paths.metrics_output, "w") as f:
            json.dump({**metrics, "params": model.named_steps["clf"].get_params()},
                      f, indent=4, default=str)

        print("[INFO] Registrando en MLflow…")
        self.mlflog.log_params_metrics(model, metrics)
        self.mlflog.log_models(model)
        self.mlflog.log_artifacts([cm_path, roc_path] + shap_paths, subdir="plots")
        self.mlflog.end_run()
        print("[SUCCESS] Entrenamiento finalizado.")


def parse_args():
    p = argparse.ArgumentParser(description="Entrena modelo XGBoost (OOP).")
    p.add_argument("--input-data", required=True, type=Path)
    p.add_argument("--model-output", required=True, type=Path)
    p.add_argument("--metrics-output", required=True, type=Path)
    p.add_argument("--plots-output", required=True, type=Path)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = Paths(
        input_data=args.input_data,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
        plots_output=args.plots_output
    )
    cfg = TrainConfig(test_size=0.2, random_state=42)
    Trainer(paths, cfg).run()
