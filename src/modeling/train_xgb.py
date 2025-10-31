# src/modeling/train_xgb.py
import argparse
from datetime import datetime
import getpass
import json
import os
from pathlib import Path
import socket
import subprocess

import cloudpickle
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-data", required=True)
    p.add_argument("--model-output", default="models/xgboost_model.pkl")
    # En dvc.yaml pasas --metrics-output y --plots-output (un directorio)
    p.add_argument("--metrics-output", default="reports/metrics.json")
    p.add_argument("--plots-output", default="reports/figures/training")
    p.add_argument("--mlflow-experiment", default="fase1_modelado_equipo46")
    return p.parse_args()


def main(args):
    # === MLflow: tracking local por defecto (evita mezclar con otros) ===
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", f"file:{Path.cwd()/'mlruns'}")
    )
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", args.mlflow_experiment))

    run_name = (
        f"train_xgb_{getpass.getuser()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if mlflow.active_run():  # por si quedó algo abierto
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name):
        # Etiquetas útiles para distinguir corridas en equipo
        mlflow.set_tag("author", getpass.getuser())
        mlflow.set_tag("host", socket.gethostname())
        try:
            mlflow.set_tag(
                "git_branch",
                subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
                .decode()
                .strip(),
            )
            mlflow.set_tag(
                "git_commit",
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode()
                .strip(),
            )
        except Exception:
            pass
        mlflow.set_tag("dvc_stage", "train")
        mlflow.set_tag("dataset", args.input_data)

        # === PASO 1: Carga y Preparación ===
        print("==== PASO 1: Carga y Preparación de Datos ====")
        data_path = Path(args.input_data).resolve()
        assert data_path.exists(), f"No existe el archivo de entrada: {data_path}"
        print(f"\nCargando: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Datos cargados con {df.shape[0]} filas y {df.shape[1]} columnas.")

        TARGET = "credit_risk"
        X = df.drop(columns=[TARGET])
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print("Datos divididos en train/test.")

        # ==== PASO 2: Preprocesamiento + Entrenamiento (Pipeline) ====
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X_train.columns if c not in num_cols]

        num_transformer = SimpleImputer(strategy="median")  # XGB no necesita escalar
        cat_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        # ColumnTransformer que aplica cada paso a sus columnas
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num_cols),
                ("cat", cat_transformer, cat_cols),
            ],
            remainder="drop",  # o "passthrough" si quieres dejar columnas no listadas
        )

        # === PASO 2: Entrenamiento (XGBoost) ===
        print("\n==== PASO 2: Entrenamiento del Modelo ====")
        params = {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
            "tree_method": "hist",
            "use_label_encoder": False,
            "n_jobs": -1,
        }
        print("Parámetros XGBoost:", params)

        # model = XGBClassifier(**params)
        # model.fit(X_train, y_train)

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", XGBClassifier(**params)),
            ]
        )

        # Entrenar el pipeline con los datos crudos (sin transformar)
        pipeline.fit(X_train, y_train)

        # === PASO 3: Evaluación ===
        print("\n==== PASO 3: Evaluación, Guardado y Registro ====")
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        print(
            f"F1: {f1:.4f} | ROC-AUC: {auc:.4f} | Acc: {acc:.4f} | Prec: {pre:.4f} | Rec: {rec:.4f}"
        )

        # === Guardar métricas JSON (para DVC) ===
        metrics = {
            "f1": float(f1),
            "roc_auc": float(auc),
            "accuracy": float(acc),
            "precision": float(pre),
            "recall": float(rec),
            "n_samples_train": int(len(y_train)),
            "n_samples_test": int(len(y_test)),
        }
        mout = Path(args.metrics_output)
        mout.parent.mkdir(parents=True, exist_ok=True)
        with open(mout, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"→ Métricas guardadas en: {mout}")

        # === Figuras (usar directorio plots_output) ===
        plots_dir = Path(args.plots_output)
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig_cm = plt.figure()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title("Matriz de confusión (XGB)")
        cm_path = plots_dir / "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close(fig_cm)
        print(f"→ Matriz de confusión guardada en: {cm_path}")

        # (Opcional) curvas ROC y PR como evidencia
        try:
            from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

            fig_roc = plt.figure()
            RocCurveDisplay.from_predictions(y_test, y_proba)
            plt.title("ROC (XGB)")
            roc_path = plots_dir / "roc_curve.png"
            plt.savefig(roc_path, bbox_inches="tight")
            plt.close(fig_roc)

            fig_pr = plt.figure()
            PrecisionRecallDisplay.from_predictions(y_test, y_proba)
            plt.title("Precision-Recall (XGB)")
            pr_path = plots_dir / "precision_recall_curve.png"
            plt.savefig(pr_path, bbox_inches="tight")
            plt.close(fig_pr)
        except Exception:
            pass

        # === Guardar modelo (pickle) ===

        model_out = Path(args.model_output)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        # joblib.dump(model, model_out)
        print(f"→ Modelo guardado en: {model_out}")

        # === Log a MLflow: métricas, artefactos y modelo ===
        # Métricas
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Artefactos (métricas y figuras)
        if mout.exists():
            mlflow.log_artifact(str(mout))
        if cm_path.exists():
            mlflow.log_artifact(str(cm_path), artifact_path="figures")
        if "roc_path" in locals() and Path(roc_path).exists():
            mlflow.log_artifact(str(roc_path), artifact_path="figures")
        if "pr_path" in locals() and Path(pr_path).exists():
            mlflow.log_artifact(str(pr_path), artifact_path="figures")

        # Firma + ejemplo de entrada
        X_example = X_test.iloc[:5].copy()
        # signature = infer_signature(X_example, model.predict_proba(X_example)[:, 1])

        # Evitar pip freeze lento
        pip_reqs = [
            f"scikit-learn=={sklearn.__version__}",
            f"cloudpickle=={cloudpickle.__version__}",
            "xgboost",
            "numpy",
            "pandas",
            "matplotlib",
            "mlflow",
            "joblib",
        ]

        try:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",  # MLflow nuevo
                input_example=X_example,
                pip_requirements=pip_reqs,
            )
        except TypeError:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",  # compatibilidad
                input_example=X_example,
                pip_requirements=pip_reqs,
            )

        print("\nRun MLflow completado.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
