# german_credit_ml/modeling/train.py

import argparse
import json
import pickle
import warnings
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Importaciones de ML y visualización
import mlflow
import mlflow.sklearn
import mlflow.xgboost
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

# --- Importaciones de Rich y utilidades ---
from german_credit_ml.utils import console, print_header # Importar consola y header
from rich.table import Table # Importar tabla de Rich

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass(frozen=True)
class Paths:
    """Almacena las rutas necesarias para el script."""
    input_data: Path
    model_output: Path
    metrics_output: Path
    plots_output: Path


@dataclass(frozen=True)
class TrainConfig:
    """Configuración del entrenamiento."""
    test_size: float = 0.2
    random_state: int = 42
    xgb_params: Dict = None
    experiment_name: str = "German Credit XGBoost"

    def get_xgb_params(self) -> Dict: # Renombrado de xgb a get_xgb_params
        """Devuelve los parámetros base de XGBoost actualizados con los específicos."""
        base = dict(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=self.random_state
        )
        # Asegurarse de que use_label_encoder no esté presente si no es necesario
        # base.pop('use_label_encoder', None)
        if self.xgb_params:
            base.update(self.xgb_params)
        return base


class DataModule:
    """Clase para cargar y dividir los datos."""
    def __init__(self, csv_path: Path, target: str = "credit_risk"):
        self.csv_path = csv_path
        self.target = target

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        console.print(f"[bold green][INFO][/bold green] Cargando datos desde: [cyan]{self.csv_path}[/cyan]")
        df = pd.read_csv(self.csv_path)
        X, y = df.drop(columns=self.target), df[self.target]
        console.print(f"[bold bright_green][SUCCESS][/bold bright_green] Datos cargados: {X.shape[0]} filas, {X.shape[1]} features.")
        return X, y

    def split(self, X: pd.DataFrame, y: pd.Series, cfg: TrainConfig):
        console.print("[bold green][INFO][/bold green] Dividiendo datos en entrenamiento y prueba...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
        )
        console.print(f"[bold bright_green][SUCCESS][/bold bright_green] División completa. Entrenamiento: {X_train.shape[0]} filas, Prueba: {X_test.shape[0]} filas.")
        return X_train, X_test, y_train, y_test


class PreprocessorFactory:
    """Clase para construir el pipeline de preprocesamiento."""
    @staticmethod
    def build(X_train: pd.DataFrame) -> ColumnTransformer:
        console.print("\n[bold green][INFO][/bold green] Definiendo pipeline de preprocesamiento...")
        num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist() # Corrección aquí
        console.print(f"  -> {len(num_cols)} cols numéricas, {len(cat_cols)} cols categóricas.")

        num_transformer = SimpleImputer(strategy="median")
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[("num", num_transformer, num_cols), ("cat", cat_transformer, cat_cols)],
            remainder="drop"
        )
        console.print("[bold bright_green][SUCCESS][/bold bright_green] Preprocesador definido.")
        return preprocessor


class Evaluator:
    """Clase para calcular métricas y generar gráficas de evaluación."""
    @staticmethod
    def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
        console.print("\n[bold green][INFO][/bold green] Calculando métricas de evaluación...")
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0) # Añadir zero_division
        metrics = {
            "f1_score_test": report.get('1', {}).get('f1-score', 0.0),
            "accuracy_test": report.get('accuracy', 0.0),
            "precision_test": report.get('1', {}).get('precision', 0.0),
            "recall_test": report.get('1', {}).get('recall', 0.0),
            "auc_test": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5, # Handle cases with only one class
            "bad_rate_test": float(np.mean(y_pred == 0)),
        }
        console.print("[bold bright_green][SUCCESS][/bold bright_green] Métricas calculadas.")
        return metrics

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, outpath: Path):
        console.print(f"  -> Generando Matriz de Confusión en [cyan]{outpath.name}[/cyan]...")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Malo (0)", "Bueno (1)"], yticklabels=["Malo (0)", "Bueno (1)"])
        plt.title("Matriz de Confusión"); plt.ylabel("Verdadero"); plt.xlabel("Predicho");
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath); plt.close()

    @staticmethod
    def plot_roc(y_true, y_proba, outpath: Path, auc_val: float):
        console.print(f"  -> Generando Curva ROC en [cyan]{outpath.name}[/cyan]...")
        if len(np.unique(y_true)) > 1: # Only plot ROC if both classes are present
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"Curva ROC (AUC = {auc_val:.2f})")
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle="--")
            plt.xlabel("Tasa de Falsos Positivos"); plt.ylabel("Tasa de Verdaderos Positivos"); plt.title("Curva ROC"); plt.legend(loc="lower right")
            outpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath); plt.close()
        else:
            console.print("[yellow]Advertencia:[/yellow] No se puede generar Curva ROC, solo hay una clase en y_true.")


class ShapInterpreter:
    """Clase para realizar el análisis de interpretabilidad con SHAP."""
    @staticmethod
    def explain(pipeline: Pipeline, X_test: pd.DataFrame, plots_dir: Path) -> Tuple[List[Path], pd.DataFrame]:
        console.print("\n[bold green][INFO][/bold green] PASO 5: Realizando análisis SHAP...")
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["clf"]

        X_test_transformed = preprocessor.transform(X_test)
        try:
            feature_names_out = preprocessor.get_feature_names_out()
        except Exception:
            console.print("[yellow]Advertencia:[/yellow] No se pudieron obtener nombres de features. Usando genéricos.")
            feature_names_out = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]
        X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names_out)

        console.print("  -> Calculando valores SHAP...")
        try:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test_transformed, check_additivity=False)
        except Exception as e:
            console.print(f"[bold red]Error al calcular SHAP con TreeExplainer:[/bold red] {e}. Intentando con explainer genérico.")
            explainer = shap.Explainer(classifier, X_test_transformed_df)
            shap_values = explainer(X_test_transformed_df).values

        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_pos_class = shap_values[1]
        else:
            shap_values_pos_class = shap_values

        console.print("  -> Top 10 features más importantes (SHAP):")
        shap_df = pd.DataFrame(shap_values_pos_class, columns=feature_names_out)
        vals = np.abs(shap_df.values).mean(0)
        shap_importance = pd.DataFrame(
            list(zip(feature_names_out, vals)),
            columns=['feature', 'importance']
        ).sort_values(by='importance', ascending=False)
        console.print(shap_importance.head(10).to_string(index=False))

        plots_dir.mkdir(parents=True, exist_ok=True)
        bar_path = plots_dir / "shap_importance_plot.png"
        shap.summary_plot(shap_values_pos_class, X_test_transformed_df, plot_type="bar", show=False)
        plt.title("Importancia de Features (SHAP | media abs)"); plt.tight_layout();
        plt.savefig(bar_path); plt.close();

        swarm_path = plots_dir / "shap_summary_plot.png"
        shap.summary_plot(shap_values_pos_class, X_test_transformed_df, show=False)
        plt.tight_layout(); plt.savefig(swarm_path); plt.close();
        console.print(f"[bold bright_green][SUCCESS][/bold bright_green] Gráficas SHAP guardadas en: {plots_dir}")

        # Añadir csv path a la lista de paths a retornar
        shap_csv_path = plots_dir / "shap_top_features.csv"
        shap_importance.head(10).to_csv(shap_csv_path, index=False)


        return [bar_path, swarm_path, shap_csv_path], shap_importance


class MlflowLogger:
    """Clase para gestionar el registro en MLflow."""
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def start_run(self) -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"run_{now}"
        self.run_info = mlflow.start_run(run_name=run_name).info # Guardar info del run
        console.print("\n" + "="*50, style="bold dim")
        console.print(f" [bold yellow]Iniciando Run MLflow: {run_name} (ID: {self.run_info.run_id})[/bold yellow] ".center(60, "="), style="bold dim")
        console.print("="*50, style="bold dim")
        return self.run_info.run_id # Devolver el ID

    def log_params_metrics(self, model: Pipeline, metrics: Dict[str, float]):
        console.print("\n[bold green][INFO][/bold green] Registrando parámetros y métricas en MLflow...")
        classifier_params = model.named_steps["clf"].get_params()
        mlflow.log_params(classifier_params)
        mlflow.log_metrics(metrics)
        console.print("[bold bright_green][SUCCESS][/bold bright_green] Parámetros y métricas registrados.")

    def log_models(self, model: Pipeline):
        console.print("\n[bold green][INFO][/bold green] Registrando modelos en MLflow...")
        mlflow.sklearn.log_model(model, artifact_path="sklearn-pipeline")
        classifier = model.named_steps["clf"]
        mlflow.xgboost.log_model(classifier, artifact_path="xgboost-model")
        console.print("[bold bright_green][SUCCESS][/bold bright_green] Modelos registrados.")

    def log_artifacts(self, artifact_paths: List[Path], subdir: str = "plots"):
        console.print(f"\n[bold green][INFO][/bold green] Registrando artefactos en MLflow (subdirectorio: {subdir})...")
        for p in artifact_paths:
            if p is None or not p.exists():
                console.print(f"[yellow]Advertencia:[/yellow] Artefacto no encontrado o nulo, omitiendo: {p}")
                continue
            try:
                mlflow.log_artifact(str(p), subdir)
            except Exception as e:
                console.print(f"[bold red]Error[/bold red] registrando artefacto {p}: {e}")
        console.print("[bold bright_green][SUCCESS][/bold bright_green] Artefactos registrados.")

    def end_run(self):
        mlflow.end_run()
        console.print("\n[bold green][INFO][/bold green] Run de MLflow finalizado.")


class Trainer:
    """Clase principal que orquesta el proceso de entrenamiento."""
    def __init__(self, paths: Paths, cfg: TrainConfig):
        self.paths = paths
        self.cfg = cfg
        self.mlflog = MlflowLogger(cfg.experiment_name)

    def build_model(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Construye el pipeline final (preprocesador + clasificador)."""
        console.print("\n[bold green][INFO][/bold green] Construyendo pipeline final...")
        xgb_clf = xgb.XGBClassifier(**self.cfg.get_xgb_params())
        model = Pipeline(steps=[("preprocessor", preprocessor), ("clf", xgb_clf)])
        console.print("[bold bright_green][SUCCESS][/bold bright_green] Pipeline construido.")
        return model

    def run(self):
        """Ejecuta todo el flujo de entrenamiento."""
        print_header() # Imprime encabezado ASCII
        run_id = self.mlflog.start_run() # Captura el Run ID

        # --- Carga y División ---
        data_module = DataModule(self.paths.input_data)
        X, y = data_module.load()
        X_train, X_test, y_train, y_test = data_module.split(X, y, self.cfg)

        # --- Preprocesamiento ---
        preprocessor = PreprocessorFactory.build(X_train)

        # --- Entrenamiento ---
        model = self.build_model(preprocessor)
        console.print("\n[bold green][INFO][/bold green] Entrenando el pipeline...")
        model.fit(X_train, y_train)
        console.print("[bold bright_green][SUCCESS][/bold bright_green] Pipeline entrenado.")

        # --- Evaluación ---
        console.print("\n[bold green][INFO][/bold green] PASO 3: Evaluando el modelo...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = Evaluator.compute_metrics(y_test, y_pred, y_proba)

        # Mostrar tabla de métricas con Rich
        metrics_table = Table(title="📊 Métricas de Evaluación (Conjunto de Prueba)")
        metrics_table.add_column("Métrica", style="cyan", no_wrap=True)
        metrics_table.add_column("Valor", style="magenta")
        for k, v in metrics.items():
            metric_name = k.replace('_test', '').replace('_', ' ').title()
            metrics_table.add_row(metric_name, f"{v:.4f}")
        console.print(metrics_table)

        # --- Generar Gráficas ---
        console.print("\n[bold green][INFO][/bold green] PASO 4: Generando gráficas de evaluación...")
        plots_dir = self.paths.plots_output
        cm_path = plots_dir / "confusion_matrix.png"
        roc_path = plots_dir / "roc_curve.png"
        Evaluator.plot_confusion_matrix(y_test, y_pred, cm_path)
        Evaluator.plot_roc(y_test, y_proba, roc_path, metrics.get("auc_test", 0.0))
        console.print(f"[bold bright_green][SUCCESS][/bold bright_green] Gráficas de evaluación guardadas.")

        # --- SHAP ---
        shap_paths, shap_importance = ShapInterpreter.explain(model, X_test, plots_dir)
        # La lista shap_paths ahora incluye el csv: [bar_path, swarm_path, shap_csv_path]

        # --- Guardado y Registro ---
        console.print("\n[bold green][INFO][/bold green] PASO 6: Guardando artefactos locales y registrando en MLflow...")
        # Guardar Pipeline .pkl para DVC
        with open(self.paths.model_output, "wb") as f:
            pickle.dump(model, f)
        console.print(f" -> Pipeline completo guardado para DVC en: [cyan]{self.paths.model_output}[/cyan]")

        # Guardar Metadatos del Modelo (incluyendo Run ID) para DVC
        model_metadata = {
            "mlflow_run_id": run_id,
            "training_timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "model_description": "XGBoost Pipeline for German Credit Risk"
        }
        metadata_path = self.paths.model_output.parent / (self.paths.model_output.stem + "_metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=4)
            console.print(f" -> Metadatos del modelo guardados en: [cyan]{metadata_path}[/cyan]")
        except Exception as e:
            console.print(f"[bold red]ERROR:[/bold red] No se pudo guardar el archivo de metadatos en {metadata_path}: {e}")


        # Guardar Métricas .json para DVC
        try:
            clf_params = model.named_steps["clf"].get_params()
        except AttributeError:
            clf_params = {}
        metrics_to_save = {**metrics, "params": clf_params}
        with open(self.paths.metrics_output, "w") as f:
            json.dump(metrics_to_save, f, indent=4, default=str)
        console.print(f" -> Métricas guardadas para DVC en: [cyan]{self.paths.metrics_output}[/cyan]")

        # Registro en MLflow
        self.mlflog.log_params_metrics(model, metrics)
        self.mlflog.log_models(model)
        # Asegurarse de pasar la lista correcta de paths a log_artifacts
        artifacts_to_log = [cm_path, roc_path] + shap_paths # shap_paths ya incluye el csv
        self.mlflog.log_artifacts(artifacts_to_log, subdir="plots")
        self.mlflog.end_run()

        console.print("\n[bold bright_green][SUCCESS][/bold bright_green] Entrenamiento finalizado exitosamente.")


# --- Bloque de ejecución principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena Pipeline Sklearn con XGBoost (OOP).")
    parser.add_argument("--input-data", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--plots-output", required=True, type=Path)

    args = parser.parse_args()

    # Crear instancias de configuración
    paths = Paths(
        input_data=args.input_data,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
        plots_output=args.plots_output
    )
    # Usar valores por defecto; DVC los sobrescribirá si están en params.yaml
    cfg = TrainConfig(test_size=0.2, random_state=42)

    # Crear e iniciar el entrenador
    trainer = Trainer(paths, cfg)
    trainer.run()

def train_model(input_data, model_output, metrics_output, plots_output):
    """Entrena el pipeline completo y registra resultados."""
    paths = Paths(
        input_data=input_data,
        model_output=model_output,
        metrics_output=metrics_output,
        plots_output=plots_output,
    )
    cfg = TrainConfig(test_size=0.2, random_state=42)
    trainer = Trainer(paths, cfg)
    trainer.run()