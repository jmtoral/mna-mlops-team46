# german_credit_ml/modeling/train.py

import argparse
import json
from pathlib import Path
import warnings
import pickle
import datetime

# Importaciones de ML, visualización e interpretabilidad
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import shap

# Importaciones para Pipeline de Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import mlflow.sklearn # Para registrar pipelines de sklearn

# Importaciones de Rich y utilidades personalizadas
from german_credit_ml.utils import console, print_header
from rich.table import Table

# Ignorar advertencias futuras para una salida más limpia en la consola
warnings.filterwarnings("ignore", category=FutureWarning)

def train_model(input_data: Path, model_output: Path, metrics_output: Path, plots_output: Path, params: dict):
    """
    Función completa para entrenar (con Pipeline Sklearn), evaluar y registrar un modelo XGBoost.
    """
    # Imprime el encabezado ASCII y el panel de información del proyecto
    print_header()

    # Configurar y nombrar el experimento en MLflow
    mlflow.set_experiment("German Credit XGBoost")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{now}"

    # Iniciar un "Run" de MLflow para registrar todo en un solo experimento
    with mlflow.start_run(run_name=run_name):

        console.print("\n" + "="*50, style="bold dim")
        console.print(f" [bold yellow]Iniciando Run: {run_name}[/bold yellow] ".center(50, "="), style="bold dim")
        console.print("="*50, style="bold dim")

        # --- PASO 1: Carga y Preparación de Datos ---
        console.print("\n[bold green][INFO][/bold green] PASO 1: Cargando y preparando datos...")
        df = pd.read_csv(input_data)
        console.print(f"Datos cargados con {df.shape[0]} filas y {df.shape[1]} columnas.")

        # Separar features (X) y target (y) ANTES del preprocesamiento
        X = df.drop(columns='credit_risk')
        y = df['credit_risk']

        # Dividir datos ANTES de preprocesar para evitar fuga de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=params.get('test_size', 0.2), # Leer tamaño de test desde params
            random_state=params.get('random_state', 42), # Leer semilla aleatoria desde params
            stratify=y # Mantener proporción de clases en la división
        )
        console.print("[bold bright_green][SUCCESS][/bold bright_green] Datos divididos en conjuntos de entrenamiento y prueba.")

        # --- DEFINIR PREPROCESAMIENTO ---
        console.print("\n[bold green][INFO][/bold green] Definiendo pipeline de preprocesamiento...")
        # Identificar columnas numéricas y categóricas en el DataFrame original (X_train)
        num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

        console.print(f"  -> Columnas numéricas detectadas: {len(num_cols)}")
        console.print(f"  -> Columnas categóricas detectadas: {len(cat_cols)}")

        # Transformador para columnas numéricas: imputar mediana
        num_transformer = SimpleImputer(strategy="median")
        # Transformador para columnas categóricas: imputar moda y luego One-Hot Encoding
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) # sparse_output=False devuelve un array numpy denso
        ])

        # Combinar transformadores usando ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num_cols),
                ("cat", cat_transformer, cat_cols),
            ],
            remainder="drop" # Descarta columnas no especificadas si las hubiera
        )

        # --- PASO 2: Entrenamiento del Modelo ---
        console.print("\n[bold green][INFO][/bold green] PASO 2: Entrenando el modelo XGBoost dentro del Pipeline...")
        # Parámetros fijos para el modelo XGBoost (se podrían leer de params.yaml)
        fixed_params = {
            'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'eval_metric': 'logloss', # Métrica para evaluación interna de XGBoost
            'random_state': params.get('random_state', 42)
        }
        console.print("Usando los siguientes parámetros fijos para XGBoost:")
        console.print(fixed_params)

        xgb_clf = xgb.XGBClassifier(**fixed_params)

        # Crear el Pipeline completo: (1) preprocesamiento -> (2) clasificador XGBoost
        model = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("clf", xgb_clf)
        ])

        # Entrenar el Pipeline completo con los datos de entrenamiento
        model.fit(X_train, y_train)
        console.print("[bold bright_green][SUCCESS][/bold bright_green] Pipeline entrenado.")

        # --- PASO 3: Evaluación del Modelo ---
        console.print("\n[bold green][INFO][/bold green] PASO 3: Evaluando el modelo...")
        # Realizar predicciones sobre el conjunto de prueba
        y_pred = model.predict(X_test)
        # Obtener probabilidades para la clase positiva (necesario para AUC y ROC)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calcular métricas de clasificación
        report = classification_report(y_test, y_pred, output_dict=True)
        # Usar .get() para evitar errores si una clase no tiene predicciones (ej. si f1=0)
        f1 = report.get('1', {}).get('f1-score', 0.0)
        accuracy = report.get('accuracy', 0.0)
        precision = report.get('1', {}).get('precision', 0.0)
        recall = report.get('1', {}).get('recall', 0.0)
        auc = roc_auc_score(y_test, y_pred_proba)
        # Calcular métrica personalizada: Proporción de predicciones "malas" (clase 0)
        bad_rate = np.mean(y_pred == 0)

        # Crear y mostrar tabla de métricas con Rich
        metrics_table = Table(title="📊 Métricas de Evaluación (Conjunto de Prueba)")
        metrics_table.add_column("Métrica", style="cyan", no_wrap=True)
        metrics_table.add_column("Valor", style="magenta")
        metrics_table.add_row("F1-Score (Clase 1)", f"{f1:.4f}")
        metrics_table.add_row("Accuracy", f"{accuracy:.4f}")
        metrics_table.add_row("AUC", f"{auc:.4f}")
        metrics_table.add_row("Precision (Clase 1)", f"{precision:.4f}")
        metrics_table.add_row("Recall (Clase 1)", f"{recall:.4f}")
        metrics_table.add_row("Bad Rate Predicho", f"{bad_rate:.4f}")
        console.print(metrics_table) # Imprime la tabla formateada

        # --- PASO 4: Generación de Gráficas de Evaluación ---
        console.print("\n[bold green][INFO][/bold green] PASO 4: Generando gráficas de evaluación...")
        plots_output.mkdir(parents=True, exist_ok=True) # Asegurar que el directorio existe

        # Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malo (0)', 'Bueno (1)'], yticklabels=['Malo (0)', 'Bueno (1)']);
        plt.title('Matriz de Confusión'); plt.ylabel('Verdadero'); plt.xlabel('Predicho');
        confusion_matrix_path = plots_output / "confusion_matrix.png"
        plt.savefig(confusion_matrix_path); plt.close(); # Guardar y cerrar la figura

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc:.2f})');
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); # Línea de referencia aleatoria
        plt.xlabel('Tasa de Falsos Positivos'); plt.ylabel('Tasa de Verdaderos Positivos'); plt.title('Curva ROC'); plt.legend(loc="lower right");
        roc_curve_path = plots_output / "roc_curve.png"
        plt.savefig(roc_curve_path); plt.close(); # Guardar y cerrar la figura
        console.print(f"[bold bright_green][SUCCESS][/bold bright_green] Gráficas de evaluación guardadas en: {plots_output}")

        # --- PASO 5: Análisis de Interpretabilidad con SHAP (Adaptado para Pipeline) ---
        console.print("\n[bold green][INFO][/bold green] PASO 5: Realizando análisis SHAP...")

        # 1) Extraer preprocesador y clasificador ya entrenados desde el pipeline
        fitted_preprocessor = model.named_steps["preprocessor"]
        fitted_classifier = model.named_steps["clf"]

        # 2) Transformar el conjunto de prueba X_test usando el preprocesador entrenado
        X_test_transformed = fitted_preprocessor.transform(X_test)
        try:
            # Intentar obtener los nombres de las features DESPUÉS de la transformación (incluye las creadas por OneHotEncoder)
            feature_names_out = fitted_preprocessor.get_feature_names_out()
        except Exception:
             # Si falla (ej. versión antigua de sklearn), usar nombres genéricos
            console.print("[yellow]Advertencia:[/yellow] No se pudieron obtener nombres de features del preprocesador. Usando genéricos.")
            feature_names_out = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]

        # Convertir el array transformado de vuelta a DataFrame para SHAP (mejor visualización)
        X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names_out)

        # 3) Crear el explainer SHAP sobre el CLASIFICADOR entrenado (no sobre todo el pipeline)
        console.print("  -> Calculando valores SHAP...")
        try:
            # TreeExplainer es específico para modelos basados en árboles como XGBoost
            explainer = shap.TreeExplainer(fitted_classifier)
            # check_additivity=False puede ayudar a evitar errores en algunas versiones
            shap_values = explainer.shap_values(X_test_transformed, check_additivity=False)
        except Exception as e:
            console.print(f"[bold red]Error al calcular SHAP con TreeExplainer:[/bold red] {e}. Intentando con explainer genérico.")
            # Fallback a Explainer genérico si TreeExplainer falla
            explainer = shap.Explainer(fitted_classifier, X_test_transformed_df)
            shap_values = explainer(X_test_transformed_df).values

        # 4) En clasificación binaria, shap_values puede ser una lista [shap_clase_0, shap_clase_1] o un solo array.
        # Nos quedamos con los valores SHAP para la clase positiva (clase 1 = 'Bueno')
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_pos_class = shap_values[1]
        else:
            shap_values_pos_class = shap_values

        # 5) Calcular y mostrar la importancia global de las features
        console.print("  -> Top 10 features más importantes (SHAP):")
        # Crear DataFrame con los valores SHAP
        shap_df = pd.DataFrame(shap_values_pos_class, columns=feature_names_out)
        # Calcular la media del valor absoluto SHAP por feature
        vals = np.abs(shap_df.values).mean(0)
        # Crear DataFrame de importancia y ordenar
        shap_importance = pd.DataFrame(
            list(zip(feature_names_out, vals)),
            columns=['feature', 'importance']
        ).sort_values(by='importance', ascending=False)
        # Imprimir las 10 más importantes
        console.print(shap_importance.head(10).to_string(index=False))

        # 6) Generar y guardar las gráficas SHAP
        # Gráfica de barras de importancia
        shap.summary_plot(shap_values_pos_class, X_test_transformed_df, plot_type="bar", show=False)
        plt.title("Importancia de Features (SHAP | media abs)"); plt.tight_layout(); # Ajustar layout
        shap_importance_path = plots_output / "shap_importance_plot.png"
        plt.savefig(shap_importance_path); plt.close();

        # Gráfica de resumen (beeswarm plot)
        shap.summary_plot(shap_values_pos_class, X_test_transformed_df, show=False)
        plt.tight_layout(); # Ajustar layout
        shap_summary_path = plots_output / "shap_summary_plot.png"
        plt.savefig(shap_summary_path); plt.close();
        console.print(f"[bold bright_green][SUCCESS][/bold bright_green] Gráficas SHAP guardadas en: {plots_output}")

        # --- PASO 6: Registro en MLflow y Guardado para DVC ---
        console.print("\n[bold green][INFO][/bold green] PASO 6: Registrando artefactos y métricas...")

        # Registrar los parámetros usados por el clasificador XGBoost
        mlflow.log_params(fitted_classifier.get_params())

        # Registrar las métricas calculadas
        mlflow.log_metrics({
            'f1_score_test': f1, 'accuracy_test': accuracy, 'auc_test': auc,
            'precision_test': precision, 'recall_test': recall, 'bad_rate_test': bad_rate
        })

        # Registrar el PIPELINE COMPLETO de Sklearn (incluye preprocesador y modelo)
        # Esto es lo ideal para poder cargar y usar el pipeline directamente para predicciones futuras
        mlflow.sklearn.log_model(sk_model=model, artifact_path="sklearn-pipeline")
        
        # Opcional: Registrar SOLO el modelo XGBoost si se desea (puede ser útil para análisis específicos)
        # mlflow.xgboost.log_model(xgb_model=fitted_classifier, artifact_path="xgboost-model")

        # Registrar todas las gráficas generadas como artefactos en MLflow, dentro de la carpeta "plots"
        mlflow.log_artifact(confusion_matrix_path, "plots")
        mlflow.log_artifact(roc_curve_path, "plots")
        mlflow.log_artifact(shap_importance_path, "plots")
        mlflow.log_artifact(shap_summary_path, "plots")

        # Guardar el PIPELINE COMPLETO como archivo .pkl para DVC
        with open(model_output, 'wb') as f:
            pickle.dump(model, f)
        console.print(f" -> Pipeline completo guardado para DVC en: {model_output}")

        # Guardar las métricas calculadas como archivo .json para DVC
        metrics = {
            'f1_score_test': f1, 'accuracy_test': accuracy, 'auc_test': auc,
            'precision_test': precision, 'recall_test': recall, 'bad_rate_test': bad_rate,
            'params': fitted_classifier.get_params() # Guardar también los parámetros del clasificador en el JSON
        }
        with open(metrics_output, 'w') as f:
            # Usar default=str por si algún parámetro no es serializable directamente a JSON
            json.dump(metrics, f, indent=4, default=str)
        console.print(f" -> Métricas guardadas para DVC en: {metrics_output}")
        console.print("[bold bright_green][SUCCESS][/bold bright_green] Modelo, métricas y gráficas registradas y guardadas.")


# --- Bloque de ejecución principal (se ejecuta si corres el script directamente) ---
if __name__ == '__main__':
    # Configurar el parser para argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Script completo para entrenar un Pipeline Sklearn con XGBoost.")
    parser.add_argument("--input-data", type=str, required=True, help="Ruta al CSV de datos limpios.")
    parser.add_argument("--model-output", type=str, required=True, help="Ruta para guardar el pipeline .pkl.")
    parser.add_argument("--metrics-output", type=str, required=True, help="Ruta para guardar las métricas en JSON.")
    parser.add_argument("--plots-output", type=str, required=True, help="Directorio para guardar las gráficas de evaluación.")
    
    # Leer los argumentos pasados desde la línea de comandos (DVC lo hará automáticamente)
    args = parser.parse_args()
    
    # Parámetros generales del pipeline (estos idealmente se leerían de params.yaml,
    # pero DVC los pasa a través de la sección 'params' de dvc.yaml)
    console.print("\n[INFO] Cargando parámetros del pipeline (¡simulado para ejecución directa!)...")
    # Usamos valores por defecto aquí; DVC usará los de params.yaml
    train_params = {'test_size': 0.2, 'random_state': 42} 
    console.print(train_params)

    # Llamar a la función principal de entrenamiento
    train_model(
        input_data=Path(args.input_data),
        model_output=Path(args.model_output),
        metrics_output=Path(args.metrics_output),
        plots_output=Path(args.plots_output),
        params=train_params # Pasar los parámetros generales a la función
    )