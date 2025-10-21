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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Ignorar advertencias futuras para una salida más limpia en la consola
warnings.filterwarnings("ignore", category=FutureWarning)

def train_model(input_data: Path, model_output: Path, metrics_output: Path, plots_output: Path, params: dict):
    """
    Función completa para entrenar, evaluar y registrar un modelo XGBoost.
    """
    # Configurar y nombrar el experimento en MLflow
    mlflow.set_experiment("German Credit XGBoost")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{now}"

    # Iniciar un "Run" de MLflow para registrar todo en un solo experimento
    with mlflow.start_run(run_name=run_name):
        
        print("\n" + "="*50)
        print(f" Iniciando Run: {run_name} ".center(50, "="))
        print("="*50)

        # --- PASO 1: Carga y Preparación de Datos ---
        print("\n[INFO] PASO 1: Cargando y preparando datos...")
        df = pd.read_csv(input_data)
        
        X = df.drop(columns='credit_risk')
        y = df['credit_risk']
        X = pd.get_dummies(X, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=params.get('test_size', 0.2),
            random_state=params.get('random_state', 42),
            stratify=y
        )
        print("[SUCCESS] Datos divididos en conjuntos de entrenamiento y prueba.")

        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X_train.columns if c not in num_cols]

        num_transformer = SimpleImputer(strategy="median")  # XGB no requiere escalar
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num_cols),
                ("cat", cat_transformer, cat_cols),
            ],
            remainder="drop"
        )
        
        # --- PASO 2: Entrenamiento del Modelo ---
        print("\n[INFO] PASO 2: Entrenando el modelo XGBoost...")
        fixed_params = {
            'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'random_state': params.get('random_state', 42)
        }

        xgb_clf = xgb.XGBClassifier(**fixed_params)

        # Pipeline total: preprocesamiento + clasificador
        #model = Pipeline(steps=[
           # ("preprocessor", preprocessor),
           # ("clf", xgb_clf)
        #])

        
        model = xgb.XGBClassifier(**fixed_params)
        model.fit(X_train, y_train)
        print("[SUCCESS] Modelo entrenado.")
        
        # --- PASO 3: Evaluación del Modelo ---
        print("\n[INFO] PASO 3: Evaluando el modelo...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report['1']['f1-score']
        accuracy = report['accuracy']
        auc = roc_auc_score(y_test, y_pred_proba)
        bad_rate = np.mean(y_pred == 0)

        print(f"  -> F1-Score: {f1:.4f}")
        print(f"  -> Accuracy: {accuracy:.4f}")
        print(f"  -> AUC: {auc:.4f}")
        print(f"  -> Bad Rate: {bad_rate:.4f}")

        # --- PASO 4: Generación de Gráficas de Evaluación ---
        print("\n[INFO] PASO 4: Generando gráficas de evaluación...")
        plots_output.mkdir(parents=True, exist_ok=True)
        
        # Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malo', 'Bueno'], yticklabels=['Malo', 'Bueno']);
        plt.title('Matriz de Confusión'); plt.ylabel('Verdadero'); plt.xlabel('Predicho');
        confusion_matrix_path = plots_output / "confusion_matrix.png"
        plt.savefig(confusion_matrix_path); plt.close();
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc:.2f})');
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--');
        plt.xlabel('Tasa de Falsos Positivos'); plt.ylabel('Tasa de Verdaderos Positivos'); plt.title('Curva ROC'); plt.legend(loc="lower right");
        roc_curve_path = plots_output / "roc_curve.png"
        plt.savefig(roc_curve_path); plt.close();
        print(f"[SUCCESS] Gráficas de evaluación guardadas en: {plots_output}")

        # --- PASO 5: Análisis de Interpretabilidad con SHAP ---
        print("\n[INFO] PASO 5: Realizando análisis SHAP...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        print("  -> Top 10 features más importantes (SHAP):")
        feature_names = X_test.columns
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        vals = np.abs(shap_df.values).mean(0)
        shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['feature', 'importance']).sort_values(by=['importance'], ascending=False)
        print(shap_importance.head(10).to_string(index=False))

        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title("Importancia de Features (Valor SHAP Absoluto Medio)"); plt.tight_layout();
        shap_importance_path = plots_output / "shap_importance_plot.png"
        plt.savefig(shap_importance_path); plt.close();

        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout();
        shap_summary_path = plots_output / "shap_summary_plot.png"
        plt.savefig(shap_summary_path); plt.close();
        print(f"[SUCCESS] Gráficas SHAP guardadas en: {plots_output}")
        
        # --- PASO 6: Registro en MLflow y Guardado para DVC ---
        print("\n[INFO] PASO 6: Registrando artefactos y métricas...")
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({'f1_score_test': f1, 'accuracy_test': accuracy, 'auc_test': auc, 'bad_rate_test': bad_rate})
        mlflow.xgboost.log_model(model, "xgboost-model")
        mlflow.log_artifact(confusion_matrix_path, "plots")
        mlflow.log_artifact(roc_curve_path, "plots")
        mlflow.log_artifact(shap_importance_path, "plots")
        mlflow.log_artifact(shap_summary_path, "plots")
        
        with open(model_output, 'wb') as f:
            pickle.dump(model, f)
        
        metrics = {'f1_score_test': f1, 'accuracy_test': accuracy, 'auc_test': auc, 'bad_rate_test': bad_rate, 'params': model.get_params()}
        with open(metrics_output, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        print("[SUCCESS] Modelo, métricas y gráficas registradas y guardadas.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script completo para entrenar un modelo XGBoost.")
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--model-output", type=str, required=True)
    parser.add_argument("--metrics-output", type=str, required=True)
    parser.add_argument("--plots-output", type=str, required=True)
    
    args = parser.parse_args()
    
    # Parámetros generales del pipeline que DVC leerá de params.yaml
    train_params = {'test_size': 0.2, 'random_state': 42}
    
    train_model(
        input_data=Path(args.input_data),
        model_output=Path(args.model_output),
        metrics_output=Path(args.metrics_output),
        plots_output=Path(args.plots_output),
        params=train_params
    )