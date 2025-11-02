from pathlib import Path
import pickle

import pandas as pd

from .train import Evaluator, ShapInterpreter  # reutilizamos las clases existentes


def evaluate_model(model_path: Path, test_csv: Path, target_col: str, output_dir: Path):
    """
    Evalúa un modelo ya entrenado contra un conjunto de prueba y
    genera métricas, curvas, matriz de confusión y análisis SHAP.
    Args:
        model_path (Path): Ruta al archivo del modelo entrenado (pickle).
        test_csv (Path): Ruta al archivo CSV del conjunto de prueba.
        target_col (str): Nombre de la columna objetivo en el conjunto de prueba.
        output_dir (Path): Directorio donde se guardarán los artefactos de evaluación.
    Returns:
        dict: Métricas de evaluación calculadas.
    """
    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=target_col)
    y_test = df_test[target_col]

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = Evaluator.compute_metrics(y_test, y_pred, y_proba)

    output_dir.mkdir(parents=True, exist_ok=True)
    Evaluator.plot_confusion_matrix(y_test, y_pred, output_dir / "confusion_matrix.png")
    Evaluator.plot_roc(
        y_test, y_proba, output_dir / "roc_curve.png", metrics.get("auc_test", 0.0)
    )
    ShapInterpreter.explain(pipeline, X_test, output_dir)

    return metrics


if __name__ == "__main__":
    '''
    Ejecuta la evaluación del modelo con parámetros de ejemplo.
    Ajusta las rutas según sea necesario para tu entorno local. 
    '''
    # Ejemplo de ejecución directa para validación local / demo:
    from pathlib import Path

    metrics = evaluate_model(
        model_path=Path("models/model.pkl"),
        test_csv=Path("data/processed/test.csv"),
        target_col="credit_risk",
        output_dir=Path("reports/eval_artifacts"),
    )
    print("[INFO] Métricas de evaluación:", metrics)
