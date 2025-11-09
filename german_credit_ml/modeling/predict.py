# german_credit_ml/modeling/predict.py

from pathlib import Path
import pickle

import pandas as pd
import typer
from loguru import logger

from german_credit_ml.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def load_pipeline(model_path: Path):
    logger.info(f"Cargando modelo desde {model_path}...")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def run_inference(
    input_data: Path,
    model_path: Path,
    output_path: Path,
):
    # 1) Cargar datos nuevos (sin credit_risk)
    df = pd.read_csv(input_data)
    logger.info(f"Datos de entrada leídos: {df.shape[0]} filas, {df.shape[1]} columnas")

    # 2) Cargar pipeline (preprocesador + modelo)
    pipeline = load_pipeline(model_path)

    # 3) Generar predicciones
    preds = pipeline.predict(df)

    # 4) Guardar resultados
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    out_df["prediction"] = preds
    out_df.to_csv(output_path, index=False)

    logger.success(f"Predicciones guardadas en {output_path}")
    return out_df


@app.command()
def main(
    input_data: Path = PROCESSED_DATA_DIR / "german_credit_clean.csv",
    model_path: Path = MODELS_DIR / "xgboost_model.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
):
    """Inferencia batch: CSV de entrada -> CSV con columna 'prediction'."""
    run_inference(input_data, model_path, output_path)


if __name__ == "__main__":
    app()
