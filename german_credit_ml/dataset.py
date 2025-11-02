from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from german_credit_ml.clean import cast_types, impute_and_enforce, normalize_credit_risk
from german_credit_ml.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "german_credit_modified.csv",
    output_path: Path = PROCESSED_DATA_DIR / "german_credit_clean.csv",
):
    """Procesa el dataset raw -> processed con limpieza incluida."""
    logger.info(f"Leyendo datos crudos desde: {input_path}")
    df = pd.read_csv(input_path)

    # aplicar limpieza
    df = normalize_credit_risk(df)
    df = cast_types(df)
    df = impute_and_enforce(df)

    logger.info("Guardando datos procesados...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Dataset limpio guardado en {output_path}, filas={len(df)}")


if __name__ == "__main__":
    app()
