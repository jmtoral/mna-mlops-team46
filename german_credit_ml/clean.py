# german_credit_ml/clean.py
# python -m german_credit_ml.clean --input data/raw/german_credit_modified.csv --output data/processed/german_credit_clean.csv

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats.mstats import winsorize

LOGGER = logging.getLogger(__name__)

COLUMN_MAPPING = {
    'laufkont': 'status', 'laufzeit': 'duration', 'moral': 'credit_history',
    'verw': 'purpose', 'hoehe': 'amount', 'sparkont': 'savings',
    'beszeit': 'employment_duration', 'rate': 'installment_rate',
    'famges': 'personal_status_sex', 'buerge': 'other_debtors',
    'wohnzeit': 'present_residence', 'verm': 'property', 'alter': 'age',
    'weitkred': 'other_installment_plans', 'wohn': 'housing',
    'bishkred': 'number_credits', 'beruf': 'job', 'pers': 'people_liable',
    'telef': 'telephone', 'gastarb': 'foreign_worker', 'kredit': 'credit_risk'
}

CATEGORIES_MAP = {
    "status": [1, 2, 3, 4], "credit_history": [0, 1, 2, 3, 4],
    "purpose": list(range(0, 11)), "savings": [1, 2, 3, 4, 5],
    "employment_duration": [1, 2, 3, 4, 5], "installment_rate": [1, 2, 3, 4],
    "personal_status_sex": [1, 2, 3, 4], "other_debtors": [1, 2, 3],
    "present_residence": [1, 2, 3, 4], "property": [1, 2, 3, 4],
    "other_installment_plans": [1, 2, 3], "housing": [1, 2, 3],
    "job": [1, 2, 3, 4], "telephone": [1, 2], "foreign_worker": [1, 2],
}

NUMERIC_COLS = ["duration", "amount", "age"]
TARGET_COL = "credit_risk"


def load_yaml(path: Path | str) -> dict:
    """Carga un archivo YAML y devuelve su contenido como diccionario."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DataCleaner:
    """Clase para limpiar el dataset German Credit."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean_dataframe(self) -> pd.DataFrame:
        """Aplica todo el pipeline de limpieza."""

        LOGGER.info("Paso 1: Renombrando columnas y eliminando columnas innecesarias...")
        self.rename_columns()
        self.drop_mixed_columns()

        LOGGER.info("Paso 2: Normalizando y filtrando la variable objetivo 'credit_risk'...")
        self.convert_numeric()
        self.normalize_target()

        LOGGER.info("Paso 3: Imputando valores faltantes usando la mediana/moda del dataset...")
        self.impute_missing()

        LOGGER.info("Paso 4: Aplicando Winsorización a las columnas numéricas...")
        self.winsorize_numeric()

        LOGGER.info("Paso 5: Aplicando tipos de datos finales y categorías...")
        self.enforce_types()

        return self.df

    def rename_columns(self):
        self.df = self.df.rename(columns=COLUMN_MAPPING)

    def drop_mixed_columns(self):
        if 'mixed_type_col' in self.df.columns:
            self.df = self.df.drop(columns=['mixed_type_col'])

    def convert_numeric(self):
        for col in self.df.columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def normalize_target(self):
        self.df[TARGET_COL] = self.df[TARGET_COL].replace({1.0: 1, 2.0: 0}).astype(float)
        self.df = self.df[self.df[TARGET_COL].isin([0.0, 1.0])].copy()

    def impute_missing(self):
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if col in NUMERIC_COLS:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif col in CATEGORIES_MAP:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        self.df = self.df.fillna(0)

    def winsorize_numeric(self):
        for col in NUMERIC_COLS:
            self.df[col] = winsorize(self.df[col], limits=[0.01, 0.01])

    def enforce_types(self):
        for col in NUMERIC_COLS:
            self.df[col] = self.df[col].astype("int64")

        for col, valid_cats in CATEGORIES_MAP.items():
            self.df[col] = self.df[col].clip(lower=min(valid_cats), upper=max(valid_cats))
            self.df[col] = self.df[col].astype("category")

        self.df[TARGET_COL] = self.df[TARGET_COL].astype("int64")

        other_int_cols = ['number_credits', 'people_liable']
        for col in other_int_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('int64')

    @staticmethod
    def final_validation(df: pd.DataFrame):
        """Realiza la validación final del DataFrame limpio."""
        LOGGER.info("Realizando validación final del DataFrame...")
        assert df.isnull().sum().sum() == 0, "Error: Aún existen valores nulos."
        uniq_targets = set(df[TARGET_COL].unique())
        assert uniq_targets.issubset({0, 1}), (
            f"Error: La columna target contiene valores no binarios: {uniq_targets}"
        )
        LOGGER.info("Validación exitosa: No hay valores nulos y target es binario.")


def run_clean(input_path: Path, output_path: Path):
    LOGGER.info(f"Leyendo datos originales desde: {input_path}")
    df = pd.read_csv(input_path)

    cleaner = DataCleaner(df)
    df_cleaned = cleaner.clean_dataframe()
    cleaner.final_validation(df_cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    LOGGER.info(f"Datos limpios guardados en: {output_path} (Shape: {df_cleaned.shape})")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Script de limpieza para el dataset German Credit.")
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración YAML (ej. params.yaml).")
    parser.add_argument("--input", type=str, help="Ruta al CSV de entrada original (sobrescribe config).")
    parser.add_argument("--output", type=str, help="Ruta para guardar el CSV limpio (sobrescribe config).")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de logging."
    )
    return parser


def main():
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    if args.config:
        cfg = load_yaml(args.config)
        input_path = Path(args.input or cfg["data_load"]["raw"])
        output_path = Path(args.output or cfg["data_load"]["processed"])
    elif args.input and args.output:
        input_path = Path(args.input)
        output_path = Path(args.output)
    else:
        raise SystemExit("Error: Debe proporcionar --config o ambos --input y --output.")

    run_clean(input_path, output_path)


if __name__ == "__main__":
    main()
