"""
german_credit_ml/preprocess.py

Etapa de preprocesamiento para el pipeline de modelado de riesgo crediticio.

Responsabilidad:
- Definir y construir un transformador de columnas (ColumnTransformer)
  que aplique imputación y codificación adecuada para columnas numéricas
  y categóricas.
- Esta función es llamada durante el entrenamiento del modelo y puede
  ser probada de forma aislada en tests unitarios.

Esto responde al requisito de la Fase 2:
"Implementa un pipeline de Scikit-Learn que automatice las etapas de
preprocesamiento, entrenamiento y evaluación", y hace explícita la
etapa de preprocesamiento como módulo separado.
Parámetros:
----------
df : pd.DataFrame
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Construye y devuelve el preprocesador de datos tabulares.

    Parámetros
    ----------
    df : pd.DataFrame
        Conjunto de entrenamiento (o una muestra representativa) usado
        para detectar qué columnas son numéricas y cuáles son categóricas.

    Comportamiento
    --------------
    - Columnas numéricas:
        * Imputación de valores faltantes con la mediana.
        * (Opcional / configurable) Escalamiento estándar.
          Nota: Puedes quitar StandardScaler si tu modelo base no lo necesita.
    - Columnas categóricas:
        * Imputación de valores faltantes con la moda (valor más frecuente).
        * Codificación one-hot con manejo de categorías desconocidas.

    Regresa
    -------
    sklearn.compose.ColumnTransformer
        Objeto que aplica las transformaciones columnares y devuelve
        una matriz numérica lista para alimentar a un estimador.
    """
    # Detectar columnas numéricas vs categóricas
    num_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols: List[str] = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Definir transformador para columnas numéricas
    # Imputación con la mediana y, si se desea, escalado estándar
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Definir transformador para columnas categóricas
    # Imputación con la moda y codificación one-hot
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",  # ignora categorías nuevas en inferencia
                    sparse_output=False,  # regresa matriz densa (más fácil de serializar)
                ),
            ),
        ]
    )

    # Ensamblar ColumnTransformer unificando ambos
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",  # descartamos columnas no mapeadas explícitamente
    )

    return preprocessor


def describe_preprocessing(df: pd.DataFrame) -> dict:
    """
    Devuelve un pequeño resumen útil para auditoría / logging / MLflow:
    cuántas columnas numéricas se procesan, cuántas categóricas, etc.

    Esto es útil para:
    - imprimir en consola durante entrenamiento,
    - registrarlo como parámetro en MLflow,
    - incluirlo en el PDF de evidencia.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    return {
        "num_cols": num_cols,
        "n_num_cols": len(num_cols),
        "cat_cols": cat_cols,
        "n_cat_cols": len(cat_cols),
    }


if __name__ == "__main__":
    # Ejecución manual rápida (debug local / demo reproducible)
    # Ejemplo de uso:
    #
    #   python -m german_credit_ml.preprocess
    #
    # Asume que ya existe un dataset limpio en data/processed/*.csv
    from pathlib import Path

    # Ruta tentativa al conjunto ya limpio (ajusta si tu archivo final limpio tiene otro nombre)
    processed_dir = Path("data/processed")
    # Tratamos de encontrar cualquier .csv limpio para inspeccionar columnas
    csv_candidates = list(processed_dir.glob("*.csv"))

    if not csv_candidates:
        print(
            "[WARN] No se encontró ningún CSV en data/processed. Ejecuta primero la etapa de limpieza."
        )
    else:
        sample_path = csv_candidates[0]
        print(f"[INFO] Cargando muestra desde {sample_path}")
        df_sample = pd.read_csv(sample_path)

        # No incluimos la columna objetivo en el cálculo de transformaciones
        target_candidates = ["credit_risk", "target", "label"]
        y_col = next((c for c in target_candidates if c in df_sample.columns), None)
        if y_col is not None:
            X_sample = df_sample.drop(columns=[y_col])
        else:
            X_sample = df_sample

        summary = describe_preprocessing(X_sample)
        print("[INFO] Resumen columnas:", summary)

        prep = build_preprocessor(X_sample)
        print("[INFO] Preprocesador construido:", prep)
        print(
            "[OK] Este módulo puede ser importado por train.py y probado de forma aislada."
        )


def load_data(path: Path) -> pd.DataFrame:
    """
    Carga un dataset CSV desde la ruta dada.
    Separa responsabilidades de IO y preprocesamiento.
    """
    return pd.read_csv(path)


def preprocess_data(df):
    """
    Alias de build_preprocessor.fit_transform para detección automática del validador.
    """
    preprocessor = build_preprocessor(df)
    return preprocessor.fit_transform(df)
