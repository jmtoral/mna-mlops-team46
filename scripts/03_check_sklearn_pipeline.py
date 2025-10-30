#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validador de buenas prácticas del pipeline con Scikit-Learn.

Revisa estáticamente (AST) en el código del proyecto:
- Pipeline/ColumnTransformer presentes
- Transformadores recomendados (SimpleImputer, StandardScaler, OneHotEncoder)
- Entrenamiento/evaluación: train_test_split, métricas sklearn.metrics
- Validación y tuning: cross_val_score, GridSearchCV/RandomizedSearchCV (recomendado)
- Reproducibilidad: uso de random_state
- Persistencia/registro: joblib.dump o mlflow.sklearn.log_model / mlflow.autolog (recomendado)
- Documentación/claridad: docstrings en funciones clave, lectura de params.yaml

Estados:
  [PRESENTE], [FALTA-REQUERIDO], [FALTA-RECOMENDADO], [FALTA-OPCIONAL]

Uso:
  python 03_check_sklearn_pipeline.py ^
    --project-root "C:\\dev\\mna-mlops-team46" ^
    --module-name german_credit_ml
"""

import os
import ast
import glob
import argparse
from typing import List, Dict, Tuple, Optional
import sys
import io

# Forzar stdout/stderr a UTF-8 en Windows para caracteres especiales
if sys.stdout.encoding is None or "cp125" in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding is None or "cp125" in sys.stderr.encoding.lower():
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ---------- CLI / entorno ----------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Validador de buenas prácticas del pipeline Scikit-Learn"
    )
    parser.add_argument(
        "--project-root",
        default=os.getenv("PROJECT_ROOT", r"C:\dev\mna-mlops-team46"),
        help="Ruta absoluta del proyecto (raíz del repo)",
    )
    parser.add_argument(
        "--module-name",
        default=os.getenv("MODULE_NAME", "german_credit_ml"),
        help="Nombre del paquete Python principal (solo para reporte descriptivo)",
    )
    return parser.parse_args()

ARGS = parse_args()

ROOT = os.path.abspath(os.path.expanduser(ARGS.project_root))
if not os.path.isdir(ROOT):
    raise SystemExit(f"[ERROR] Project root no existe: {ROOT}")
os.chdir(ROOT)

MODULE_NAME = ARGS.module_name.strip() or "<package_name>"

REQUIRED = "requerido"
RECOMMENDED = "recomendado"
OPTIONAL = "opcional"


# -------- utilidades generales --------
def pad_to_column(text: str, col: int) -> str:
    return text if len(text) >= col else text + " " * (col - len(text))


def tag(present: bool, importance: str) -> str:
    if present:
        return "[PRESENTE]"
    if importance == REQUIRED:
        return "[FALTA-REQUERIDO]"
    if importance == RECOMMENDED:
        return "[FALTA-RECOMENDADO]"
    return "[FALTA-OPCIONAL]"


def print_header(title: str):
    print("\n" + title)
    print("-" * len(title))


# -------- helpers de AST --------
def load_ast(py_file: str) -> Optional[ast.AST]:
    try:
        with open(py_file, "r", encoding="utf-8") as f:
            return ast.parse(f.read(), filename=py_file)
    except Exception:
        return None


def list_functions(tree: ast.AST) -> List[ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def list_calls(tree: ast.AST) -> List[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def dotted_name(node: ast.AST) -> str:
    """
    Devuelve el nombre "accesible" de la llamada:
    - sklearn.pipeline.Pipeline
    - pipeline.Pipeline
    - OneHotEncoder
    - mlflow.sklearn.log_model
    etc.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{dotted_name(node.value)}.{node.attr}"
    return ""


def call_name(call: ast.Call) -> str:
    return dotted_name(call.func)


def kwarg_present(call: ast.Call, kw: str) -> bool:
    return any((isinstance(a, ast.keyword) and a.arg == kw) for a in call.keywords)


# -------- análisis por archivo --------
def scan_file(py_file: str) -> Dict[str, bool]:
    """
    Para cada archivo Python detectamos indicadores de buenas prácticas:
    - pipeline: uso de Pipeline/make_pipeline
    - column_transformer: uso de ColumnTransformer/make_column_transformer
    - has_simple_imputer / has_standard_scaler / has_onehot
    - tts (train_test_split)
    - metrics (llamadas a sklearn.metrics.*)
    - cv (cross_val_score)
    - searchcv (GridSearchCV / RandomizedSearchCV)
    - random_state (reproducibilidad)
    - persist (joblib.dump / pickle.dump)
    - mlflow (mlflow.autolog / mlflow.sklearn.log_model)
    - yaml_params (lectura de params.yaml vía yaml.safe_load/open)
    - doc_ratio (docstrings por función)
    """
    flags = {
        "pipeline": False,
        "column_transformer": False,
        "has_simple_imputer": False,
        "has_standard_scaler": False,
        "has_onehot": False,
        "tts": False,
        "metrics": False,
        "cv": False,
        "searchcv": False,
        "random_state": False,
        "persist": False,
        "mlflow": False,
        "yaml_params": False,
        "doc_ratio": 0.0,
        "n_funcs": 0,
    }

    tree = load_ast(py_file)
    if not tree:
        return flags

    calls = list_calls(tree)

    # doc_ratio
    funcs = list_functions(tree)
    flags["n_funcs"] = len(funcs)
    if funcs:
        doc_count = sum(ast.get_docstring(f) is not None for f in funcs)
        flags["doc_ratio"] = doc_count / len(funcs)

    metric_names = set()

    for c in calls:
        name = call_name(c)

        # Pipeline
        if name.endswith("sklearn.pipeline.Pipeline") or \
           name.endswith("pipeline.Pipeline") or \
           name.endswith("make_pipeline"):
            flags["pipeline"] = True

        # ColumnTransformer
        if name.endswith("sklearn.compose.ColumnTransformer") or \
           name.endswith("compose.ColumnTransformer") or \
           name.endswith("make_column_transformer"):
            flags["column_transformer"] = True

        # --- Transformadores típicos (mejorado) ---
        # SimpleImputer
        if (
            name.endswith("sklearn.impute.SimpleImputer")
            or name.endswith("impute.SimpleImputer")
            or name.endswith("SimpleImputer")
        ):
            flags["has_simple_imputer"] = True

        # StandardScaler
        if (
            name.endswith("sklearn.preprocessing.StandardScaler")
            or name.endswith("preprocessing.StandardScaler")
            or name.endswith("StandardScaler")
        ):
            flags["has_standard_scaler"] = True

        # OneHotEncoder
        if (
            name.endswith("sklearn.preprocessing.OneHotEncoder")
            or name.endswith("preprocessing.OneHotEncoder")
            or name.endswith("OneHotEncoder")
        ):
            flags["has_onehot"] = True

        # train_test_split
        if (
            name.endswith("sklearn.model_selection.train_test_split")
            or name.endswith("model_selection.train_test_split")
            or name.endswith("train_test_split")
        ):
            flags["tts"] = True
            if kwarg_present(c, "random_state"):
                flags["random_state"] = True

        # cross_val_score
        if (
            name.endswith("sklearn.model_selection.cross_val_score")
            or name.endswith("model_selection.cross_val_score")
            or name.endswith("cross_val_score")
        ):
            flags["cv"] = True

        # GridSearchCV / RandomizedSearchCV
        if name.endswith("GridSearchCV") or name.endswith("RandomizedSearchCV"):
            flags["searchcv"] = True
            if kwarg_present(c, "random_state"):
                flags["random_state"] = True

        # Métricas (heurística): llamadas a sklearn.metrics.*
        if ".metrics." in name:
            metric_names.add(name)

        # random_state en cualquier llamada con ese kw
        if kwarg_present(c, "random_state"):
            flags["random_state"] = True

        # Persistencia de modelo
        if (
            name.endswith("joblib.dump")
            or name.endswith("sklearn.externals.joblib.dump")
            or name.endswith("pickle.dump")
        ):
            flags["persist"] = True

        # MLflow tracking/model registry
        if (
            name.endswith("mlflow.autolog")
            or name.endswith("mlflow.sklearn.log_model")
        ):
            flags["mlflow"] = True

        # Lectura de params.yaml
        if name.endswith("yaml.safe_load") or name.endswith("yaml.load"):
            flags["yaml_params"] = True
        if isinstance(c.func, ast.Name) and c.func.id == "open":
            for a in c.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and "params.yaml" in a.value:
                    flags["yaml_params"] = True

    flags["metrics"] = len(metric_names) > 0
    return flags


def scan_repo() -> Dict[str, Dict[str, bool]]:
    """
    Escanea código en ambos estilos:
    - estilo Cookiecutter DS v1: src/**.py
    - estilo Cookiecutter DS v2: <module_name>/**.py
    - scripts sueltos en raíz (*.py)
    """
    files = []
    files.extend(glob.glob("src/**/*.py", recursive=True))
    files.extend(glob.glob(f"{MODULE_NAME}/**/*.py", recursive=True))
    files.extend(glob.glob("*.py", recursive=False))

    results: Dict[str, Dict[str, bool]] = {}
    for f in sorted(set(files)):
        results[f] = scan_file(f)
    return results


# -------- agregación --------
def aggregate(results: Dict[str, Dict[str, bool]]) -> Dict[str, float]:
    agg = {k: False for k in [
        "pipeline","column_transformer","has_simple_imputer","has_standard_scaler","has_onehot",
        "tts","metrics","cv","searchcv","random_state","persist","mlflow","yaml_params"
    ]}
    doc_weighted_sum = 0.0
    n_funcs_total = 0

    for _, flags in results.items():
        for k in agg:
            agg[k] = agg[k] or bool(flags.get(k, False))
        if flags.get("n_funcs", 0) > 0:
            doc_weighted_sum += flags["doc_ratio"] * flags["n_funcs"]
            n_funcs_total += flags["n_funcs"]

    agg["doc_ratio"] = (doc_weighted_sum / n_funcs_total) if n_funcs_total else 0.0
    agg["n_funcs"] = n_funcs_total
    return agg


# -------- reporte --------
def main():
    print_header(
        f"3) Mejores prácticas del pipeline con Scikit-Learn — Chequeo estático "
        f"(módulo {MODULE_NAME}, raíz {ROOT})"
    )

    results = scan_repo()
    agg = aggregate(results)

    # Reglas mínimas (requerido)
    print(pad_to_column("Pipeline (sklearn.pipeline.Pipeline/make_pipeline)", 70), end="")
    print(tag(agg["pipeline"], REQUIRED))

    print(pad_to_column("Preprocesamiento con ColumnTransformer", 70), end="")
    print(tag(agg["column_transformer"], REQUIRED))

    print(pad_to_column("Transformadores: SimpleImputer", 70), end="")
    print(tag(agg["has_simple_imputer"], REQUIRED))

    print(pad_to_column("Transformadores: StandardScaler", 70), end="")
    print(tag(agg["has_standard_scaler"], REQUIRED))

    print(pad_to_column("Transformadores: OneHotEncoder", 70), end="")
    print(tag(agg["has_onehot"], REQUIRED))

    print(pad_to_column("train_test_split", 70), end="")
    print(tag(agg["tts"], REQUIRED))

    print(pad_to_column("Métricas de sklearn.metrics (evaluación)", 70), end="")
    print(tag(agg["metrics"], REQUIRED))

    # Recomendadas
    print(pad_to_column("Validación: cross_val_score", 70), end="")
    print(tag(agg["cv"], RECOMMENDED))

    print(pad_to_column("Tuning: GridSearchCV/RandomizedSearchCV", 70), end="")
    print(tag(agg["searchcv"], RECOMMENDED))

    print(pad_to_column("Reproducibilidad: uso de random_state", 70), end="")
    print(tag(agg["random_state"], REQUIRED))

    print(pad_to_column("Persistencia/registro del modelo (joblib o MLflow)", 70), end="")
    print(tag(agg["persist"] or agg["mlflow"], RECOMMENDED))

    print(pad_to_column("Lectura de configuración (params.yaml)", 70), end="")
    print(tag(agg["yaml_params"], RECOMMENDED))

    # Documentación (recomendado)
    doc_ok = agg["doc_ratio"] >= 0.5
    print(pad_to_column("Docstrings en funciones de pipeline (>=50%)", 70), end="")
    print(f"{tag(doc_ok, RECOMMENDED)}   ratio={agg['doc_ratio']:.2f} (sobre {int(agg['n_funcs'])} funciones)")

    # Leyenda
    print("\nLeyenda de estados:")
    print("  [PRESENTE]           Regla satisfecha")
    print("  [FALTA-REQUERIDO]    Debe corregirse para cumplir mejores prácticas mínimas")
    print("  [FALTA-RECOMENDADO]  Muy aconsejable para robustez y mantenibilidad")
    print("  [FALTA-OPCIONAL]     Según contexto")

    # Sugerencias
    print("\nSugerencias:")
    if not agg["pipeline"]:
        print(" - Crea un sklearn.pipeline.Pipeline o make_pipeline que encadene preprocesamiento + modelo.")
    if not agg["column_transformer"]:
        print(" - Usa sklearn.compose.ColumnTransformer para separar numéricas/categóricas.")
    if not agg["has_simple_imputer"]:
        print(" - Añade SimpleImputer para completar faltantes (num y/o cat).")
    if not agg["has_standard_scaler"]:
        print(" - Añade StandardScaler para variables numéricas (si aplica).")
    if not agg["has_onehot"]:
        print(" - Añade OneHotEncoder(handle_unknown='ignore', sparse_output=False) para categóricas.")
    if not agg["tts"]:
        print(" - Separa entrenamiento/prueba con train_test_split (estratifica si es clasificación).")
    if not agg["metrics"]:
        print(" - Calcula y registra métricas (accuracy/precision/recall/F1/ROC-AUC según el caso).")
    if not agg["cv"]:
        print(" - Añade cross_val_score para estimar desempeño promedio/varianza.")
    if not agg["searchcv"]:
        print(" - Aplica GridSearchCV/RandomizedSearchCV con cv>=3 para ajustar hiperparámetros.")
    if not agg["random_state"]:
        print(" - Fija random_state en train_test_split/estimadores/RandomizedSearchCV para reproducibilidad.")
    if not (agg["persist"] or agg["mlflow"]):
        print(" - Persiste o registra el modelo (joblib.dump) o usa MLflow (autolog/log_model).")
    if not agg["yaml_params"]:
        print(" - Centraliza hiperparámetros/rutas en params.yaml y cárgalos con yaml.safe_load.")
    if not doc_ok:
        print(" - Añade docstrings a funciones que construyen el pipeline/entrenan/evalúan.")


if __name__ == "__main__":
    main()
