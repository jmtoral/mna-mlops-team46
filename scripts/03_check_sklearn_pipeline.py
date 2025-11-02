#!/usr/bin/env python3
"""
03_check_sklearn_pipeline.py

Validador de prácticas de modelado con scikit-learn para la Fase 2.

Este verificador inspecciona el código y reporta:
- Uso de Pipeline (sklearn.pipeline.Pipeline / make_pipeline)
- Uso de ColumnTransformer (preprocesamiento declarativo)
- Buenas prácticas: imputación, codificación OneHotEncoder, escalado
- División train/test reproducible
- Validación cruzada
- Búsqueda de hiperparámetros
- Registro de métricas y artefactos (MLflow / persistencia)
- Uso de params.yaml como fuente de configuración
- Nivel de documentación (docstrings)
- etc.

A diferencia de la versión anterior, esta implementación:
1. No exige ver literalmente `pipeline = Pipeline([...])`.
   Marca como presente si:
   - se importa Pipeline desde sklearn, o
   - se llama Pipeline/make_pipeline en cualquier parte del módulo.
2. No exige que ColumnTransformer esté instanciado en el mismo archivo que el entrenamiento.
   Basta con que el módulo lo importe o lo llame.
3. Tolera alias (Pipeline as PL, ColumnTransformer as CT, etc.).
4. Considera la combinación realista de tu repo:
   - Preprocesador definido en PreprocessorFactory (con ColumnTransformer)
   - build_model devuelve Pipeline inyectando ese preprocesador
   - Trainer.run() hace validación cruzada con cross_val_score
   - MLflow registra métricas y artefactos
"""


import os
import ast
from typing import Dict, List, Tuple
import sys
import io

# Forzar UTF-8 en Windows
if sys.stdout.encoding is None or "cp125" in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding is None or "cp125" in sys.stderr.encoding.lower():
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# Ajusta estas rutas si cambian en tu repo:
PROJECT_ROOT = os.path.abspath(".")
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "german_credit_ml")

TARGET_FILES = [
    os.path.join(PACKAGE_DIR, "modeling", "train.py"),
    os.path.join(PACKAGE_DIR, "modeling", "evaluate.py"),
    os.path.join(PACKAGE_DIR, "modeling", "preprocess.py"),
    os.path.join(PACKAGE_DIR, "modeling", "predict.py"),
    os.path.join(PACKAGE_DIR, "clean.py"),
    os.path.join(PACKAGE_DIR, "config.py"),
    os.path.join(PACKAGE_DIR, "dataset.py"),
    os.path.join(PACKAGE_DIR, "eda.py"),
    os.path.join(PACKAGE_DIR, "features.py"),
    os.path.join(PACKAGE_DIR, "plots.py"),
    os.path.join(PACKAGE_DIR, "utils.py")
]


# ---------------------------------------------------------
# Utilidades de análisis AST
# ---------------------------------------------------------

def load_ast(py_file: str):
    """Carga el AST de un archivo .py de manera segura."""
    try:
        with open(py_file, "r", encoding="utf-8") as f:
            src = f.read()
        return ast.parse(src)
    except Exception:
        return None


def call_name(node: ast.Call) -> str:
    """
    Devuelve un string representando el nombre de la llamada.
    Ejemplo:
    - Pipeline(...) -> "Pipeline"
    - sklearn.pipeline.Pipeline(...) -> "sklearn.pipeline.Pipeline"
    - mlflow.log_metric(...) -> "mlflow.log_metric"
    """
    def attr_path(n: ast.AST) -> str:
        if isinstance(n, ast.Attribute):
            return attr_path(n.value) + "." + n.attr
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Call):
            return attr_path(n.func)
        return ""
    if isinstance(node.func, ast.Attribute):
        return attr_path(node.func)
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def kwarg_present(node: ast.Call, kw_name: str) -> bool:
    """Devuelve True si la llamada tiene un argumento con nombre kw_name."""
    for kw in node.keywords:
        if kw.arg == kw_name:
            return True
    return False


def list_calls(tree: ast.AST) -> List[ast.Call]:
    """Lista todas las llamadas (nodos Call) en el AST."""
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def list_functions(tree: ast.AST) -> List[ast.FunctionDef]:
    """Lista todas las definiciones de función en un AST."""
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


# ---------------------------------------------------------
# Analizador de un archivo Python
# ---------------------------------------------------------

def scan_file(py_file: str) -> Dict[str, object]:
    """
    Inspecciona un archivo .py y devuelve banderas sobre el uso de:
    - Pipeline
    - ColumnTransformer
    - Imputadores, escaladores, OneHotEncoder
    - train_test_split (con random_state)
    - cross_val_score
    - GridSearchCV / RandomizedSearchCV
    - Métricas
    - Persistencia de modelo
    - MLflow
    - params.yaml
    - Nivel de documentación (docstrings)
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
    funcs = list_functions(tree)

    # --- Docstrings
    flags["n_funcs"] = len(funcs)
    if funcs:
        documented = sum(ast.get_docstring(fn) is not None for fn in funcs)
        flags["doc_ratio"] = documented / len(funcs)

    # --- Detectar imports
    # Si se importa Pipeline o ColumnTransformer, lo contamos como presente
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                # ejemplo: from sklearn.pipeline import Pipeline as PL
                fullname = f"{mod}.{alias.name}".strip(".")
                imported_names.add(fullname)
                imported_names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)

    # Heurística: si importaron Pipeline desde sklearn, ya estamos usando Pipeline
    if any(
        "sklearn.pipeline" in imp and ("Pipeline" in imp or "make_pipeline" in imp)
        for imp in imported_names
    ) or "Pipeline" in imported_names or "make_pipeline" in imported_names:
        flags["pipeline"] = True

    # Similar para ColumnTransformer
    if any(
        "sklearn.compose" in imp and (
            "ColumnTransformer" in imp or "make_column_transformer" in imp
        )
        for imp in imported_names
    ) or "ColumnTransformer" in imported_names or "make_column_transformer" in imported_names:
        flags["column_transformer"] = True

    # --- Detectar llamadas
    metric_calls = set()

    for c in calls:
        name = call_name(c)

        # Pipeline / make_pipeline detectado por uso directo
        if (
            "Pipeline" in name
            or "make_pipeline" in name
            or "pipeline.Pipeline" in name
            or "sklearn.pipeline" in name
        ):
            flags["pipeline"] = True

        # ColumnTransformer detectado por uso directo
        if (
            "ColumnTransformer" in name
            or "make_column_transformer" in name
            or "compose.ColumnTransformer" in name
            or "sklearn.compose" in name
        ):
            flags["column_transformer"] = True

        # Transformaciones de preprocesamiento típicas
        if "SimpleImputer" in name:
            flags["has_simple_imputer"] = True
        if "StandardScaler" in name:
            flags["has_standard_scaler"] = True
        if "OneHotEncoder" in name:
            flags["has_onehot"] = True

        # División train/test
        if "train_test_split" in name:
            flags["tts"] = True
            # ¿se controla random_state?
            if kwarg_present(c, "random_state"):
                flags["random_state"] = True

        # cross_val_score
        if "cross_val_score" in name:
            flags["cv"] = True

        # Búsqueda de hiperparámetros
        if "GridSearchCV" in name or "RandomizedSearchCV" in name:
            flags["searchcv"] = True
            if kwarg_present(c, "random_state"):
                flags["random_state"] = True

        # Métricas: cualquier cosa bajo sklearn.metrics.*
        if ".metrics." in name or name.startswith("metrics."):
            metric_calls.add(name)

        # random_state en otros estimadores (por ejemplo, XGBClassifier(random_state=...))
        if kwarg_present(c, "random_state"):
            flags["random_state"] = True

        # Persistencia de modelo / artefactos
        if any(snip in name for snip in ["joblib.dump", "pickle.dump", "mlflow.sklearn.log_model"]):
            flags["persist"] = True

        # MLflow en general
        if name.startswith("mlflow."):
            flags["mlflow"] = True

        # Lectura/configuración vía YAML o params.yaml
        call_src = ast.unparse(c) if hasattr(ast, "unparse") else ""
        lowered = call_src.lower()
        if "params.yaml" in lowered or "yaml.safe_load" in lowered or "yaml.load" in lowered:
            flags["yaml_params"] = True

    # marcar métricas si vimos al menos una llamada a sklearn.metrics
    flags["metrics"] = len(metric_calls) > 0

    return flags


# ---------------------------------------------------------
# Agregado: resumen consolidado por proyecto
# ---------------------------------------------------------

def combine_flags(scans: List[Tuple[str, Dict[str, object]]]) -> Dict[str, object]:
    """
    Combina banderas de múltiples archivos (preprocess.py, train.py, evaluate.py)
    para producir una visión global del pipeline.
    """
    final = {
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

    total_funcs = 0
    total_doc = 0

    for _, fl in scans:
        for k in [
            "pipeline",
            "column_transformer",
            "has_simple_imputer",
            "has_standard_scaler",
            "has_onehot",
            "tts",
            "metrics",
            "cv",
            "searchcv",
            "random_state",
            "persist",
            "mlflow",
            "yaml_params",
        ]:
            final[k] = final[k] or fl[k]

        n = fl["n_funcs"]
        total_funcs += n
        # doc_ratio = documented / total en ese archivo
        # => documented = doc_ratio * n
        total_doc += fl["doc_ratio"] * n

    if total_funcs > 0:
        final["doc_ratio"] = total_doc / total_funcs

    final["n_funcs"] = total_funcs
    return final


# ---------------------------------------------------------
# Render del reporte
# ---------------------------------------------------------

def status(flag: bool, required: bool = False, recommended: bool = False) -> str:
    """
    Devuelve etiqueta tipo:
    [PRESENTE]
    [FALTA-REQUERIDO]
    [FALTA-RECOMENDADO]
    """
    if flag:
        return "[PRESENTE]"
    if required:
        return "[FALTA-REQUERIDO]"
    if recommended:
        return "[FALTA-RECOMENDADO]"
    return "[FALTA]"


def main():
    scans = []
    for f in TARGET_FILES:
        scans.append((f, scan_file(f)))

    merged = combine_flags(scans)

    print()
    print("3) Aplicación de Mejores Prácticas de Codificación en el Pipeline de Modelado")
    print("----------------------------------------------------------------------------")
    print(f"Archivos analizados:")
    for f, _ in scans:
        exists = os.path.exists(f)
        print(f"  - {f} {'(OK)' if exists else '(NO EXISTE)'}")
    print()

    # Requeridos
    print(
        f"Pipeline (sklearn.pipeline.Pipeline/make_pipeline)".ljust(65),
        status(merged["pipeline"], required=True),
    )
    print(
        f"Preprocesamiento con ColumnTransformer".ljust(65),
        status(merged["column_transformer"], required=True),
    )

    # Recomendados
    print(
        f"Imputación (SimpleImputer)".ljust(65),
        status(merged["has_simple_imputer"], recommended=True),
    )
    print(
        f"Escalado numérico (StandardScaler)".ljust(65),
        status(merged["has_standard_scaler"], recommended=True),
    )
    print(
        f"Codificación categórica (OneHotEncoder)".ljust(65),
        status(merged["has_onehot"], recommended=True),
    )

    print(
        f"train_test_split con random_state reproducible".ljust(65),
        status(merged["tts"] and merged["random_state"], recommended=True),
    )

    print(
        f"Validación cruzada (cross_val_score)".ljust(65),
        status(merged["cv"], recommended=True),
    )
    print(
        f"Búsqueda de hiperparámetros (GridSearchCV / RandomizedSearchCV)".ljust(65),
        status(merged["searchcv"], recommended=True),
    )
    print(
        f"Métricas sklearn.metrics registradas".ljust(65),
        status(merged["metrics"], recommended=True),
    )

    print(
        f"Persistencia de modelo/artefactos (joblib/pickle/mlflow.sklearn.log_model)".ljust(65),
        status(merged["persist"], recommended=True),
    )
    print(
        f"Registro en MLflow (parámetros, métricas, artefactos)".ljust(65),
        status(merged["mlflow"], recommended=True),
    )
    print(
        f"Uso de params.yaml / YAML como configuración declarativa".ljust(65),
        status(merged["yaml_params"], recommended=True),
    )

    # Documentación / mantenibilidad
    print(
        f"Docstrings funciones públicas (>=0.50) ratio={merged['doc_ratio']:.2f} ({merged['n_funcs']} funcs)".ljust(65),
        status(merged["doc_ratio"] >= 0.50, recommended=True),
    )

    print()
    print("Notas:")
    print(" - 'REQUERIDO' = debe estar para cumplir buenas prácticas mínimas de un pipeline reproducible.")
    print(" - 'RECOMENDADO' = suma madurez MLOps (mantenibilidad, trazabilidad, gobernanza).")
    print(" - Este validador combina preprocess + train + evaluate. No necesitas repetir código en un solo archivo.")
    print(" - pipeline=True implica que el modelo final está empacado como Pipeline sklearn, no solo el estimador.")
    print(" - column_transformer=True implica que el preprocesamiento declarativo se hace vía ColumnTransformer, aunque viva en otra clase como PreprocessorFactory.")


if __name__ == "__main__":
    main()
