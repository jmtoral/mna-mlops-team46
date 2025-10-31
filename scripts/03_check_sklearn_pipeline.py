#!/usr/bin/env python3
"""
03_check_sklearn_pipeline.py

<<<<<<< HEAD
Revisa estáticamente (AST) en el código del proyecto:
- Pipeline/ColumnTransformer presentes
- Transformadores recomendados (SimpleImputer, StandardScaler, OneHotEncoder)
- Entrenamiento/evaluación: train_test_split, métricas sklearn.metrics
- Validación y tuning: cross_val_score, GridSearchCV/RandomizedSearchCV (recomendado)
- Reproducibilidad: uso de random_state
- Persistencia/registro: joblib.dump o mlflow.sklearn.log_model / mlflow.autolog (recomendado)
- Documentación/claridad: docstrings en funciones clave, lectura de params.yaml
=======
Validador de prácticas de modelado con scikit-learn para la Fase 2.
>>>>>>> 72306f4c801cea0547580fd749a5734862eb14a9

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

import ast
<<<<<<< HEAD
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
=======
import os
from typing import Dict, List, Tuple

# Ajusta estas rutas si cambian en tu repo:
PROJECT_ROOT = os.path.abspath(".")
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "german_credit_ml")

TARGET_FILES = [
    os.path.join(PACKAGE_DIR, "preprocess.py"),
    os.path.join(PACKAGE_DIR, "modeling", "train.py"),
    os.path.join(PACKAGE_DIR, "modeling", "evaluate.py"),
]


# ---------------------------------------------------------
# Utilidades de análisis AST
# ---------------------------------------------------------
>>>>>>> 72306f4c801cea0547580fd749a5734862eb14a9


<<<<<<< HEAD
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
=======
def load_ast(py_file: str):
    """Carga el AST de un archivo .py de manera segura."""
>>>>>>> 72306f4c801cea0547580fd749a5734862eb14a9
    try:
        with open(py_file, "r", encoding="utf-8") as f:
            src = f.read()
        return ast.parse(src)
    except Exception:
        return None

<<<<<<< HEAD

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
=======

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
>>>>>>> 72306f4c801cea0547580fd749a5734862eb14a9
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
    if (
        any(
            "sklearn.pipeline" in imp and ("Pipeline" in imp or "make_pipeline" in imp)
            for imp in imported_names
        )
        or "Pipeline" in imported_names
        or "make_pipeline" in imported_names
    ):
        flags["pipeline"] = True

    # Similar para ColumnTransformer
    if (
        any(
            "sklearn.compose" in imp
            and ("ColumnTransformer" in imp or "make_column_transformer" in imp)
            for imp in imported_names
        )
        or "ColumnTransformer" in imported_names
        or "make_column_transformer" in imported_names
    ):
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

<<<<<<< HEAD
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
=======
        # Transformaciones de preprocesamiento típicas
        if "SimpleImputer" in name:
            flags["has_simple_imputer"] = True
        if "StandardScaler" in name:
            flags["has_standard_scaler"] = True
        if "OneHotEncoder" in name:
            flags["has_onehot"] = True

        # División train/test
        if "train_test_split" in name:
>>>>>>> 72306f4c801cea0547580fd749a5734862eb14a9
            flags["tts"] = True
            # ¿se controla random_state?
            if kwarg_present(c, "random_state"):
                flags["random_state"] = True

        # cross_val_score
<<<<<<< HEAD
        if (
            name.endswith("sklearn.model_selection.cross_val_score")
            or name.endswith("model_selection.cross_val_score")
            or name.endswith("cross_val_score")
        ):
            flags["cv"] = True

        # GridSearchCV / RandomizedSearchCV
        if name.endswith("GridSearchCV") or name.endswith("RandomizedSearchCV"):
=======
        if "cross_val_score" in name:
            flags["cv"] = True

        # Búsqueda de hiperparámetros
        if "GridSearchCV" in name or "RandomizedSearchCV" in name:
>>>>>>> 72306f4c801cea0547580fd749a5734862eb14a9
            flags["searchcv"] = True
            if kwarg_present(c, "random_state"):
                flags["random_state"] = True

        # Métricas: cualquier cosa bajo sklearn.metrics.*
        if ".metrics." in name or name.startswith("metrics."):
            metric_calls.add(name)

        # random_state en otros estimadores (por ejemplo, XGBClassifier(random_state=...))
        if kwarg_present(c, "random_state"):
            flags["random_state"] = True

<<<<<<< HEAD
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
=======
        # Persistencia de modelo / artefactos
        if any(
            snip in name
            for snip in ["joblib.dump", "pickle.dump", "mlflow.sklearn.log_model"]
        ):
            flags["persist"] = True

        # MLflow en general
        if name.startswith("mlflow."):
>>>>>>> 72306f4c801cea0547580fd749a5734862eb14a9
            flags["mlflow"] = True

        # Lectura/configuración vía YAML o params.yaml
        call_src = ast.unparse(c) if hasattr(ast, "unparse") else ""
        lowered = call_src.lower()
        if (
            "params.yaml" in lowered
            or "yaml.safe_load" in lowered
            or "yaml.load" in lowered
        ):
            flags["yaml_params"] = True

    # marcar métricas si vimos al menos una llamada a sklearn.metrics
    flags["metrics"] = len(metric_calls) > 0

    return flags

<<<<<<< HEAD

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
=======

# ---------------------------------------------------------
# Agregado: resumen consolidado por proyecto
# ---------------------------------------------------------

>>>>>>> 72306f4c801cea0547580fd749a5734862eb14a9

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


<<<<<<< HEAD

# -------- reporte --------
=======
>>>>>>> 72306f4c801cea0547580fd749a5734862eb14a9
def main():
    scans = []
    for f in TARGET_FILES:
        scans.append((f, scan_file(f)))

    merged = combine_flags(scans)

    print()
    print(
        "3) Aplicación de Mejores Prácticas de Codificación en el Pipeline de Modelado"
    )
    print(
        "----------------------------------------------------------------------------"
    )
    print("Archivos analizados:")
    for f, _ in scans:
        exists = os.path.exists(f)
        print(f"  - {f} {'(OK)' if exists else '(NO EXISTE)'}")
    print()

    # Requeridos
    print(
        "Pipeline (sklearn.pipeline.Pipeline/make_pipeline)".ljust(65),
        status(merged["pipeline"], required=True),
    )
    print(
        "Preprocesamiento con ColumnTransformer".ljust(65),
        status(merged["column_transformer"], required=True),
    )

    # Recomendados
    print(
        "Imputación (SimpleImputer)".ljust(65),
        status(merged["has_simple_imputer"], recommended=True),
    )
    print(
        "Escalado numérico (StandardScaler)".ljust(65),
        status(merged["has_standard_scaler"], recommended=True),
    )
    print(
        "Codificación categórica (OneHotEncoder)".ljust(65),
        status(merged["has_onehot"], recommended=True),
    )

    print(
        "train_test_split con random_state reproducible".ljust(65),
        status(merged["tts"] and merged["random_state"], recommended=True),
    )

    print(
        "Validación cruzada (cross_val_score)".ljust(65),
        status(merged["cv"], recommended=True),
    )
    print(
        "Búsqueda de hiperparámetros (GridSearchCV / RandomizedSearchCV)".ljust(65),
        status(merged["searchcv"], recommended=True),
    )
    print(
        "Métricas sklearn.metrics registradas".ljust(65),
        status(merged["metrics"], recommended=True),
    )

    print(
        "Persistencia de modelo/artefactos (joblib/pickle/mlflow.sklearn.log_model)".ljust(
            65
        ),
        status(merged["persist"], recommended=True),
    )
    print(
        "Registro en MLflow (parámetros, métricas, artefactos)".ljust(65),
        status(merged["mlflow"], recommended=True),
    )
    print(
        "Uso de params.yaml / YAML como configuración declarativa".ljust(65),
        status(merged["yaml_params"], recommended=True),
    )

    # Documentación / mantenibilidad
    print(
        f"Docstrings funciones públicas (>=0.50) ratio={merged['doc_ratio']:.2f} ({merged['n_funcs']} funcs)".ljust(
            65
        ),
        status(merged["doc_ratio"] >= 0.50, recommended=True),
    )

    print()
    print("Notas:")
    print(
        " - 'REQUERIDO' = debe estar para cumplir buenas prácticas mínimas de un pipeline reproducible."
    )
    print(
        " - 'RECOMENDADO' = suma madurez MLOps (mantenibilidad, trazabilidad, gobernanza)."
    )
    print(
        " - Este validador combina preprocess + train + evaluate. No necesitas repetir código en un solo archivo."
    )
    print(
        " - pipeline=True implica que el modelo final está empacado como Pipeline sklearn, no solo el estimador."
    )
    print(
        " - column_transformer=True implica que el preprocesamiento declarativo se hace vía ColumnTransformer, aunque viva en otra clase como PreprocessorFactory."
    )

<<<<<<< HEAD
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
=======
>>>>>>> 72306f4c801cea0547580fd749a5734862eb14a9


if __name__ == "__main__":
    main()
