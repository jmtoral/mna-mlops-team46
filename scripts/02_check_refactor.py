#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validador de refactorización y modularización del código (estático).

Ajustes:
- Considera también german_credit_ml/modeling/train.py y evaluate.py
  como válidos para las responsabilidades de entrenamiento y evaluación.
"""

import os
import ast
import glob
import argparse
from typing import List, Tuple, Optional, Dict
import sys
import io

# Forzar UTF-8 en Windows
if sys.stdout.encoding is None or "cp125" in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding is None or "cp125" in sys.stderr.encoding.lower():
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="Validador de refactorización/modularización")
    parser.add_argument(
        "--project-root",
        default=os.getenv("PROJECT_ROOT", r"C:\dev\mna-mlops-team46"),
        help="Ruta absoluta del proyecto (raíz del repo)",
    )
    parser.add_argument(
        "--module-name",
        default=os.getenv("MODULE_NAME", "german_credit_ml"),
        help="Nombre del paquete Python principal (por ejemplo german_credit_ml)",
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

# --- Patrones flexibles de módulos clave ---
MODULE_PATTERNS = {
    "preprocess.py": [
        "src/**/preprocess.py",
        "src/**/data_preprocess.py",
        f"{MODULE_NAME}/**/preprocess.py",
        f"{MODULE_NAME}/**/data_preprocess.py",
    ],
    "train.py": [
        "src/**/train.py",
        "src/**/fit.py",
        "src/**/model_train.py",
        f"{MODULE_NAME}/**/train.py",
        f"{MODULE_NAME}/**/fit.py",
        f"{MODULE_NAME}/**/model_train.py",
        f"{MODULE_NAME}/modeling/train.py",
    ],
    "evaluate.py": [
        "src/**/evaluate.py",
        "src/**/eval.py",
        "src/**/model_evaluate.py",
        f"{MODULE_NAME}/**/evaluate.py",
        f"{MODULE_NAME}/**/eval.py",
        f"{MODULE_NAME}/**/model_evaluate.py",
        f"{MODULE_NAME}/modeling/evaluate.py",
    ],
}

EXPECTED_FUNCS = {
    "preprocess.py": [["load_", "read_", "ingest_", "get_"], ["preprocess", "transform", "clean"]],
    "train.py":      [["train", "fit"], ["save_", "log_", "register_", "mlflow", "dvc"]],
    "evaluate.py":   [["evaluate", "score", "metrics", "report"], []],
}

SUGGESTED_CLASSES = ["ModelTrainer", "DataPipeline"]

MIN_DOCSTRING_RATIO = 0.5
MIN_ANNOTATED_FUNCS_RATIO = 0.3


def glob_any(patterns: List[str]) -> List[str]:
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))
    return sorted(set(paths))


def load_ast(py_file: str) -> Optional[ast.AST]:
    try:
        with open(py_file, "r", encoding="utf-8") as f:
            return ast.parse(f.read(), filename=py_file)
    except Exception:
        return None


def list_functions(tree: ast.AST) -> List[ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def list_classes(tree: ast.AST) -> List[ast.ClassDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def has_main_guard(tree: ast.AST) -> bool:
    for n in ast.walk(tree):
        if isinstance(n, ast.If):
            cond = n.test
            if isinstance(cond, ast.Compare):
                left = cond.left
                comps = cond.comparators
                if (
                    isinstance(left, ast.Name)
                    and left.id == "__name__"
                    and len(comps) == 1
                    and isinstance(comps[0], ast.Constant)
                    and comps[0].value == "__main__"
                ):
                    return True
    return False


def fn_has_docstring(fn: ast.FunctionDef) -> bool:
    return ast.get_docstring(fn) is not None


def fn_is_annotated(fn: ast.FunctionDef) -> bool:
    ann_args = sum([1 for a in fn.args.args if a.annotation is not None])
    ann_ret = 1 if fn.returns is not None else 0
    return (ann_args + ann_ret) > 0


def name_matches_any(name: str, prefixes: List[str]) -> bool:
    return any(name.startswith(p) for p in prefixes)


def summarize_funcs(funcs: List[ast.FunctionDef]) -> Tuple[float, float, List[str]]:
    if not funcs:
        return 0.0, 0.0, []
    doc_ratio = sum(fn_has_docstring(f) for f in funcs) / len(funcs)
    ann_ratio = sum(fn_is_annotated(f) for f in funcs) / len(funcs)
    public = [f.name for f in funcs if not f.name.startswith("_")]
    return doc_ratio, ann_ratio, public


def check_modules_presence() -> Dict[str, List[str]]:
    found = {}
    for logical, patterns in MODULE_PATTERNS.items():
        paths = glob_any(patterns)
        found[logical] = paths
    return found


def check_functions_responsibility(py_file: str, logical_name: str) -> Tuple[bool, str]:
    tree = load_ast(py_file)
    if not tree:
        return False, "No se pudo parsear AST."
    funcs = list_functions(tree)
    _, _, public = summarize_funcs(funcs)
    if not public:
        return False, "No hay funciones públicas."

    rules = EXPECTED_FUNCS.get(logical_name, [])
    if not rules:
        return True, "OK"

    for group in rules:
        if not group:
            continue
        if not any(name_matches_any(fn_name, group) for fn_name in public):
            return False, f"No se detectó función que cumpla prefijos esperados: {group}"
    return True, "OK"


def check_oop_suggested(py_file: str) -> Tuple[bool, List[str]]:
    tree = load_ast(py_file)
    if not tree:
        return False, []
    classes = list_classes(tree)
    names = [c.name for c in classes]
    hits = [c for c in names if c in SUGGESTED_CLASSES]
    return (len(hits) > 0, hits)


def check_quality(py_files: List[str]) -> Dict[str, float]:
    all_funcs: List[ast.FunctionDef] = []
    for f in py_files:
        t = load_ast(f)
        if not t:
            continue
        all_funcs.extend(list_functions(t))
    if not all_funcs:
        return {"doc_ratio": 0.0, "ann_ratio": 0.0, "n_funcs": 0}
    doc_ratio, ann_ratio, _ = summarize_funcs(all_funcs)
    return {"doc_ratio": doc_ratio, "ann_ratio": ann_ratio, "n_funcs": len(all_funcs)}


def check_tests_presence() -> Tuple[bool, List[str]]:
    tests = sorted(glob.glob("tests/test_*.py"))
    return (len(tests) > 0, tests)


def check_main_guards(py_files: List[str]) -> List[str]:
    flagged = []
    for f in py_files:
        base = os.path.basename(f)
        if base in ("train.py", "evaluate.py"):
            t = load_ast(f)
            if t and not has_main_guard(t):
                flagged.append(f)
    return flagged


def state_tag(present: bool, importance: str) -> str:
    if present:
        return "[PRESENTE]"
    if importance == REQUIRED:
        return "[FALTA-REQUERIDO]"
    if importance == RECOMMENDED:
        return "[FALTA-RECOMENDADO]"
    return "[FALTA-OPCIONAL]"


def pad_to_column(text: str, col: int) -> str:
    return text if len(text) >= col else text + " " * (col - len(text))


def print_header(title: str):
    print("\n" + title)
    print("-" * len(title))


def main():
    print_header("2) Refactorización y modularización del código — Chequeo estático")

    modules = check_modules_presence()

    print(f"{MODULE_NAME}/ (visión lógica de pipeline ML)")
    print("├─ preprocess.py".ljust(50), end="")
    present_pre = len(modules["preprocess.py"]) > 0
    print(state_tag(present_pre, REQUIRED))

    print("├─ train.py (incluye modeling/train.py)".ljust(50), end="")
    present_train = len(modules["train.py"]) > 0
    print(state_tag(present_train, REQUIRED))

    print("└─ evaluate.py (incluye modeling/evaluate.py)".ljust(50), end="")
    present_eval = len(modules["evaluate.py"]) > 0
    print(state_tag(present_eval, REQUIRED))

    print_header("Funciones por responsabilidad (mínimos por módulo)")
    for logical, paths in modules.items():
        target = paths[0] if paths else None
        line = f"{logical}".ljust(30)
        if not target:
            print(f"{line}{state_tag(False, REQUIRED)}   No encontrado.")
            continue
        ok, msg = check_functions_responsibility(target, logical)
        print(f"{line}{state_tag(ok, REQUIRED)}   {os.path.relpath(target)} -> {msg}")

    print_header("POO sugerida (clases reutilizables)")
    oop_hits = []
    for _, paths in modules.items():
        for p in paths:
            _, hits = check_oop_suggested(p)
            for h in hits:
                oop_hits.append((p, h))

    if oop_hits:
        for p, h in oop_hits:
            print(f"{('Clase ' + h):30}{state_tag(True, RECOMMENDED)}   {os.path.relpath(p)}")
    else:
        print(f"{'Clases sugeridas':30}{state_tag(False, RECOMMENDED)}   No se detectaron {SUGGESTED_CLASSES}")

    print_header("Pruebas unitarias básicas (pytest)")
    has_tests, test_files = check_tests_presence()
    print(
        f"{'tests/test_*.py':30}"
        f"{state_tag(has_tests, REQUIRED)}   "
        f"{', '.join(map(os.path.relpath, test_files)) if test_files else 'No hay tests'}"
    )

    print_header("Buenas prácticas (docstrings, type hints, main guard)")
    py_files = sorted(glob.glob(f"{MODULE_NAME}/**/*.py", recursive=True)) + \
               sorted(glob.glob("src/**/*.py", recursive=True))
    q = check_quality(py_files)

    doc_ok = q["doc_ratio"] >= MIN_DOCSTRING_RATIO
    ann_ok = q["ann_ratio"] >= MIN_ANNOTATED_FUNCS_RATIO

    print(
        f"{'Docstrings funciones (>=50%)':30}"
        f"{state_tag(doc_ok, RECOMMENDED)}   "
        f"ratio={q['doc_ratio']:.2f} ({q['n_funcs']} funcs)"
    )
    print(
        f"{'Type hints funciones (>=30%)':30}"
        f"{state_tag(ann_ok, RECOMMENDED)}   "
        f"ratio={q['ann_ratio']:.2f}"
    )

    need_main_guard = check_main_guards(py_files)
    if need_main_guard:
        for f in need_main_guard:
            print(
                f"{'__main__ en scripts':30}"
                f"{state_tag(False, RECOMMENDED)}   "
                f"Falta guardia en {os.path.relpath(f)}"
            )
    else:
        print(
            f"{'__main__ en scripts':30}"
            f"{state_tag(True, RECOMMENDED)}   "
            f"OK"
        )

    print("\nLeyenda de estados:")
    print("  [PRESENTE]           Regla satisfecha")
    print("  [FALTA-REQUERIDO]    Debe corregirse para cumplir la refactorización mínima")
    print("  [FALTA-RECOMENDADO]  Muy aconsejable para mantenibilidad/testabilidad")
    print("  [FALTA-OPCIONAL]     Depende del contexto del proyecto")

    print("\nSugerencias:")
    if not present_pre:
        print(f" - Crea {MODULE_NAME}/preprocess.py (carga/limpieza/features).")
    if not present_train:
        print(f" - Crea {MODULE_NAME}/modeling/train.py (entrenamiento; logging de métricas y artefactos).")
    if not present_eval:
        print(f" - Crea {MODULE_NAME}/modeling/evaluate.py (evaluación y reporte de métricas).")
    if not has_tests:
        print(" - Agrega tests en tests/test_*.py (al menos lectura de datos, preprocess y métrica principal).")
    if not doc_ok:
        print(" - Añade docstrings breves a funciones públicas (qué hace, entradas, salidas).")
    if not ann_ok:
        print(" - Añade anotaciones de tipos en funciones clave (mejora lint/CI).")
    if need_main_guard:
        print(" - Añade guardia if __name__ == '__main__': en train.py/evaluate.py si se ejecutan directo.")


if __name__ == "__main__":
    main()
