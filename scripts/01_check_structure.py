#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificador de estructura Cookiecutter Data Science v2 con estados:
[PRESENTE], [FALTA-REQUERIDO], [FALTA-RECOMENDADO], [FALTA-OPCIONAL].

Ajustes:
- Acepta german_credit_ml/modeling/ como equivalente a german_credit_ml/models/
  (equipo usa 'modeling' para código de entrenamiento/evaluación).
"""

import os
import glob
import argparse
from typing import List, Optional, Tuple
import sys
import io

# Forzar stdout/stderr a UTF-8 en Windows
if sys.stdout.encoding is None or "cp125" in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding is None or "cp125" in sys.stderr.encoding.lower():
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="Validador de estructura Cookiecutter Data Science v2")
    parser.add_argument(
        "--project-root",
        default=os.getenv("PROJECT_ROOT", r"C:\dev\mna-mlops-team46"),
        help="Ruta absoluta del proyecto (raíz del repo)",
    )
    parser.add_argument(
        "--module-name",
        default=os.getenv("MODULE_NAME", "german_credit_ml"),
        help="Nombre del paquete Python al nivel raíz (Cookiecutter v2)",
    )
    return parser.parse_args()

ARGS = parse_args()

ROOT = os.path.abspath(os.path.expanduser(ARGS.project_root))
if not os.path.isdir(ROOT):
    raise SystemExit(f"[ERROR] Project root no existe: {ROOT}")
os.chdir(ROOT)

ROOT_NAME = os.path.basename(ROOT.rstrip(os.sep)) or "project-name"
MODULE_NAME = ARGS.module_name.strip() or "<package_name>"

REQUIRED = "requerido"
RECOMMENDED = "recomendado"
OPTIONAL = "opcional"


def exists(path: str) -> bool:
    return os.path.exists(os.path.join(ROOT, path))


def any_exists(paths):
    """True si al menos una de las rutas existe."""
    for p in paths:
        if os.path.exists(os.path.join(ROOT, p)):
            return True
    return False


def find_top_level_module() -> Optional[str]:
    candidate = os.path.join(MODULE_NAME, "__init__.py")
    if os.path.isfile(candidate):
        return MODULE_NAME
    for d in sorted([d for d in glob.glob("*") if os.path.isdir(d)]):
        if os.path.isfile(os.path.join(d, "__init__.py")):
            return os.path.basename(d)
    return None


def get_package_name() -> str:
    auto = find_top_level_module()
    if auto:
        return auto
    return MODULE_NAME or "<package_name>"


def pyproject_or_requirements_setup() -> Tuple[bool, str]:
    pyproject = exists("pyproject.toml")
    req_setup = exists("requirements.txt") and exists("setup.cfg")
    label = "pyproject.toml     (o requirements.txt + setup.cfg)"
    return (pyproject or req_setup, label)


class Node:
    def __init__(
        self,
        label: str,
        path: Optional[str] = None,
        children: Optional[List["Node"]] = None,
        importance: str = REQUIRED,
        optional_note: Optional[str] = None,
        right_note: Optional[str] = None,
        exists_override: Optional[bool] = None,
    ):
        self.label = label
        self.path = path
        self.children = children or []
        self.importance = importance
        self.optional_note = optional_note
        self.right_note = right_note
        self.exists_override = exists_override

    def exists(self) -> bool:
        if self.exists_override is not None:
            return self.exists_override
        if self.path is None:
            return True
        return exists(self.path)

    def state_tag(self) -> str:
        if self.path is None and self.exists_override is None:
            return ""
        if self.exists():
            return "[PRESENTE]"
        if self.importance == REQUIRED:
            return "[FALTA-REQUERIDO]"
        if self.importance == RECOMMENDED:
            return "[FALTA-RECOMENDADO]"
        return "[FALTA-OPCIONAL]"


def build_template_tree() -> Node:
    pkg = get_package_name()
    ok_req, label_req = pyproject_or_requirements_setup()

    # carpeta de modelado interna:
    # aceptar pkg/models o pkg/modeling
    models_like_exists = any_exists([f"{pkg}/models", f"{pkg}/modeling"])

    return Node(f"{ROOT_NAME}/", path=None, importance=REQUIRED, children=[
        Node("README.md", "README.md", importance=REQUIRED),
        Node("LICENSE", "LICENSE", importance=RECOMMENDED, optional_note="(opcional pero recomendado)"),
        Node(".gitignore", ".gitignore", importance=REQUIRED),
        Node(".env.example", ".env.example", importance=RECOMMENDED, optional_note="(variables, sin secretos)"),

        Node(label_req, None, importance=REQUIRED, exists_override=ok_req),

        Node("Makefile", "Makefile", importance=RECOMMENDED, optional_note="(tareas comunes)"),

        Node("data/", "data", importance=REQUIRED, children=[
            Node("raw/", "data/raw", importance=REQUIRED, optional_note="(solo lectura)"),
            Node("interim/", "data/interim", importance=REQUIRED, optional_note="(steps intermedios)"),
            Node("processed/", "data/processed", importance=REQUIRED, optional_note="(dataset limpio para modeling)"),
        ]),

        Node("models/", "models", importance=RECOMMENDED, optional_note="(artefactos entrenados)"),
        Node("notebooks/", "notebooks", importance=RECOMMENDED,
             optional_note="(numerados: 0x-autor-propósito)"),
        Node("reports/", "reports", importance=RECOMMENDED, optional_note="(figuras, tablas)"),

        Node(f"{pkg}/", f"{pkg}", importance=REQUIRED,
             optional_note="(paquete Python principal del proyecto)", children=[
            Node("__init__.py", f"{pkg}/__init__.py", importance=REQUIRED),
            Node("data/", f"{pkg}/data", importance=RECOMMENDED, optional_note="(IO de datos)"),
            Node("features/", f"{pkg}/features", importance=RECOMMENDED, optional_note="(featurización)"),
            Node("models/ (o modeling/)", f"{pkg}/models", importance=RECOMMENDED,
                 optional_note="(train, predict, evaluate)",
                 exists_override=models_like_exists),
            Node("utils/", f"{pkg}/utils", importance=RECOMMENDED,
                 optional_note="(helpers/utilidades compartidas)"),
        ]),

        Node("tests/", "tests", importance=RECOMMENDED, optional_note="(pytest)"),

        Node("dvc.yaml", "dvc.yaml", importance=OPTIONAL, optional_note="(si usas DVC)"),
        Node("params.yaml", "params.yaml", importance=RECOMMENDED, optional_note="(hiperparámetros/config)"),
        Node("mlruns/", "mlruns", importance=OPTIONAL,
             optional_note="(si MLflow local)", right_note="← puede estar ignorado"),
        Node(".pre-commit-config.yaml", ".pre-commit-config.yaml", importance=RECOMMENDED,
             optional_note="(formato/linters)"),
    ])


def pad_to_column(text: str, col: int) -> str:
    return text if len(text) >= col else text + " " * (col - len(text))


def render_tree(node: Node, prefix: str = "", is_last: bool = True, right_note_col: int = 70):
    connector = "└─ " if is_last else "├─ "
    line_prefix = prefix + connector if prefix else ""
    line = f"{line_prefix}{node.label}"

    if node.optional_note:
        line = pad_to_column(line, 40) + node.optional_note

    state = node.state_tag()
    if state:
        line = pad_to_column(line, 70) + state

    if node.right_note:
        line = pad_to_column(line, right_note_col) + node.right_note

    print(line)

    child_prefix = prefix + ("   " if is_last else "│  ")
    for i, child in enumerate(node.children):
        render_tree(
            child,
            prefix=child_prefix,
            is_last=(i == len(node.children) - 1),
            right_note_col=right_note_col,
        )


def print_legend():
    print("\nLeyenda de estados:")
    print("  [PRESENTE]           Elemento encontrado en el repo")
    print("  [FALTA-REQUERIDO]    Elemento requerido ausente (debe corregirse)")
    print("  [FALTA-RECOMENDADO]  Elemento recomendado ausente (muy aconsejable)")
    print("  [FALTA-OPCIONAL]     Elemento opcional ausente (según necesidades)")


def print_header(title: str):
    print("\n" + title)
    print("-" * len(title))


def main():
    print_header("1) Estructuración de Proyectos con Cookiecutter Data Science v2")
    tree = build_template_tree()
    render_tree(tree, prefix="", is_last=True)
    print_legend()


if __name__ == "__main__":
    main()
