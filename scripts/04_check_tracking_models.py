#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validador de:
4) Seguimiento de Experimentos, Visualización de Resultados y Gestión de Modelos

Ajustes:
- "Artefactos de modelo versionados" se considera satisfecho si:
  a) existen archivos serializados en ./models/, O
  b) existe al menos un run en MLflow con artifacts/model/MLmodel.
- Escaneo de código soporta tanto src/** como <module_name>/**.
"""

import os
import re
import glob
import yaml
import argparse
from typing import Dict, List, Tuple, Optional
import sys
import io

# Forzar UTF-8 en Windows
if sys.stdout.encoding is None or "cp125" in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding is None or "cp125" in sys.stderr.encoding.lower():
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validador de tracking de experimentos, versionado y registro de modelos (MLflow / DVC)"
    )
    parser.add_argument(
        "--project-root",
        default=os.getenv("PROJECT_ROOT", r"C:\dev\mna-mlops-team46"),
        help="Ruta absoluta del proyecto (raíz del repo)",
    )
    parser.add_argument(
        "--module-name",
        default=os.getenv("MODULE_NAME", "german_credit_ml"),
        help="Nombre del paquete Python principal del proyecto (por ejemplo german_credit_ml)",
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


def tag(present: bool, importance: str) -> str:
    if present:
        return "[PRESENTE]"
    if importance == REQUIRED:
        return "[FALTA-REQUERIDO]"
    if importance == RECOMMENDED:
        return "[FALTA-RECOMENDADO]"
    return "[FALTA-OPCIONAL]"


def pad(text: str, col: int) -> str:
    return text if len(text) >= col else text + " " * (col - len(text))


def print_header(title: str):
    print("\n" + title)
    print("-" * len(title))


def candidate_code_paths(module_name: str) -> List[str]:
    patterns = []
    patterns.append("src/**/*.py")
    patterns.append(f"{module_name}/**/*.py")
    patterns.append("*.py")
    return patterns


def grep_code(patterns_regex: List[str], module_name: str) -> set:
    found = set()

    py_files = []
    for pat in candidate_code_paths(module_name):
        py_files.extend(glob.glob(pat, recursive=True))
    py_files = sorted(set(py_files))

    nb_files = sorted(glob.glob("notebooks/**/*.ipynb", recursive=True))

    for f in py_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                txt = fh.read()
            for p in patterns_regex:
                if re.search(p, txt):
                    found.add(p)
        except Exception:
            pass

    for f in nb_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                txt = fh.read()
            for p in patterns_regex:
                if re.search(p, txt):
                    found.add(p)
        except Exception:
            pass

    return found


def has_path(p: str) -> bool:
    return os.path.exists(p)


def read_file_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def dvc_config_info() -> Tuple[Optional[str], List[str], List[str]]:
    cfg_path = os.path.join(".dvc", "config")
    txt = read_file_text(cfg_path)
    remotes = re.findall(r'\[remote\s+"([^"]+)"\]', txt or "")
    urls = re.findall(r'url\s*=\s*(.+)', txt or "")
    return (cfg_path if txt else None, remotes, urls)


def find_mlruns_dir() -> Optional[str]:
    env_dir = os.getenv("MLRUNS_DIR")
    if env_dir and os.path.isdir(env_dir):
        return os.path.normpath(env_dir)

    if os.path.isdir("mlruns"):
        return os.path.abspath("mlruns")

    skip = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache"}
    for d in glob.glob("**/mlruns", recursive=True):
        parts = set(d.split(os.sep))
        if parts.isdisjoint(skip) and os.path.isdir(d):
            return os.path.abspath(d)

    return None


def scan_mlruns() -> Dict[str, int]:
    base = find_mlruns_dir()
    info = {
        "exists": bool(base),
        "n_experiments": 0,
        "n_runs": 0,
        "runs_with_params": 0,
        "runs_with_metrics": 0,
        "runs_with_artifacts": 0,
        "runs_with_model_artifact": 0,
        "runs_with_visuals": 0,
    }
    if not base:
        return info

    exps = [
        d for d in glob.glob(os.path.join(base, "*"))
        if os.path.isdir(d) and os.path.basename(d) not in {"models", ".trash"}
    ]
    info["n_experiments"] = len(exps)

    for exp in exps:
        run_dirs = [d for d in glob.glob(os.path.join(exp, "*")) if os.path.isdir(d)]
        for run in run_dirs:
            info["n_runs"] += 1
            pdir = os.path.join(run, "params")
            mdir = os.path.join(run, "metrics")
            adir = os.path.join(run, "artifacts")

            if os.path.isdir(pdir) and glob.glob(os.path.join(pdir, "*")):
                info["runs_with_params"] += 1
            if os.path.isdir(mdir) and glob.glob(os.path.join(mdir, "*")):
                info["runs_with_metrics"] += 1
            if os.path.isdir(adir) and glob.glob(os.path.join(adir, "**/*"), recursive=True):
                info["runs_with_artifacts"] += 1

            if os.path.isfile(os.path.join(adir, "model", "MLmodel")):
                info["runs_with_model_artifact"] += 1

            visuals = []
            for ext in ("*.png", "*.html", "*.pdf"):
                visuals.extend(glob.glob(os.path.join(adir, "**", ext), recursive=True))
            if visuals:
                info["runs_with_visuals"] += 1

    return info


def scan_models_dir() -> List[str]:
    found: List[str] = []
    for ext in ("*.joblib", "*.pkl", "*.pickle", "*.onnx", "*.pt", "*.pytorch", "*.pb", "*.h5"):
        found.extend(glob.glob(os.path.join("models", "**", ext), recursive=True))
    return sorted(set(found))


def scan_dvc_files() -> Tuple[List[str], bool]:
    dvc_files = sorted(glob.glob("**/*.dvc", recursive=True))
    return dvc_files, has_path("dvc.lock")


def read_params_yaml_keys() -> List[str]:
    if not has_path("params.yaml"):
        return []
    try:
        with open("params.yaml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return list(data.keys())
        return []
    except Exception:
        return []


def readme_has_sections() -> Dict[str, bool]:
    txt = read_file_text("README.md").lower()
    return {
        "repro": any(k in txt for k in ["reproduc", "reproducible", "cómo ejecutar", "make ", "pipeline"]),
        "mlflow": "mlflow" in txt,
        "dvc": "dvc" in txt or "data version control" in txt,
        "experiments": any(k in txt for k in ["experimento", "experimentos", "runs", "mlruns"]),
    }


def main():
    print_header(
        f"4) Seguimiento de Experimentos, Visualización y Gestión de Modelos — Chequeo estático "
        f"(módulo {MODULE_NAME}, raíz {ROOT})"
    )

    # MLflow en código
    mlflow_patterns = [
        r"\bmlflow\.set_tracking_uri\b",
        r"\bmlflow\.set_experiment\b",
        r"\bmlflow\.start_run\b",
        r"\bmlflow\.end_run\b",
        r"\bmlflow\.log_param\b",
        r"\bmlflow\.log_params\b",
        r"\bmlflow\.log_metric\b",
        r"\bmlflow\.log_metrics\b",
        r"\bmlflow\.log_artifact\b",
        r"\bmlflow\.log_artifacts\b",
        r"\bmlflow\.autolog\b",
        r"\bmlflow\.sklearn\.log_model\b",
        r"\bmlflow\.register_model\b",
        r"\bMlflowClient\b",
        r"\btransition_model_version_stage\b",
        r"\bcreate_registered_model\b",
        r"\bcreate_model_version\b",
    ]
    mlflow_hits = grep_code(mlflow_patterns, MODULE_NAME)

    print(pad("MLflow en código (tracking básico)", 65), tag(
        any(k in mlflow_hits for k in [
            r"\bmlflow\.set_experiment\b",
            r"\bmlflow\.start_run\b",
            r"\bmlflow\.autolog\b",
            r"\bmlflow\.log_param\b",
            r"\bmlflow\.log_metric\b",
            r"\bmlflow\.log_artifact\b"
        ]),
        REQUIRED
    ))

    print(pad("MLflow: configuración explícita (set_tracking_uri)", 65),
          tag(r"\bmlflow\.set_tracking_uri\b" in mlflow_hits, RECOMMENDED))

    print(pad("MLflow: registro de modelo (log_model/registry)", 65),
          tag(any(k in mlflow_hits for k in [
              r"\bmlflow\.sklearn\.log_model\b",
              r"\bmlflow\.register_model\b",
              r"\bcreate_model_version\b"
          ]), RECOMMENDED))

    print(pad("MLflow: uso de Model Registry (MlflowClient/transition stage)", 65),
          tag(any(k in mlflow_hits for k in [
              r"\bMlflowClient\b",
              r"\btransition_model_version_stage\b",
              r"\bcreate_registered_model\b"
          ]), RECOMMENDED))

    # mlruns en disco
    info = scan_mlruns()
    print(pad("mlruns/ presente (tracking local)", 65), tag(info["exists"], OPTIONAL))
    print(pad("Experimentos en mlruns/ (>=1)", 65), tag(info["n_experiments"] > 0, RECOMMENDED))
    print(pad("Runs en mlruns/ (>=2 para comparar)", 65), tag(info["n_runs"] >= 2, RECOMMENDED))
    print(pad("Runs con parámetros (params/)", 65), tag(info["runs_with_params"] > 0, REQUIRED))
    print(pad("Runs con métricas (metrics/)", 65), tag(info["runs_with_metrics"] > 0, REQUIRED))
    print(pad("Runs con artifacts (artifacts/)", 65), tag(info["runs_with_artifacts"] > 0, RECOMMENDED))
    print(pad("Artifacts de modelo MLflow (artifacts/model/MLmodel)", 65),
          tag(info["runs_with_model_artifact"] > 0, RECOMMENDED))
    print(pad("Artifacts de visualización (png/html/pdf) en mlruns/", 65),
          tag(info["runs_with_visuals"] > 0, RECOMMENDED))

    # Gestión de modelos
    saved_models = scan_models_dir()
    model_versioning_ok = (len(saved_models) > 0) or (info["runs_with_model_artifact"] > 0)
    print(pad("Artefactos de modelo versionados (carpeta models/ o MLflow)", 65),
          tag(model_versioning_ok, RECOMMENDED))

    # DVC
    dvc_yaml = has_path("dvc.yaml")
    dvc_lock = has_path("dvc.lock")
    dvc_cfg_path, remotes, urls = dvc_config_info()
    dvc_files, has_lock = scan_dvc_files()

    print(pad("DVC: dvc.yaml (pipeline/datos versionados)", 65), tag(dvc_yaml, REQUIRED))
    print(pad("DVC: dvc.lock (congelado de pipeline/datos)", 65), tag(dvc_lock or has_lock, RECOMMENDED))
    print(pad("DVC: .dvc/config (remotos configurados)", 65), tag(bool(dvc_cfg_path), REQUIRED))

    has_remote = bool(remotes)
    has_s3 = any(u.strip().startswith("s3://") for u in urls)
    print(pad("DVC: remoto configurado (alguno)", 65), tag(has_remote, REQUIRED))
    print(pad("DVC: remoto S3 (recomendado en tu proyecto)", 65), tag(has_s3, RECOMMENDED))

    print(pad("DVC: archivos .dvc (datos bajo control de DVC)", 65),
          tag(len(dvc_files) > 0, RECOMMENDED))

    # Documentación / params
    params_keys = read_params_yaml_keys()
    print(pad("params.yaml con claves (hiperparámetros/config)", 65),
          tag(len(params_keys) > 0, RECOMMENDED))

    r = readme_has_sections()
    print(pad("README: cómo reproducir/ejecutar (make/pipeline)", 65),
          tag(r["repro"], RECOMMENDED))
    print(pad("README: sección sobre MLflow/experimentos", 65),
          tag(r["mlflow"] or r["experiments"], RECOMMENDED))
    print(pad("README: sección sobre DVC/datos", 65),
          tag(r["dvc"], RECOMMENDED))

    print("\nLeyenda de estados:")
    print("  [PRESENTE]           Regla satisfecha")
    print("  [FALTA-REQUERIDO]    Debe corregirse para cumplir seguimiento/versionado mínimo")
    print("  [FALTA-RECOMENDADO]  Muy aconsejable para comparación/visualización/registro")
    print("  [FALTA-OPCIONAL]     Depende del contexto (por ejemplo tracking remoto sin mlruns local)")

    print("\nSugerencias:")
    if not any(k in mlflow_hits for k in [
        r"\bmlflow\.set_experiment\b",
        r"\bmlflow\.start_run\b",
        r"\bmlflow\.autolog\b"
    ]):
        print(" - Inicia y nombra experimentos en MLflow (set_experiment/start_run/autolog).")
    if not any(k in mlflow_hits for k in [
        r"\bmlflow\.log_param\b",
        r"\bmlflow\.log_metric\b"
    ]):
        print(" - Registra parámetros y métricas clave por run (mlflow.log_params / log_metrics).")
    if info["exists"] and info["n_runs"] < 2:
        print(" - Genera al menos 2 corridas comparables para justificar selección de modelo en la entrega.")
    if info["exists"] and info["runs_with_visuals"] == 0:
        print(" - Registra curvas ROC/PR, matriz de confusión, etc. como artifacts (png/html) en MLflow.")
    if not (dvc_yaml and (dvc_lock or has_lock)):
        print(" - Define/actualiza pipeline en dvc.yaml y congélalo con dvc.lock (dvc repro; dvc commit).")
    if not remotes:
        print(" - Configura un remoto DVC (ej. S3) y empuja datos/modelos con dvc push.")
    if remotes and not has_s3:
        print(" - Considera remoto S3 para alinear con la infraestructura del equipo (eu-north-1).")
    if not model_versioning_ok:
        print(" - Versiona modelos: guarda en models/*.joblib o como artifacts de MLflow (log_model / register_model).")
    if dvc_yaml and len(dvc_files) == 0:
        print(" - Versiona datasets grandes con .dvc (dvc add data/raw/...; dvc push).")
    if len(params_keys) == 0:
        print(" - Centraliza hiperparámetros y rutas en params.yaml y cárgalos desde el código.")
    if not (r["repro"] and (r["mlflow"] or r["experiments"]) and r["dvc"]):
        print(" - Documenta en README: cómo reproducir (Makefile), cómo ver resultados en MLflow, "
              "y cómo restaurar datos/modelos con DVC.")


if __name__ == "__main__":
    main()
