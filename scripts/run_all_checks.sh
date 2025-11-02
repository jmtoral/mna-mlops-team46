#!/bin/bash
# ==============================================================
# Ejecuta los 4 validadores MLOps (estructura, refactor, pipeline, tracking)
# --------------------------------------------------------------
# Uso:
#   bash scripts/run_all_checks.sh
#
# Parámetros configurables:
#   PROJECT_ROOT : raíz del proyecto
#   MODULE_NAME  : nombre del paquete principal (Cookiecutter v2)
# ==============================================================

set -e  # detener si algún script falla
set -u  # error si hay variables no definidas

# --- CONFIGURACIÓN ---
PROJECT_ROOT="C:/dev/mna-mlops-team46"
MODULE_NAME="german_credit_ml"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# --- LOG FILE ---
timestamp=$(date +"%Y%m%d_%H%M")
LOG_FILE="${LOG_DIR}/validation_${timestamp}.txt"

echo "==============================================================" | tee "$LOG_FILE"
echo "  VALIDACIÓN COMPLETA DEL PROYECTO MLOps" | tee -a "$LOG_FILE"
echo "  Raíz del proyecto : $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "  Módulo Python      : $MODULE_NAME" | tee -a "$LOG_FILE"
echo "  Fecha de ejecución : $(date)" | tee -a "$LOG_FILE"
echo "==============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# --- EJECUCIÓN SECUENCIAL DE LOS SCRIPTS ---
for script in \
  "01_check_structure.py" \
  "02_check_refactor.py" \
  "03_check_sklearn_pipeline.py" \
  "04_check_tracking_models.py"
do
  echo "--------------------------------------------------------------" | tee -a "$LOG_FILE"
  echo "Ejecutando $script ..." | tee -a "$LOG_FILE"
  echo "--------------------------------------------------------------" | tee -a "$LOG_FILE"

  python "scripts/$script" \
    --project-root "$PROJECT_ROOT" \
    --module-name "$MODULE_NAME" | tee -a "$LOG_FILE"

  echo "" | tee -a "$LOG_FILE"
done

echo "==============================================================" | tee -a "$LOG_FILE"
echo "  VALIDACIÓN COMPLETA FINALIZADA CORRECTAMENTE" | tee -a "$LOG_FILE"
echo "  Log guardado en: $LOG_FILE"
echo "==============================================================" | tee -a "$LOG_FILE"
