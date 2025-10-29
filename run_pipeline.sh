#!/bin/bash
# Quitamos temporalmente 'set -e' para depurar
# set -e

# --- Función Auxiliar (Más Detallada) ---
check() {
    TYPE=$1
    TARGET=$2
    MSG_SUCCESS=$3
    MSG_FAIL=${4:-$3}

    echo -n "  Verificando: ${MSG_SUCCESS} ... "

    COMMAND_STATUS=1 # Asumir fallo

    case "$TYPE" in
        dir)
            if [ -d "$TARGET" ]; then COMMAND_STATUS=0; fi
            ;;
        file)
            if [ -f "$TARGET" ]; then COMMAND_STATUS=0; fi
            ;;
        text)
            TEXT_TO_FIND=$2
            FILE_TO_SEARCH=$3
            MSG_SUCCESS=$4
            MSG_FAIL=${5:-$4}
            echo -n "  Verificando: ${MSG_SUCCESS} ... " # Repetir mensaje
            # Usar ruta completa a grep si es necesario
            if command -v grep &>/dev/null && grep -q -i -E "$TEXT_TO_FIND" "$FILE_TO_SEARCH"; then COMMAND_STATUS=0; fi
            ;;
        dvcfile)
            if [ -f "${TARGET}.dvc" ]; then COMMAND_STATUS=0; fi
            ;;
        *)
            echo " TIPO DE CHECK DESCONOCIDO: $TYPE"
            return 1
            ;;
    esac

    if [ $COMMAND_STATUS -eq 0 ]; then
        echo "✅ OK"
        return 0
    else
        echo "❌ FALLÓ"
        # Mostrar directorio actual si falla un chequeo de archivo/directorio
        if [[ "$TYPE" == "dir" || "$TYPE" == "file" || "$TYPE" == "dvcfile" ]]; then
             echo "     -> ¿Estás en el directorio correcto? Actual: $(pwd)"
             echo "     -> Buscando: '$TARGET'"
        fi
        if [ "$MSG_SUCCESS" != "$MSG_FAIL" ]; then
            echo "     -> $MSG_FAIL"
        fi
        # Si queremos que el script pare aquí, descomentar la siguiente línea
        # exit 1
        return 1
    fi
}

# --- Bloque de Chequeos de la Rúbrica ---
echo ""
echo "=================================================="
echo " 📋 CHEQUEO DE CUMPLIMIENTO (RÚBRICA FASE 2) "
echo "=================================================="

# 1) Estructura de Proyecto
echo "[INFO] 1. Verificando Estructura del Proyecto..."
STRUCTURE_OK=true
check dir "data/raw" "Carpeta 'data/raw' existe" || STRUCTURE_OK=false
check dir "data/processed" "Carpeta 'data/processed' existe" || STRUCTURE_OK=false
check dir "models" "Carpeta 'models' existe" || STRUCTURE_OK=false
check dir "notebooks" "Carpeta 'notebooks' existe" || STRUCTURE_OK=false
check dir "reports/figures" "Carpeta 'reports/figures' existe" || STRUCTURE_OK=false
check dir "german_credit_ml" "Carpeta de código fuente ('german_credit_ml') existe" || STRUCTURE_OK=false
if [ "$STRUCTURE_OK" = false ]; then
    echo "  ⚠️  Advertencia: Faltan algunas carpetas clave de la estructura."
fi

# 2) Refactorización del Código
echo "[INFO] 2. Verificando Refactorización (Scripts Principales)..."
REFACTOR_OK=true
check file "german_credit_ml/clean.py" "Script 'clean.py' existe" || REFACTOR_OK=false
check file "german_credit_ml/eda.py" "Script 'eda.py' existe" || REFACTOR_OK=false
check file "german_credit_ml/modeling/train.py" "Script 'train.py' existe" || REFACTOR_OK=false
if [ "$REFACTOR_OK" = false ]; then
    echo "  ⚠️  Advertencia: Faltan algunos scripts .py esenciales."
fi

# 3) Uso de Pipeline Scikit-Learn
echo "[INFO] 3. Verificando uso de Pipeline Scikit-Learn en train.py..."
check text "Pipeline\(|make_pipeline\(" "german_credit_ml/modeling/train.py" \
    "Uso de Pipeline/make_pipeline encontrado en train.py" \
    "Uso de Pipeline/make_pipeline NO encontrado en train.py (Revisar manualmente)"

# 4) Seguimiento y Versionado
echo "[INFO] 4. Verificando Herramientas de Versionado y Tracking..."
check dir ".dvc" "Directorio '.dvc' (DVC inicializado) existe"
check file "dvc.yaml" "Archivo 'dvc.yaml' (Pipeline DVC) existe" # <-- Este es el que fallaba
check dvcfile "data/processed/german_credit_clean.csv" "Datos limpios rastreados por DVC"
check dvcfile "models/xgboost_model.pkl" "Modelo rastreado por DVC"
check dvcfile "reports/figures/training" "Gráficas de entrenamiento rastreadas por DVC"
check dvcfile "reports/metrics.json" "Métricas rastreadas por DVC"
check text "import mlflow" "german_credit_ml/modeling/train.py" \
    "Importación de MLflow encontrada en train.py" \
    "Importación de MLflow NO encontrada en train.py"
check dir "mlruns" "Directorio 'mlruns' (logs de MLflow) existe" "Directorio 'mlruns' NO existe (ejecuta el pipeline al menos una vez)"

echo "=================================================="
echo ""

# --- Chequeos de Estado Previos (Git & DVC Status) ---
# Ahora sí debería llegar a esta parte
echo "=================================================="
echo " 🩺 CHEQUEO DE ESTADO PREVIO (GIT & DVC) "
echo "=================================================="
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "🌿 Rama Git actual: $CURRENT_BRANCH"
echo "📊 Estado de Git:"
git status --short
echo "📦 Estado de DVC:"
dvc status || true
echo "=================================================="
echo ""

# --- Ejecutar el pipeline ---
echo "🚀 Reproduciendo el pipeline con 'dvc repro'..."
dvc repro

# --- Verificar cambios y hacer commit/push ---
echo ""
echo "💾 Verificando si hay cambios para guardar..."
# Verifica si dvc.yaml o dvc.lock fueron modificados
if [[ -z $(git status --porcelain dvc.yaml dvc.lock) ]]; then
    echo "✅ Pipeline verificado. No se detectaron cambios en las salidas."
else
    echo "   -> Nueva versión de artefactos detectada. Guardando cambios..."
    git add dvc.yaml dvc.lock
    # Intenta añadir .gitignore si existe (para outputs nuevos)
    git add reports/.gitignore models/.gitignore data/processed/.gitignore 2>/dev/null || true
    git commit -m "pipe: Regenerate pipeline outputs"
    echo "☁️ Subiendo metadatos a Git y datos a DVC (S3)..."
    git push
    dvc push
    echo "🎉 ¡Listo! Nueva versión del pipeline guardada y subida."
fi

echo ""
echo "✅ Script finalizado."