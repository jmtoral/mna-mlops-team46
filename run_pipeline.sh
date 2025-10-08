#!/bin/bash

# Este comando asegura que el script se detenga si algún paso falla.
set -e

# --- PASO 1: Ejecutar el pipeline ---
# Llama a 'dvc repro' para que DVC revise las dependencias (código, datos)
# y vuelva a ejecutar la etapa de limpieza si algo cambió.
echo "🚀 Reproduciendo el pipeline con 'dvc repro'..."

echo "Runner : $(whoami) @ $(hostname)"
echo "Branch : $(git rev-parse --abbrev-ref HEAD)"
echo "Commit : $(git rev-parse --short HEAD)"

# (opcional) fija tracking local de MLflow para no mezclar:
export MLFLOW_TRACKING_URI="file:$(pwd)/mlruns"
export MLFLOW_EXPERIMENT_NAME="fase1_modelado_equipo46"

dvc repro --force

# --- PASO 2: Verificar si hubo cambios ---
# Se revisa si el archivo 'dvc.yaml' fue modificado. Si 'dvc repro' generó
# una nueva versión de los datos, este archivo habrá cambiado.
echo "💾 Verificando si hay cambios para guardar..."
if [[ -z $(git status --porcelain dvc.yaml) ]]; then
    # Si no hay cambios, el script lo notifica y termina.
    echo "✅ Pipeline verificado. No se detectaron cambios en la salida de datos."
else
    # Si SÍ hay cambios, procede a guardar la nueva versión.
    echo "   -> Nueva versión de datos detectada. Guardando cambios..."
    
    # --- PASO 3: Guardar la nueva versión ---
    # Se añade el 'dvc.yaml' modificado a Git y se hace un commit.
    # Este commit es el registro permanente de la nueva versión de tus datos.
    git add dvc.yaml
    git commit -m "pipe: Regenerate cleaned data"
    
    # --- PASO 4: Subir todo a los remotos ---
    # Sube los cambios del código/metadatos a GitHub y los datos a DVC (S3).
    echo "☁️ Subiendo metadatos a Git y datos a DVC (S3)..."
    git push
    dvc push
    
    echo "🎉 ¡Listo! Nueva versión del pipeline guardada y subida."
fi