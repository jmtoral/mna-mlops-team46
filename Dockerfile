# Dockerfile para German Credit Risk API
# Equipo 46 - MLOps

FROM python:3.11-slim

# Metadata
LABEL maintainer="Equipo 46 <equipo46@tec.mx>"
LABEL description="API de predicción de riesgo crediticio con XGBoost"
LABEL version="1.0.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/xgboost_model.pkl

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements y instalar dependencias
COPY requirements-api.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements-api.txt

# Copiar código de la aplicación
COPY api.py .

# Crear directorio para el modelo
RUN mkdir -p /app/models

# Nota: El modelo debe ser montado o copiado en /app/models/xgboost_model.pkl
# Ejemplo: docker run -v $(pwd)/models:/app/models ...

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Comando de inicio
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]