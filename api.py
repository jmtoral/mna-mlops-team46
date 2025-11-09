"""
API FastAPI para el Modelo de Riesgo Crediticio Alemán
Equipo: 46 | MLOps | Sep-Dic 2025

Este servicio expone el modelo XGBoost entrenado para predicción de riesgo crediticio.
"""

from fastapi import FastAPI, APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import List, Dict, Any
import logging
import pandas as pd
import pickle, json, os
from datetime import datetime
from pathlib import Path

# ======================== CONFIG ========================
class Settings(BaseSettings):
    MODEL_PATH: str = "models/xgboost_model.pkl"
    COLUMNS_PATH: str = "models/columns.json"
    MODEL_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"
    MAX_BATCH_SIZE: int = 100
    TEAM_NAME: str = "Equipo 46 - MLOps"
    TEAM_EMAIL: str = "equipo46@tec.mx"

    class Config:
        env_file = ".env"

settings = Settings()

# ======================== LOGGING ========================
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== MODEL LOADER ========================
class ModelLoader:
    """Clase responsable de cargar y mantener el modelo en memoria."""
    def __init__(self, model_path: str, columns_path: str):
        self.model_path = Path(model_path)
        self.columns_path = Path(columns_path)
        self.model = None
        self.columns = None
        self.model_version = settings.MODEL_VERSION
        self.load_model()

    def load_model(self):
        if not self.model_path.exists():
            logger.error(f"Modelo no encontrado en: {self.model_path}")
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.columns_path, 'r') as f:
                self.columns = json.load(f)
            logger.info(f"Modelo y columnas cargados exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise

    def predict(self, df: pd.DataFrame):
        try:
            preds = self.model.predict(df[self.columns])
            probas = self.model.predict_proba(df[self.columns])
            return preds.tolist(), probas.tolist()
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise HTTPException(status_code=500, detail="Error al generar predicciones")

model_loader = ModelLoader(settings.MODEL_PATH, settings.COLUMNS_PATH)

# ======================== SCHEMAS ========================
class CreditInput(BaseModel):
    duration: int = Field(..., ge=1, le=72, description="Duración del crédito en meses (1-72)")
    amount: int = Field(..., ge=250, le=20000, description="Monto del crédito en DM (250-20000)")
    age: int = Field(..., ge=18, le=100, description="Edad del solicitante (18-100)")
    status: int = Field(..., ge=1, le=4, description="Estado de cuenta corriente (1-4)")
    credit_history: int = Field(..., ge=0, le=4, description="Historial crediticio (0-4)")
    purpose: int = Field(..., ge=0, le=10, description="Propósito del crédito (0-10)")
    savings: int = Field(..., ge=1, le=5, description="Ahorros/bonos (1-5)")
    employment_duration: int = Field(..., ge=1, le=5, description="Tiempo de empleo actual (1-5)")
    installment_rate: int = Field(..., ge=1, le=4, description="Tasa de cuota (1-4)")
    personal_status_sex: int = Field(..., ge=1, le=4, description="Estado personal y sexo (1-4)")
    other_debtors: int = Field(..., ge=1, le=3, description="Otros deudores/garantes (1-3)")
    present_residence: int = Field(..., ge=1, le=4, description="Residencia actual en años (1-4)")
    property: int = Field(..., ge=1, le=4, description="Propiedad (1-4)")
    other_installment_plans: int = Field(..., ge=1, le=3, description="Otros planes de cuotas (1-3)")
    housing: int = Field(..., ge=1, le=3, description="Vivienda (1-3)")
    number_credits: int = Field(..., ge=1, le=4, description="Número de créditos existentes (1-4)")
    job: int = Field(..., ge=1, le=4, description="Tipo de trabajo (1-4)")
    people_liable: int = Field(..., ge=1, le=2, description="Personas a cargo (1-2)")
    telephone: int = Field(..., ge=1, le=2, description="Teléfono (1=no, 2=sí)")
    foreign_worker: int = Field(..., ge=1, le=2, description="Trabajador extranjero (1=sí, 2=no)")

class BatchCreditInput(BaseModel):
    applications: List[CreditInput] = Field(..., min_items=1, max_items=settings.MAX_BATCH_SIZE)

class PredictionOutput(BaseModel):
    prediction: int
    probability_good: float
    probability_bad: float
    risk_level: str
    timestamp: str

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total_processed: int
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str

# ======================== FUNCIONES AUXILIARES ========================
def determine_risk_level(probability_good: float) -> str:
    if probability_good >= 0.7:
        return "BAJO"
    elif probability_good >= 0.4:
        return "MEDIO"
    else:
        return "ALTO"

# ======================== ROUTERS ========================
router_health = APIRouter(prefix="/health", tags=["Health"])
router_predict = APIRouter(prefix="/predict", tags=["Predicción"])

@router_health.get("", response_model=HealthResponse)
def health_check():
    try:
        model_loader.load_model()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_version=model_loader.model_version,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_version=model_loader.model_version,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

@router_predict.post("", response_model=PredictionOutput)
def predict_single(request: CreditInput):
    df = pd.DataFrame([request.dict()])
    preds, probas = model_loader.predict(df)
    prediction = int(preds[0])
    prob_bad = float(probas[0][0])
    prob_good = float(probas[0][1])
    risk_level = determine_risk_level(prob_good)
    return PredictionOutput(
        prediction=prediction,
        probability_good=prob_good,
        probability_bad=prob_bad,
        risk_level=risk_level,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )

@router_predict.post("/batch", response_model=BatchPredictionOutput)
def predict_batch(request: BatchCreditInput):
    records = [r.dict() for r in request.applications]
    if len(records) > settings.MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch demasiado grande, máximo permitido: {settings.MAX_BATCH_SIZE}")
    df = pd.DataFrame(records)
    preds, probas = model_loader.predict(df)
    results = []
    for i, row in enumerate(records):
        prob_bad = float(probas[i][0])
        prob_good = float(probas[i][1])
        risk_level = determine_risk_level(prob_good)
        results.append(PredictionOutput(
            prediction=int(preds[i]),
            probability_good=prob_good,
            probability_bad=prob_bad,
            risk_level=risk_level,
            timestamp=datetime.utcnow().isoformat() + "Z"
        ))
    return BatchPredictionOutput(
        predictions=results,
        total_processed=len(results),
        timestamp=datetime.utcnow().isoformat() + "Z"
    )

# ======================== FASTAPI APP ========================
app = FastAPI(
    title="German Credit Risk API",
    description="API para predicción de riesgo crediticio (XGBoost) - Production-ready",
    version=settings.MODEL_VERSION,
    contact={
        "name": settings.TEAM_NAME,
        "email": settings.TEAM_EMAIL
    }
)

app.include_router(router_health)
app.include_router(router_predict)

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "German Credit Risk API",
        "version": settings.MODEL_VERSION,
        "team": {
            "name": settings.TEAM_NAME,
            "email": settings.TEAM_EMAIL
        },
        "endpoints": ["/predict", "/predict/batch", "/health", "/docs"]
    }

# ======================== MANEJO DE ERRORES ========================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Error interno del servidor",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )

# ======================== RUN ========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
