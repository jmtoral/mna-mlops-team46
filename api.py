"""
API FastAPI para el Modelo de Riesgo Crediticio Alemán
Equipo: 46 | MLOps | Sep-Dic 2025

Este servicio expone el modelo XGBoost entrenado para predicción de riesgo crediticio.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metadata de la API
app = FastAPI(
    title="German Credit Risk API",
    description="API para predicción de riesgo crediticio usando XGBoost",
    version="1.0.0",
    contact={
        "name": "Equipo 46 - MLOps",
        "email": "equipo46@tec.mx"
    },
    license_info={
        "name": "MIT",
    }
)

# ============================================================================
# MODELOS PYDANTIC PARA VALIDACIÓN
# ============================================================================

class CreditInput(BaseModel):
    """
    Schema de entrada para una solicitud de predicción individual.
    
    Todos los campos corresponden a las características del dataset German Credit.
    """
    # Variables numéricas
    duration: int = Field(..., ge=1, le=72, description="Duración del crédito en meses (1-72)")
    amount: int = Field(..., ge=250, le=20000, description="Monto del crédito en DM (250-20000)")
    age: int = Field(..., ge=18, le=100, description="Edad del solicitante (18-100)")
    
    # Variables categóricas (codificadas como enteros según el dataset)
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
    
    class Config:
        schema_extra = {
            "example": {
                "duration": 12,
                "amount": 5000,
                "age": 35,
                "status": 1,
                "credit_history": 2,
                "purpose": 3,
                "savings": 1,
                "employment_duration": 3,
                "installment_rate": 2,
                "personal_status_sex": 2,
                "other_debtors": 1,
                "present_residence": 3,
                "property": 1,
                "other_installment_plans": 1,
                "housing": 1,
                "number_credits": 1,
                "job": 2,
                "people_liable": 1,
                "telephone": 2,
                "foreign_worker": 2
            }
        }


class BatchCreditInput(BaseModel):
    """Schema para predicciones por lote."""
    applications: List[CreditInput] = Field(..., min_items=1, max_items=100)
    
    class Config:
        schema_extra = {
            "example": {
                "applications": [
                    {
                        "duration": 12, "amount": 5000, "age": 35,
                        "status": 1, "credit_history": 2, "purpose": 3,
                        "savings": 1, "employment_duration": 3,
                        "installment_rate": 2, "personal_status_sex": 2,
                        "other_debtors": 1, "present_residence": 3,
                        "property": 1, "other_installment_plans": 1,
                        "housing": 1, "number_credits": 1, "job": 2,
                        "people_liable": 1, "telephone": 2, "foreign_worker": 2
                    }
                ]
            }
        }


class PredictionOutput(BaseModel):
    """Schema de salida para predicción individual."""
    prediction: int = Field(..., description="Predicción: 0=Riesgo Alto, 1=Riesgo Bajo")
    probability_good: float = Field(..., ge=0, le=1, description="Probabilidad de buen crédito")
    probability_bad: float = Field(..., ge=0, le=1, description="Probabilidad de mal crédito")
    risk_level: str = Field(..., description="Nivel de riesgo: BAJO, MEDIO, ALTO")
    timestamp: str = Field(..., description="Timestamp de la predicción")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability_good": 0.78,
                "probability_bad": 0.22,
                "risk_level": "BAJO",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


class BatchPredictionOutput(BaseModel):
    """Schema de salida para predicciones por lote."""
    predictions: List[PredictionOutput]
    total_processed: int
    timestamp: str


class HealthResponse(BaseModel):
    """Schema para el health check."""
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Schema para respuestas de error."""
    error: str
    detail: str
    timestamp: str


# ============================================================================
# CARGA DEL MODELO
# ============================================================================

class ModelLoader:
    """Clase para cargar y mantener el modelo en memoria."""
    
    def __init__(self, model_path: str = "models/xgboost_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.model_version = "1.0.0"
        self.load_model()
    
    def load_model(self):
        """Carga el modelo desde el archivo pickle."""
        try:
            if not self.model_path.exists():
                logger.error(f"Modelo no encontrado en: {self.model_path}")
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            
            logger.info(f"Modelo cargado exitosamente desde {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            raise
    
    def predict(self, input_data: pd.DataFrame) -> tuple:
        """
        Realiza predicciones con el modelo.
        
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado")
        
        try:
            predictions = self.model.predict(input_data)
            probabilities = self.model.predict_proba(input_data)
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            raise


# Instancia global del modelo
try:
    model_loader = ModelLoader()
except Exception as e:
    logger.warning(f"No se pudo cargar el modelo al inicio: {str(e)}")
    model_loader = None


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def input_to_dataframe(credit_input: CreditInput) -> pd.DataFrame:
    """Convierte el input Pydantic a DataFrame."""
    data = {
        'status': [credit_input.status],
        'duration': [credit_input.duration],
        'credit_history': [credit_input.credit_history],
        'purpose': [credit_input.purpose],
        'amount': [credit_input.amount],
        'savings': [credit_input.savings],
        'employment_duration': [credit_input.employment_duration],
        'installment_rate': [credit_input.installment_rate],
        'personal_status_sex': [credit_input.personal_status_sex],
        'other_debtors': [credit_input.other_debtors],
        'present_residence': [credit_input.present_residence],
        'property': [credit_input.property],
        'age': [credit_input.age],
        'other_installment_plans': [credit_input.other_installment_plans],
        'housing': [credit_input.housing],
        'number_credits': [credit_input.number_credits],
        'job': [credit_input.job],
        'people_liable': [credit_input.people_liable],
        'telephone': [credit_input.telephone],
        'foreign_worker': [credit_input.foreign_worker]
    }
    return pd.DataFrame(data)


def determine_risk_level(probability_good: float) -> str:
    """Determina el nivel de riesgo basado en la probabilidad."""
    if probability_good >= 0.7:
        return "BAJO"
    elif probability_good >= 0.4:
        return "MEDIO"
    else:
        return "ALTO"


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "German Credit Risk API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            #"predict_batch": "/predict/batch",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check del servicio.
    
    Verifica que el modelo esté cargado y el servicio operativo.
    """
    return HealthResponse(
        status="healthy" if model_loader and model_loader.model else "unhealthy",
        model_loaded=model_loader is not None and model_loader.model is not None,
        model_version=model_loader.model_version if model_loader else "unknown",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.post("/predict", 
          response_model=PredictionOutput, 
          status_code=status.HTTP_200_OK,
          tags=["Predicción"],
          summary="Predice riesgo crediticio individual",
          description="Realiza una predicción de riesgo crediticio para una solicitud individual")
async def predict(credit_input: CreditInput):
    """
    Endpoint principal de predicción.
    
    Recibe los datos de una solicitud de crédito y retorna:
    - Predicción binaria (0=Alto Riesgo, 1=Bajo Riesgo)
    - Probabilidades para cada clase
    - Nivel de riesgo categórico
    """
    try:
        # Verificar que el modelo esté cargado
        if model_loader is None or model_loader.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo no disponible. El servicio está inicializando."
            )
        
        # Convertir input a DataFrame
        df_input = input_to_dataframe(credit_input)
        
        # Realizar predicción
        predictions, probabilities = model_loader.predict(df_input)
        
        prediction = int(predictions[0])
        prob_bad = float(probabilities[0][0])
        prob_good = float(probabilities[0][1])
        
        # Determinar nivel de riesgo
        risk_level = determine_risk_level(prob_good)
        
        logger.info(f"Predicción exitosa: {prediction}, probabilidad_bueno={prob_good:.3f}")
        
        return PredictionOutput(
            prediction=prediction,
            probability_good=prob_good,
            probability_bad=prob_bad,
            risk_level=risk_level,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar la predicción: {str(e)}"
        )




# ============================================================================
# MANEJO DE ERRORES
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Manejador personalizado de excepciones HTTP."""
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
    """Manejador de excepciones generales."""
    logger.error(f"Error no manejado: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Error interno del servidor",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)