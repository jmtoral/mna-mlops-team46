# 🏦 German Credit Risk API

**Equipo:** 46  
**Curso:** Operaciones de Aprendizaje Automático (MLOps)  
**Periodo:** Sep – Dic 2025

API REST para predicción de riesgo crediticio usando XGBoost sobre el dataset German Credit.

---

## 📋 Tabla de Contenidos

- [Características](#características)
- [Modelo](#modelo)
- [Instalación](#instalación)
- [Uso](#uso)
- [API Endpoints](#api-endpoints)
- [Esquemas de Datos](#esquemas-de-datos)

---

## ✨ Características

- ✅ **Endpoint POST `/predict`** para predicciones individuales
- ✅ **Validación de entrada** con Pydantic
- ✅ **Manejo robusto de errores** con códigos HTTP apropiados
- ✅ **Documentación automática** con Swagger
- ✅ **Health check** endpoint
- ✅ **Dockerizado** para portabilidad

---

## 🤖 Modelo

### Información del Artefacto

- **Tipo de Modelo:** XGBoost Classifier (Pipeline completo)
- **Ruta del Modelo:** `models/xgboost_model.pkl`
- **Versión:** 1.0.0
- **Framework:** scikit-learn + XGBoost
- **Registro MLflow:** `models:/<german-credit-xgboost>/<1>`

### Características del Modelo

**Variables numéricas (3):**
- `duration`: Duración del crédito en meses
- `amount`: Monto del crédito en DM
- `age`: Edad del solicitante

**Variables categóricas (17):**
- `status`: Estado de cuenta corriente
- `credit_history`: Historial crediticio
- `purpose`: Propósito del crédito
- `savings`: Ahorros/bonos
- `employment_duration`: Tiempo de empleo actual
- `installment_rate`: Tasa de cuota
- `personal_status_sex`: Estado personal y sexo
- `other_debtors`: Otros deudores/garantes
- `present_residence`: Residencia actual
- `property`: Propiedad
- `other_installment_plans`: Otros planes de cuotas
- `housing`: Vivienda
- `number_credits`: Número de créditos existentes
- `job`: Tipo de trabajo
- `people_liable`: Personas a cargo
- `telephone`: Disponibilidad de teléfono
- `foreign_worker`: Trabajador extranjero

### Preprocesamiento

El pipeline incluye:
1. **Imputación:** Mediana para numéricas, moda para categóricas
2. **Codificación:** OneHotEncoder para variables categóricas
3. **Clasificación:** XGBoost con 150 estimadores

### Métricas de Rendimiento

```json
{
  "accuracy": 0.75,
  "precision": 0.72,
  "recall": 0.68,
  "f1_score": 0.70,
  "auc_roc": 0.79
}
```

---

## 🚀 Instalación

### Opción 1: Local

```bash
# Clonar repositorio
git clone https://github.com/jmtoral/mna-mlops-team46.git
cd mna-mlops-team46

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements-api.txt

```
### Docker

```bash
# Construir imagen
docker build -t german-credit-api:1.0.0 .

# Ejecutar contenedor
docker run -d `  -p 8000:8000 `  -v "${PWD}\models:/app/models:ro" `  -e MODEL_PATH=/app/models/xgboost_model.pkl `  --name credit-api `  german-credit-api:1.0.0
```

### Usar DockerHub

```bash
Publicar una nueva versión y actualizar latest
# Construir una nueva versión
docker build -t german-credit-api:1.1.0 .

# Etiquetar para tu repositorio en Docker Hub
docker tag german-credit-api:1.1.0 monicadelrivero/german-credit-api:1.1.0

#  Actualizar latest para que apunte a la estable
docker tag german-credit-api:1.1.0 monicadelrivero/german-credit-api:latest

# Subir ambas etiquetas
docker push monicadelrivero/german-credit-api:1.1.0
docker push monicadelrivero/german-credit-api:latest

Cómo hacer  pull descargar la imagen 
#Versión específica 
docker pull monicadelrivero/german-credit-api:1.0.0

#Ultima estable
docker pull monicadelrivero/german-credit-api:latest

#Ejecutar contenedor
docker run -d --name credit-api -p 8000:8000 -v "%cd%\models:/app/models:ro" -e MODEL_PATH=/app/models/xgboost_model.pkl german-credit-api:1.0.0
```

---

## 💻 Uso

### Iniciar el servidor

```bash
# Desarrollo local
uvicorn api:app --reload --host 0.0.0.0 --port 8000

```

La API estará disponible en: `http://localhost:8000`

### Documentación interactiva

- **Swagger UI:** http://localhost:8000/docs

---

## 🔌 API Endpoints

### 1. Health Check

```http
GET /health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### 2. Predicción 

```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "prediction": 1,
  "probability_good": 0.78,
  "probability_bad": 0.22,
  "risk_level": "BAJO",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Interpretación:**
- `prediction`: 0 = Alto Riesgo (malo), 1 = Bajo Riesgo (bueno)
- `probability_good`: Probabilidad de ser buen pagador
- `probability_bad`: Probabilidad de ser mal pagador
- `risk_level`: BAJO (≥0.7), MEDIO (0.4-0.7), ALTO (<0.4)



---


### Output Schema (PredictionOutput)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| prediction | int | 0=Alto Riesgo, 1=Bajo Riesgo |
| probability_good | float | Probabilidad de buen crédito |
| probability_bad | float | Probabilidad de mal crédito |
| risk_level | str | BAJO, MEDIO o ALTO |
| timestamp | str | ISO 8601 timestamp |

---

