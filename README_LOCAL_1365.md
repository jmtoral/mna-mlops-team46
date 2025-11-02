# mna-mlops-team46

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# Proyecto - Equipo 46

Este proyecto implementa un flujo de **MLOps** con el dataset **German Credit**, 
con el objetivo de predecir el riesgo crediticio de clientes a partir de datos históricos.  


---

## 🎯 Objetivos
- Analizar la problemática del dataset German Credit.
- Realizar **EDA** (Exploratory Data Analysis) y limpieza de datos.
- Aplicar técnicas de **preprocesamiento** (codificación, normalización, manejo de outliers).
- Implementar **versionado de datos** con DVC para trazabilidad.
- Construir, entrenar y evaluar **modelos de Machine Learning**.
- Documentar los resultados y roles de equipo en un flujo de trabajo estilo **MLOps**.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         german_credit_ml and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── german_credit_ml   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes german_credit_ml a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

---

## 📊 Resultados Esperados
- Comparación de dataset crudo vs dataset limpio (valores nulos, outliers, transformaciones).
- Visualización de patrones y correlaciones en los datos.
- Modelos base (Regresión Logística, Árboles de Decisión) entrenados y evaluados.
- Métricas reportadas: Accuracy, Recall, ROC-AUC.

---

## 🛠️ Herramientas Utilizadas
- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- **DVC** (Data Version Control)
- **GitHub** (control de versiones y colaboración)
- **Makefile** (automatización de tareas)
- **Jupyter Notebooks**

---

## 👥 Roles del Equipo
| Integrante | Matrícula | Rol |
|---|---|---|
| Jesús Alberto Jiménez Ramos | `A01796903` | 📊 Data Engineer |
| Mónica María Del Rivero Sánchez | `A01362368` | 👩‍🔬 Data Scientist |
| Montserrat Gaytán Morales | `A01332220` | 💻 Software Engineer |
| José Manuel Toral Cruz | `A01122243` | 🤖 ML Engineer |
| Jeanette Rios Martinez | `A01688888` | 🛠️ SRE / DevOps |

---

## 📑 Entregables
- Reporte en PDF (`reports/fase1_equipo46.pdf`) con:
  - Análisis de requerimientos (ML Canvas).
  - Exploración, limpieza y preprocesamiento de datos.
  - Versionado con DVC.
  - Construcción, ajuste y evaluación de modelos.
  - Conclusiones y reflexión final.
- Link al **video explicativo (5-10 min)** en equipo.
- Link a este repositorio de GitHub.

---

## 📜 Licencia
Este proyecto está bajo la licencia MIT. Ver archivo [LICENSE](LICENSE) para más detalles.


## OOP

# Credit Risk MLOps (scaffold)

Estructura modular con MLflow + DVC lista para integrar el `3_modelado.ipynb`.

## Requisitos
- Python 3.10+
- `pip install -e .` (dentro del directorio del proyecto)

## Ejecución
1) (Opcional) MLflow tracking server:
```
mlflow server --host 127.0.0.1 --port 5000
```
edita `params.yaml` → `mlflow.tracking_uri` si lo usas.

2) Entrenamiento:
```
python -m scripts.train run --params_path params.yaml
```
o con DVC:
```
dvc repro
```

## Mapeo Notebook → Módulos
- Carga/partición: `src/credit_risk/data.py`
- Transformaciones: `src/credit_risk/features.py`
- Modelado/búsqueda: `src/credit_risk/modeling.py`
- Métricas: `src/credit_risk/metrics.py`
- Figuras: `src/credit_risk/viz.py`
- Orquestación + MLflow: `src/credit_risk/pipeline.py`
- CLI: `scripts/train.py`

