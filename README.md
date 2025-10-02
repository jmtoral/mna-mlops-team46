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

---

## 📂 Estructura del Proyecto
```bash
├── LICENSE            <- Licencia abierta (ej. MIT)
├── Makefile           <- Atajos de ejecución (make data, make train, etc.)
├── README.md          <- Este documento
├── data
│   ├── external       <- Datos de terceros (ej. dataset limpio proporcionado)
│   ├── interim        <- Datos intermedios (transformaciones temporales)
│   ├── processed      <- Datos finales para modelado
│   └── raw            <- Datos crudos originales
│
├── docs               <- Apuntes, slides, notas de clase
├── models             <- Modelos entrenados y predicciones
├── notebooks          <- Notebooks de EDA y modelado (ej. `1.0-eda.ipynb`)
├── pyproject.toml     <- Configuración del proyecto
├── references         <- Diccionarios de datos, manuales
├── reports            <- Reportes en PDF, LaTeX, etc.
│   └── figures        <- Gráficas generadas
├── requirements.txt   <- Dependencias del proyecto
├── setup.cfg          <- Configuración de estilo (flake8)
└── mlops              <- Código fuente modular
    ├── config.py
    ├── dataset.py
    ├── features.py
    ├── modeling
    │   ├── train.py
    │   └── predict.py
    └── plots.py
```

---

## ⚙️ Instalación
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/jmtoral/mna-mlops-team46.git
   cd mna-mlops-team46
   ```

2. Crear entorno virtual e instalar dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate   # en Linux/Mac
   venv\Scripts\activate      # en Windows

   pip install -r requirements.txt
   ```

3. Inicializar DVC (si no está configurado):
   ```bash
   dvc init
   dvc pull   # recupera datasets desde almacenamiento remoto
   ```

---

## 🚀 Uso

### Preparar datos
```bash
make data
```

### Entrenar modelo
```bash
make train
```

### Realizar predicciones
```bash
make predict
```

### Ejecutar notebooks (EDA, limpieza, modelado)
```bash
jupyter notebook notebooks/
```

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
- Limpieza de datos: primer cambio local
