# monitor_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import altair as alt

# --- Importar nuestros módulos personalizados ---
# Asegúrate de que german_credit_ml/drift_utils.py exista con las funciones correctas
from german_credit_ml.drift_utils import run_drift_analysis, calculate_ks_test, calculate_psi
# Reutilizamos la función de carga de modelo de tu otra app
from predict_app import load_model_and_metadata 

# Ignorar advertencias futuras y de usuario para una salida limpia
warnings.filterwarnings("ignore")

# --- Configuración de la Página ---
st.set_page_config(page_title="Dashboard Monitoreo MLOps", layout="wide", page_icon="📡")
st.title("📡 Centro de Monitoreo de Datos & Modelo")

# --- Constantes y Rutas ---
CLEAN_DATA_PATH = Path("data/processed/german_credit_clean.csv")
DRIFTED_DATA_PATH = Path("data/processed/german_credit_drifted.csv")
CHAOS_DATA_PATH = Path("data/processed/german_credit_chaos.csv")
CONCEPT_DATA_PATH = Path("data/processed/german_credit_concept.csv")

# Columnas esperadas por el modelo (para asegurar orden correcto al predecir)
EXPECTED_RAW_COLS = [
    'status', 'duration', 'credit_history', 'purpose', 'amount',
    'savings', 'employment_duration', 'installment_rate',
    'personal_status_sex', 'other_debtors', 'present_residence',
    'property', 'age', 'other_installment_plans', 'housing',
    'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker'
]

# --- Sidebar de Configuración ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selector de Escenario de Datos
    data_source = st.radio(
        "Fuente de Datos Recientes:",
        [
            "Producción (Normal)", 
            "Simulación (Crisis/Data Drift)", 
            "Simulación (Caos/Adversario)",
            "Simulación (Concept/Estructural)"
        ],
        help="Selecciona un escenario para probar las capacidades de monitoreo del sistema."
    )
    
    st.markdown("---")
    st.caption("Definición de Referencia:")
    # Slider para definir qué parte de los datos originales se usa como "base"
    split_point = st.slider("Corte de Entrenamiento:", 50, 90, 70) / 100

# --- Lógica de Carga de Datos ---
if not CLEAN_DATA_PATH.exists():
    st.error("❌ No se encuentran los datos base (german_credit_clean.csv). Ejecuta el pipeline `bash run_pipeline.sh` primero.")
    st.stop()

# Cargar datos originales limpios
df_clean = pd.read_csv(CLEAN_DATA_PATH)

# 1. Definir Referencia (Siempre viene de la primera parte de los datos limpios originales - Train)
split_idx = int(len(df_clean) * split_point)
reference_df = df_clean.iloc[:split_idx].copy()

# 2. Definir Datos Actuales (Depende del selector)
if data_source == "Producción (Normal)":
    # Usamos el resto del archivo original (Test/Producción normal)
    current_df = df_clean.iloc[split_idx:].copy()
    st.success(f"✅ Monitoreando flujo normal de datos ({len(current_df)} registros).")

elif data_source == "Simulación (Crisis/Data Drift)":
    if not DRIFTED_DATA_PATH.exists():
        st.warning("⚠️ No se encontró el archivo de simulación drift.")
        st.info("Ejecuta: `python simulate_drift.py` en tu terminal.")
        st.stop()
    current_df = pd.read_csv(DRIFTED_DATA_PATH)
    st.warning(f"⚠️ MODO CRISIS: Datos con drift severo ({len(current_df)} registros).")

elif data_source == "Simulación (Caos/Adversario)":
    if not CHAOS_DATA_PATH.exists():
        st.warning("⚠️ No se encontró el archivo de simulación caos.")
        st.info("Ejecuta: `python simulate_chaos.py` en tu terminal.")
        st.stop()
    current_df = pd.read_csv(CHAOS_DATA_PATH)
    st.error(f"🔥 MODO CAOS: Datos aleatorizados ({len(current_df)} registros).")

elif data_source == "Simulación (Concept/Estructural)":
    if not CONCEPT_DATA_PATH.exists():
        st.warning("⚠️ No se encontró el archivo de simulación concept.")
        st.info("Ejecuta: `python simulate_concept.py` en tu terminal.")
        st.stop()
    current_df = pd.read_csv(CONCEPT_DATA_PATH)
    st.info(f"🧠 MODO CONCEPT: Datos estructuralmente incoherentes ({len(current_df)} registros).")


# Cargar modelo para Prediction Drift (si existe)
model, _, _, _ = load_model_and_metadata()


# --- Ejecución del Análisis ---
st.markdown("---")
col1, col2 = st.columns([1, 3])

with col1:
    st.metric("Registros Referencia", len(reference_df))
with col2:
    st.metric("Registros Actuales", len(current_df), delta="Fuente: " + data_source)

# Botón principal de ejecución
if st.button("🔍 Ejecutar Diagnóstico Completo", type="primary", use_container_width=True):
    
    # === 1. DATA DRIFT (Variables de Entrada) ===
    st.header("1. Análisis de Data Drift (Entradas)")
    with st.spinner("Analizando distribución de variables (KS & PSI)..."):
        # Ejecutar análisis manual usando drift_utils
        drift_results = run_drift_analysis(reference_df, current_df)
    
    # Métricas de Alto Nivel
    n_total = len(drift_results)
    # Contamos cuántas filas tienen "SÍ" en la columna "Drift Detectado"
    n_drift = len(drift_results[drift_results["Drift Detectado"].str.contains("SÍ")])
    pct_drift = (n_drift / n_total) * 100
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Variables Analizadas", n_total)
    m2.metric("Variables con Drift", n_drift, delta_color="inverse")
    
    status_color = "green"
    status_msg = "ESTABLE"
    if pct_drift > 20:
        status_color = "red"
        status_msg = "CRÍTICO"
    elif pct_drift > 0:
        status_color = "orange"
        status_msg = "ADVERTENCIA"
        
    m3.markdown(f"Estado de Datos: **:{status_color}[{status_msg}]**")
    
    # Tabla Detallada
    with st.expander("Ver Detalle por Variable", expanded=(n_drift > 0)):
        # Colorear filas con drift
        st.dataframe(
            drift_results.style.applymap(
                lambda v: "background-color: #ffcdd2; color: black" if "SÍ" in str(v) else "",
                subset=["Drift Detectado"]
            ),
            use_container_width=True
        )

    # === 2. PREDICTION DRIFT (Concept Drift Proxy) ===
    st.markdown("---")
    st.header("2. Monitoreo de Concept Drift (Predicciones)")
    
    if model:
        with st.spinner("Generando predicciones y analizando comportamiento del modelo..."):
            try:
                # Asegurar que las columnas coincidan con lo que espera el modelo
                # (Filtrar y reordenar columnas)
                # Nota: Asumimos que current_df tiene las columnas necesarias.
                X_ref = reference_df[EXPECTED_RAW_COLS]
                X_cur = current_df[EXPECTED_RAW_COLS]
                
                # Generar probabilidades de la clase positiva (1 = Bueno)
                # El modelo es un Pipeline, maneja el preprocesamiento internamente
                ref_preds = model.predict_proba(X_ref)[:, 1]
                cur_preds = model.predict_proba(X_cur)[:, 1]
                
                # Convertir a Series para nuestras funciones de utilidad
                ref_series = pd.Series(ref_preds, name="Probabilidad (Referencia)")
                cur_series = pd.Series(cur_preds, name="Probabilidad (Actual)")
                
                # Calcular Drift usando nuestras funciones manuales (KS y PSI)
                # Usamos KS porque son probabilidades contínuas
                ks_stat, p_value = calculate_ks_test(ref_series, cur_series)
                psi = calculate_psi(ref_series, cur_series, bins=10)
                
                drift_detected_ks = p_value < 0.05
                drift_detected_psi = psi > 0.2
                
                # Mostrar Resultados
                c1, c2, c3 = st.columns(3)
                c1.metric("PSI de Predicciones", f"{psi:.4f}")
                c2.metric("KS P-Value", f"{p_value:.4f}")
                
                status_pred = "🚨 SÍ" if (drift_detected_ks or drift_detected_psi) else "✅ No"
                color_pred = "inverse" if "SÍ" in status_pred else "normal"
                c3.metric("¿Drift en Predicciones?", status_pred, delta_color=color_pred)
                
                # Gráfica de Distribución de Predicciones (Comparativa)
                st.subheader("Distribución de Probabilidades Predichas")
                
                # Crear DataFrame largo para Altair
                chart_data = pd.DataFrame({
                    "Referencia": ref_series,
                    "Actual": cur_series
                }).melt(var_name='Dataset', value_name='Probabilidad')
                
                chart = alt.Chart(chart_data).mark_area(
                    opacity=0.5,
                    interpolate='step'
                ).encode(
                    alt.X('Probabilidad:Q', bin=alt.Bin(maxbins=20), title="Probabilidad de Riesgo"),
                    alt.Y('count()', stack=None, title="Frecuencia"),
                    alt.Color('Dataset:N', scale=alt.Scale(scheme='category10'))
                ).properties(
                    title="Comparación de Distribuciones de Riesgo (Train vs. Prod)"
                )
                
                st.altair_chart(chart, use_container_width=True)

                # Diagnóstico Final
                if "SÍ" in status_pred:
                    st.error("⚠️ ¡ALERTA! El modelo está emitiendo predicciones muy diferentes a lo esperado. Posible Concept Drift.")
                else:
                    st.success("✅ El comportamiento del modelo es estable.")

            except Exception as e:
                st.error(f"Error al generar predicciones: {e}")
                st.warning("Verifica que las columnas de datos coincidan con las del entrenamiento.")
    else:
        st.warning("⚠️ Modelo no cargado. No se puede analizar Prediction Drift.")

st.markdown("---")
st.caption("Dashboard de Monitoreo MLOps | Equipo 46 MNA")