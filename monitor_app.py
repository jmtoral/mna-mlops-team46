# monitor_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import warnings

# --- Importar nuestro módulo de cálculo manual ---
from german_credit_ml.drift_utils import run_drift_analysis

# Ignorar advertencias
warnings.filterwarnings("ignore")

# Configuración de Streamlit
st.set_page_config(page_title="Dashboard Monitoreo MLOps", layout="wide")
st.title("📊 Dashboard de Monitoreo - Riesgo Crediticio")
st.markdown("Análisis de Deriva de Datos (Data Drift) calculado manualmente.")

# --- Carga de Datos ---
DATA_FILE = Path("data/processed/german_credit_clean.csv")

@st.cache_data
def load_data(file_path):
    if not file_path.exists():
        st.error(f"Archivo de datos no encontrado: {file_path}. Ejecuta `dvc pull`.")
        return None
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

df = load_data(DATA_FILE)

if df is not None:
    # --- Selección de Datos ---
    st.header("1. Configuración del Análisis")
    split_point = st.slider(
        "Selecciona el porcentaje de datos para 'Referencia':",
        min_value=10, max_value=90, value=70, step=5,
        help="El resto se usará como datos 'Actuales' para la comparación."
    )
    
    reference_data_end_index = int(len(df) * (split_point / 100))
    reference_df = df.iloc[:reference_data_end_index].copy()
    current_df = df.iloc[reference_data_end_index:].copy()

    st.info(f"Comparando {len(reference_df)} filas (Referencia) vs. {len(current_df)} filas (Actuales).")
    
    if st.button("Ejecutar Análisis de Deriva", type="primary"):
        with st.spinner("Calculando métricas KS y PSI..."):
            drift_results_df = run_drift_analysis(reference_df, current_df)
        
        st.success("¡Análisis completado!")
        st.markdown("---")
        
        # --- Resumen (Velocímetros) ---
        st.header("2. Resumen de Deriva")
        
        num_cols_total = len(drift_results_df["Columna"].unique())
        num_cols_drifted = len(drift_results_df[drift_results_df["Drift Detectado"] == "🚨 SÍ"])
        drift_percentage = (num_cols_drifted / num_cols_total) * 100
        
        delta_color = "normal"
        help_text = "Deriva baja. El modelo está estable."
        if drift_percentage > 25:
            delta_color = "inverse" # Rojo
            help_text = "Peligro. Se recomienda re-entrenamiento."
        elif drift_percentage > 10:
            delta_color = "inverse" # Naranja (pero `inverse` se ve rojo)
            help_text = "Advertencia. Monitorear de cerca."

        st.metric(
            label="Columnas con Deriva Detectada",
            value=f"{num_cols_drifted} / {num_cols_total}",
            delta=f"{drift_percentage:.1f}% del total",
            delta_color=delta_color,
            help=help_text
        )

        # --- Resultados Detallados ---
        st.header("3. Reporte Detallado por Columna")
        st.dataframe(drift_results_df, use_container_width=True)

else:
    st.warning("No se pudieron cargar los datos para el análisis.")

st.markdown("---")
st.caption("Dashboard de Monitoreo MLOps | Equipo 46 MNA")