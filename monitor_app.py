# monitor_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import datetime
import warnings

# --- Corrected Evidently Import ---
from evidently.pipeline.column_mapping import ColumnMapping # <-- FIX HERE
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Ignorar advertencias futuras para una salida más limpia
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning) # A veces Evidently da UserWarnings

# Configuración de Streamlit
st.set_page_config(page_title="Dashboard Monitoreo MLOps", layout="wide")
st.title("📊 Dashboard de Monitoreo - Riesgo Crediticio")

# --- Carga de Datos ---
# Define la ruta al archivo de datos procesados
DATA_FILE = Path("data/processed/german_credit_clean.csv")

@st.cache_data # Cache para evitar recargar datos en cada interacción
def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    if not file_path.exists():
        st.error(f"Archivo de datos no encontrado: {file_path}. Ejecuta `dvc pull`.")
        return None
    try:
        df = pd.read_csv(file_path)
        # Convertir columnas categóricas si Evidently las necesita así
        # (Ajustar según los tipos de datos reales después de clean.py)
        # for col in df.select_dtypes(include='category').columns:
        #     df[col] = df[col].astype(str)
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

df = load_data(DATA_FILE)

if df is not None:
    st.header("Análisis de Deriva de Datos (Data Drift)")

    # --- Selección de Datos de Referencia y Actuales ---
    st.markdown("Selecciona los rangos para los datos de referencia y actuales.")

    # Slider para seleccionar el punto de división
    split_point = st.slider(
        "Porcentaje del dataset para datos de referencia:",
        min_value=10, max_value=90, value=70, step=5,
        help="El resto del dataset se usará como datos 'actuales' para la comparación."
    )
    reference_data_end_index = int(len(df) * (split_point / 100))

    reference_df = df[:reference_data_end_index].copy() # Usar .copy() para evitar warnings
    current_df = df[reference_data_end_index:].copy()

    st.write(f"Comparando datos de referencia ({len(reference_df)} filas) vs. datos actuales ({len(current_df)} filas).")

    # --- Configuración y Ejecución del Reporte Evidently ---
    # Mapeo de columnas (Opcional pero recomendado para mayor precisión)
    # Identificar columnas automáticamente o definirlas manualmente si es necesario
    column_mapping = ColumnMapping()
    # Ejemplo: Si tu target y prediction se llaman diferente, o quieres especificar tipos
    # target_col = 'credit_risk' # Asegúrate que este sea el nombre correcto
    # num_features = df.select_dtypes(include=np.number).columns.drop(target_col, errors='ignore').tolist()
    # cat_features = df.select_dtypes(exclude=np.number).columns.tolist()
    # column_mapping.target = target_col
    # column_mapping.numerical_features = num_features
    # column_mapping.categorical_features = cat_features

    # Crear el reporte con el preset de Data Drift
    drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    # Ejecutar el reporte comparando los dos DataFrames
    with st.spinner("Calculando métricas de deriva de datos... Este proceso puede tardar."):
        drift_report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

    # --- Visualización del Reporte ---
    st.subheader("Reporte de Deriva de Datos (Generado por Evidently AI)")

    # Guardar el reporte como HTML temporalmente
    report_path = Path(f"temp_drift_report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.html")
    drift_report.save_html(str(report_path))

    # Leer el HTML y mostrarlo en Streamlit usando el componente `html`
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        # El componente html permite mostrar contenido HTML. Ajusta height según necesites.
        st.components.v1.html(html_content, height=800, scrolling=True)
    except Exception as e:
        st.error(f"Error al mostrar el reporte HTML: {e}")
    finally:
        # Limpiar el archivo temporal después de mostrarlo
        if report_path.exists():
            try:
                report_path.unlink()
            except OSError as e:
                st.warning(f"No se pudo eliminar el archivo temporal {report_path}: {e}")

    # (Opcional) Extraer y mostrar métricas clave directamente desde el reporte:
    try:
        report_dict = drift_report.as_dict()
        drift_details = report_dict['metrics'][0]['result']
        num_columns = drift_details['number_of_columns']
        num_drifted_columns = drift_details['number_of_drifted_columns']
        share_drifted_columns = drift_details['share_of_drifted_columns']

        st.subheader("Resumen de Deriva")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Columnas Analizadas", num_columns)
        col2.metric("Columnas con Drift Detectado", num_drifted_columns)
        col3.metric("Porcentaje de Columnas con Drift", f"{share_drifted_columns*100:.2f}%")
    except (KeyError, IndexError, TypeError) as e:
        st.warning(f"No se pudo extraer el resumen de métricas del reporte: {e}")


else:
    st.warning("No se pudieron cargar los datos para el análisis.")

st.markdown("---")
st.caption("Dashboard de Monitoreo MLOps | Equipo 46 MNA")