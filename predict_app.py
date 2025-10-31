import datetime
import json
from pathlib import Path
import pickle
import warnings

import mlflow
import pandas as pd
from PIL import Image
import streamlit as st

warnings.filterwarnings("ignore")

# --- Konfiguration MUSS ZUERST gehen ---
st.set_page_config(
    page_title="Análisis Riesgo Crediticio",
    page_icon="🥨",
    layout="wide",  # Correcto: layout ancho, centrado con CSS
    initial_sidebar_state="expanded",
)

# =============================
#  CONSTANTES
# =============================
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
METADATA_PATH = MODEL_DIR / "xgboost_model_metadata.json"

# Columnas esperadas por el modelo en el orden correcto
EXPECTED_RAW_COLS = [
    "status",
    "duration",
    "credit_history",
    "purpose",
    "amount",
    "savings",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
]

STATUS_OPTS = {
    1: "< 0 DM (sobregiro)",
    2: "0 - 200 DM",
    3: "≥ 200 DM",
    4: "Sin cuenta (Kein Konto)",
}

HISTORY_OPTS = {
    0: "Sin créditos previos",
    1: "Todo pagado puntualmente",
    2: "Créditos actuales al día",
    3: "Retrasos en el pasado",
    4: "Cuenta crítica (¡Sehr schlecht!)",
}

# Valores por defecto para los campos que NO están en el formulario
DEFAULT_FORM_VALUES = {
    "employment_duration": 3.0,
    "installment_rate": 4.0,
    "personal_status_sex": 3.0,  # Asumiendo un valor (p.ej. 'male divorced/separated')
    "other_debtors": 1.0,
    "present_residence": 4.0,
    "property": 3.0,
    "other_installment_plans": 3.0,
    "housing": 2.0,
    "number_credits": 1.0,
    "job": 3.0,
    "people_liable": 1.0,
    "telephone": 1.0,
    "foreign_worker": 2.0,
}


# =============================
#  CSS / THEME
# =============================
def load_custom_css():
    # Tema oscuro forzado + barra superior negra
    bg_color = "#0E1117"
    surface = "#11151D"
    card_bg = "#1B212C"
    text = "#F6F7F9"
    subtext = "#AAB2C3"
    accent = "#FF6B6B"
    success = "#51CF66"
    border = "#263043"
    shadow = "0 6px 18px rgba(0,0,0,.35)"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {bg_color} !important;
    }}

    /* Barra negra superior (header) */
    [data-testid="stHeader"] {{
        background-color: #000 !important;
        box-shadow: 0 1px 0 rgba(255,255,255,0.06) inset, 0 2px 8px rgba(0,0,0,.35) !important;
    }}

    /* Contenedor principal centrado */
    .block-container {{
        padding: 2rem 2rem 3rem !important;
        max-width: 1200px !important;  /* Límite de ancho */
        margin: 0 auto !important;      /* Centrado horizontal */
        background-color: {surface} !important;
        border-radius: 20px !important;
        box-shadow: {shadow} !important;
        border: 1px solid {border} !important;
    }}

    h1, h2, h3 {{
        font-family: 'Poppins', sans-serif !important;
        color: {text} !important;
        text-align: center; /* Centrado por defecto */
        font-weight: 700 !important;
        margin-top: .25rem !important;
    }}
    h1 {{
        font-size: 2.4rem !important;
        line-height: 1.15 !important;
        background: linear-gradient(135deg, {accent}, {success});
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: .25rem !important;
    }}
    h2 {{ font-size: 1.6rem !important; }}

    p, .stMarkdown, label, .stCaption, .stText {{
        color: {text} !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.6 !important;
    }}

    .stForm, .streamlit-expanderContent {{
        background: {card_bg} !important;
        border-radius: 16px !important;
        box-shadow: {shadow} !important;
        border: 1px solid {border} !important;
    }}

    .stSelectbox, .stNumberInput, .stSlider {{ border-radius: 10px !important; }}

    .stButton>button {{
        background: linear-gradient(135deg, {accent}, #C0392B) !important;
        color: white !important; font-weight: 600 !important;
        font-size: 1.05rem !important; padding: .7rem 1.5rem !important;
        border-radius: 12px !important; border: none !important;
        box-shadow: 0 6px 18px rgba(231,76,60,.28) !important; transition: .25s ease !important;
        font-family: 'Poppins', sans-serif !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: {card_bg} !important;
        border-right: 1px solid {border} !important;
    }}

    [data-testid="stMetricValue"] {{ color: {accent} !important; font-weight: 700 !important; }}

    hr {{ height: 2px !important; border: 0 !important; margin: 1.75rem 0 !important;
          background: linear-gradient(90deg, transparent, {accent}, transparent) !important; }}

    img {{ border-radius: 14px !important; box-shadow: {shadow} !important; max-width: 100% !important; height: auto !important; }}

    .stAlert {{ border-radius: 12px !important; border-left: 4px solid {accent} !important; }}

    .theme-note {{ color: {subtext}; font-size: .85rem; text-align: center; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =============================
#  Funciones de Carga (Cacheadas)
# =============================
@st.cache_resource
def load_model_and_metadata():
    model, metadata = None, None
    model_status = "❌ Modelo NO Cargado"
    metadata_status = "❌ Metadatos NO Cargados"

    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            model_status = f"✅ Modelo Cargado ({size_mb:.2f} MB)"
        except Exception as e:
            model_status = f"⚠️ Error al cargar modelo: {e}"
    else:
        model_status = (
            f"❌ Archivo de modelo ({MODEL_PATH}) no encontrado. Ejecuta `dvc pull`."
        )

    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
            metadata_status = "✅ Metadatos Cargados"
        except Exception as e:
            metadata_status = f"⚠️ Error al cargar metadatos: {e}"
    else:
        metadata_status = f"❌ Archivo de metadatos ({METADATA_PATH}) no encontrado."

    return model, metadata, model_status, metadata_status


@st.cache_data
def get_mlflow_run_data(run_id):
    if not run_id:
        return None, "No se proporcionó Run ID."
    try:
        run = mlflow.get_run(run_id)
        return run, f"✅ Datos de MLflow Run '{run_id[:8]}...' obtenidos."
    except Exception as e:
        st.warning(f"No se pudo conectar a MLflow para Run ID: {run_id}. {e}")
        return None, f"⚠️ Error al obtener datos de MLflow: {e}."


# =============================
#  Función Principal de la App
# =============================
def main():
    # --- Cargar CSS y Tema ---
    st.session_state["theme"] = "dark"
    load_custom_css()

    # --- Inicializar Estado ---
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    # --- Carga de Modelo y Metadatos ---
    model, metadata, model_status, metadata_status = load_model_and_metadata()

    mlflow_run, mlflow_status = None, "⏳ Buscando datos de MLflow..."
    if metadata and "mlflow_run_id" in metadata:
        mlflow_run, mlflow_status = get_mlflow_run_data(metadata["mlflow_run_id"])
    else:
        mlflow_status = "❌ No se encontró Run ID en metadatos."

    # =============================
    #  Sidebar (Panel de Control)
    # =============================
    with st.sidebar:
        st.header("📊 Panel de Kontrol")
        st.caption("Tema: oscuro (forzado)")

        if model:
            st.success(model_status)
        else:
            st.error(model_status)
        st.caption(metadata_status)
        st.caption(mlflow_status)

        st.markdown("---")
        st.subheader("ℹ️ Información del Modelo")
        if mlflow_run:
            try:
                start_time_ms = mlflow_run.info.start_time
                start_time_dt = datetime.datetime.fromtimestamp(start_time_ms / 1000)
                st.metric(
                    label="Fecha de Entrenamiento",
                    value=start_time_dt.strftime("%Y-%m-%d %H:%M"),
                )
            except Exception:
                st.caption("No se pudo obtener la fecha.")

            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    label="F1 Score",
                    value=f"{mlflow_run.data.metrics.get('f1_score_test', 0):.4f}",
                )
                st.metric(
                    label="AUC",
                    value=f"{mlflow_run.data.metrics.get('auc_test', 0):.4f}",
                )
            with c2:
                st.metric(
                    label="Accuracy",
                    value=f"{mlflow_run.data.metrics.get('accuracy_test', 0):.4f}",
                )
                st.metric(
                    label="Bad Rate",
                    value=f"{mlflow_run.data.metrics.get('bad_rate_test', 0):.4f}",
                )

            with st.expander("🔧 Ver Hiperparámetros"):
                st.write(
                    f"**n_estimators:** {mlflow_run.data.params.get('n_estimators', 'N/A')}"
                )
                st.write(
                    f"**max_depth:** {mlflow_run.data.params.get('max_depth', 'N/A')}"
                )
                st.write(
                    f"**learning_rate:** {mlflow_run.data.params.get('learning_rate', 'N/A')}"
                )
        else:
            st.warning("No se pudo cargar la información de MLflow.")

        st.markdown("---")
        st.caption("📚 Versiones de Librerías:")
        try:
            import xgboost

            st.caption(f"XGBoost: {xgboost.__version__}")
        except ImportError:
            st.caption("XGBoost: N/A")
        st.caption(f"Pandas: {pd.__version__}")
        st.caption(f"Streamlit: {st.__version__}")

    # =============================
    #  Contenido Principal
    # =============================
    st.title("🥨 Análisis de Riesgo Crediticio")
    st.markdown(
        "<p style='text-align:center;font-size:1.1rem;color:#AAB2C3;margin:-.25rem 0 1.25rem'>Intelligentes Finanzbewertungssystem</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # --- Imagen Centrada ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            image = Image.open("references/german_family.jpg")
            # <--- CAMBIO: use_column_width -> use_container_width ---
            st.image(image, use_container_width=True)
        except FileNotFoundError:
            st.info("💡 Imagen no encontrada: references/german_family.jpg")

    # --- Mensaje de Bienvenida ---
    st.markdown(
        "<p style='text-align:center;font-size:1.3rem;margin-top:1rem'> <strong>Willkommen!</strong> Introduzca los datos para evaluar la solicitud.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # --- Formulario de Entrada ---
    # Define los widgets del formulario. Las variables (status, duration, etc.)
    # están disponibles en el scope de main()
    with st.form("kredit_form"):
        st.subheader("📋 Información del Solicitante")

        c1, c2 = st.columns(2)
        with c1:
            status = st.selectbox(
                "🏦 Estado Cuenta Corriente",
                options=list(STATUS_OPTS.keys()),
                format_func=lambda x: STATUS_OPTS[x],
            )
            duration = st.slider("📅 Duración Crédito (meses)", 4, 72, 24)
            amount = st.number_input(
                "💰 Monto (DM)", min_value=250, max_value=20000, value=2500, step=250
            )
        with c2:
            credit_history = st.selectbox(
                "📊 Historial Crediticio",
                options=list(HISTORY_OPTS.keys()),
                format_func=lambda x: HISTORY_OPTS[x],
            )
            age = st.slider("👤 Edad", 18, 75, 35)

        c3, c4 = st.columns(2)
        with c3:
            purpose_opts = {
                0: "new car",
                1: "used car",
                2: "furniture/equipment",
                3: "radio/TV",
                4: "domestic appliance",
                5: "repairs",
                6: "education",
                7: "vacation",
                8: "retraining",
                9: "business",
                10: "others",
            }
            purpose = st.selectbox(
                "🎯 Propósito",
                options=list(purpose_opts.keys()),
                format_func=lambda x: purpose_opts[x],
                index=3,
            )
        with c4:
            savings_opts = {
                1: "<100",
                2: "100-500",
                3: "500-1000",
                4: ">=1000",
                5: "unknown",
            }
            savings = st.selectbox(
                "💰 Ahorros",
                options=list(savings_opts.keys()),
                format_func=lambda x: savings_opts[x],
                index=0,
            )

        # Botón de envío
        submitted = st.form_submit_button(
            "🔍 ¡Evaluar Solicitud!", use_container_width=True
        )

    # --- Lógica de Estado ---
    # Si se hace clic en el botón, activa el estado de sesión
    if submitted:
        st.session_state.submitted = True

    # =============================
    #  Lógica de Predicción (Solo si el formulario se envió)
    # =============================

    # Este bloque se ejecuta SIEMPRE que el botón se haya presionado al menos una vez
    if st.session_state.submitted:
        # <--- CAMBIO CLAVE (FIX UnboundLocalError): Creación de datos movida AQUÍ ---
        # 1. Recoge los inputs del usuario (usando las variables del form de arriba)
        user_inputs = {
            "status": float(status),
            "duration": float(duration),
            "credit_history": float(credit_history),
            "amount": float(amount),
            "age": float(age),
            "purpose": float(purpose),
            "savings": float(savings),
        }

        # 2. Combina los defaults con los inputs del usuario
        input_data = {**DEFAULT_FORM_VALUES, **user_inputs}
        # --- FIN DEL CAMBIO ---

        st.markdown("---")
        with st.expander("📄 Ver Datos Introducidos (Completos)"):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**🏦 Estado Cuenta:** {STATUS_OPTS.get(status, 'N/A')}")
                st.write(f"**📅 Duración:** {duration} meses")
                st.write(f"**💰 Monto:** {amount:,} DM")
            with c2:
                st.write(f"**📊 Historial:** {HISTORY_OPTS.get(credit_history, 'N/A')}")
                st.write(f"**👤 Edad:** {age} años")

            st.caption(
                "Valores por defecto asumidos (no solicitados en el formulario):"
            )
            st.json({k: v for k, v in input_data.items() if k not in user_inputs})

        prediction = None
        if model:
            with st.spinner("🔄 Analizando solicitud..."):
                try:
                    df_input = pd.DataFrame([input_data]).reindex(
                        columns=EXPECTED_RAW_COLS, fill_value=0
                    )

                    prediction = int(model.predict(df_input)[0])
                    st.success("✅ ¡Análisis completado!")

                except Exception as e:
                    st.error(f"⚠️ ¡Achtung! Error durante la predicción: {e}")
                    st.dataframe(df_input)
                    prediction = None
        else:
            st.error("❌ El modelo no está cargado. No se puede predecir.")

        st.markdown("---")
        st.subheader("🎯 Veredicto Final")

        if prediction == 1:
            st.success("### ✅ ¡APROBADO! Cliente con bajo riesgo crediticio.")
            st.balloons()
            st.markdown(
                "<p style='text-align:center;font-size:1.05rem;color:#51CF66'>🎉 <strong>Ausgezeichnet!</strong> La solicitud cumple con los criterios.</p>",
                unsafe_allow_html=True,
            )
        elif prediction == 0:
            st.error("### 🛑 ¡RECHAZADO! Cliente con alto riesgo crediticio.")
            st.markdown(
                "<p style='text-align:center;font-size:1.05rem;color:#FF6B6B'>⚠️ Se recomienda revisar los parámetros de la solicitud.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("No se pudo obtener una predicción debido a un error previo.")

    # =============================
    #  Footer
    # =============================
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#AAB2C3;font-size:.9rem;margin-top:1rem'>🏛️ Sistema de Evaluación de Riesgo | Equipo 46 MNA | Powered by XGBoost & MLflow</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
