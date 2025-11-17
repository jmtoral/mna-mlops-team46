# german_credit_ml/drift_utils.py
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Tuple, Dict, List
from german_credit_ml.utils import console

def calculate_psi(base_series: pd.Series, current_series: pd.Series, bins: int = 10) -> float:
    """
    Calcula el Índice de Estabilidad de Población (PSI) para una variable.
    Maneja automáticamente variables numéricas (binning) y categóricas.
    """
    
    # 1. Manejo de Variables Numéricas (Binning)
    if pd.api.types.is_numeric_dtype(base_series) and len(base_series.unique()) > 20:
        try:
            # Crear bins basados en la distribución de referencia
            _, bin_edges = pd.qcut(base_series, q=bins, retbins=True, duplicates='drop')
            # Asegurar bordes únicos y extender límites para cubrir nuevos datos
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2: # Si no hay suficientes bins, tratar como categórica
                 pass 
            else:
                # Extender rango para evitar errores con valores fuera de rango en current
                bin_edges[0] = min(bin_edges[0], current_series.min())
                bin_edges[-1] = max(bin_edges[-1], current_series.max())
                
                base_binned = pd.cut(base_series, bins=bin_edges, include_lowest=True)
                current_binned = pd.cut(current_series, bins=bin_edges, include_lowest=True)
                
                base_dist = base_binned.value_counts(normalize=True, dropna=False)
                current_dist = current_binned.value_counts(normalize=True, dropna=False)
                
                return _compute_psi_from_dist(base_dist, current_dist)
        except Exception:
             # Si falla el binning numérico, intentar como categórico
             pass

    # 2. Manejo de Variables Categóricas (o Numéricas con pocos valores únicos)
    # Convertir a string para asegurar comparación correcta de categorías
    base_dist = base_series.astype(str).value_counts(normalize=True, dropna=False)
    current_dist = current_series.astype(str).value_counts(normalize=True, dropna=False)
    
    return _compute_psi_from_dist(base_dist, current_dist)

def _compute_psi_from_dist(base_dist: pd.Series, current_dist: pd.Series) -> float:
    """Función auxiliar matemática para PSI."""
    # Alinear índices (categorías/bins) para que coincidan
    all_categories = set(base_dist.index) | set(current_dist.index)
    base_dist = base_dist.reindex(all_categories, fill_value=0)
    current_dist = current_dist.reindex(all_categories, fill_value=0)

    # Reemplazar 0 con un valor epsilon muy pequeño para evitar división por cero/log(0)
    epsilon = 0.0001
    base_dist = base_dist.replace(0, epsilon)
    current_dist = current_dist.replace(0, epsilon)

    # Fórmula PSI
    psi_values = (current_dist - base_dist) * np.log(current_dist / base_dist)
    psi = np.sum(psi_values)
    return psi

def calculate_ks_test(base_series: pd.Series, current_series: pd.Series) -> Tuple[float, float]:
    """Ejecuta el Test KS de 2 muestras (Solo válido para numéricas continuas)."""
    if not pd.api.types.is_numeric_dtype(base_series):
        return 0.0, 1.0 # P-value 1.0 significa "no hay diferencia" (no aplicable)
        
    # El test KS no funciona bien con NaNs
    ks_stat, p_value = ks_2samp(base_series.dropna(), current_series.dropna())
    return ks_stat, p_value

def run_drift_analysis(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta el análisis de deriva en todas las columnas."""
    
    results = []
    target_col = 'credit_risk' 
    
    for col in reference_df.columns:
        if col == target_col:
            continue
            
        base_series = reference_df[col]
        current_series = current_df[col]
        
        drift_detected = False
        metric_name = ""
        metric_val = 0.0
        
        # Decidir qué test usar
        # Usamos KS para numéricas con muchos valores únicos
        if pd.api.types.is_numeric_dtype(base_series) and len(base_series.unique()) > 20:
            ks_stat, p_value = calculate_ks_test(base_series, current_series)
            metric_name = "KS Test (p-value)"
            metric_val = p_value
            # Si p < 0.05 rechazamos H0 (son diferentes) -> Drift
            drift_detected = p_value < 0.05
            tipo_real = "Numérica (Contínua)"
        else:
            # Para todo lo demás (categóricas o numéricas discretas), usamos PSI
            psi = calculate_psi(base_series, current_series)
            metric_name = "PSI"
            metric_val = psi
            # Regla general: PSI > 0.2 indica cambio significativo
            drift_detected = psi > 0.2
            tipo_real = "Categórica/Discreta"
            
        results.append({
            "Columna": col,
            "Tipo": tipo_real,
            "Métrica": metric_name,
            "Valor": f"{metric_val:.4f}",
            "Drift Detectado": "🚨 SÍ" if drift_detected else "✅ No"
        })
            
    return pd.DataFrame(results)

# ... (importaciones y funciones calculate_psi, calculate_ks_test existentes) ...

def run_prediction_drift_analysis(reference_predictions: pd.Series, current_predictions: pd.Series) -> Dict:
    """
    Analiza si la distribución de las predicciones ha cambiado.
    Usa PSI para comparar las probabilidades o clases predichas.
    """
    # Calcular PSI sobre las predicciones
    psi = calculate_psi(reference_predictions, current_predictions, bins=10)
    
    # Calcular KS sobre las predicciones (si son probabilidades continuas)
    ks_stat, p_value = calculate_ks_test(reference_predictions, current_predictions)
    
    drift_detected_psi = psi > 0.2
    drift_detected_ks = p_value < 0.05
    
    return {
        "Metric": "Prediction Drift",
        "PSI": psi,
        "KS P-Value": p_value,
        "Drift Detected (PSI)": "🚨 SÍ" if drift_detected_psi else "✅ No",
        "Drift Detected (KS)": "🚨 SÍ" if drift_detected_ks else "✅ No"
    }
    
# --- Bloque de ejecución para pruebas manuales ---
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    data_path = Path("data/processed/german_credit_clean.csv")
    
    if not data_path.exists():
        console.print(f"[bold red]Error:[/bold red] No se encontró {data_path}")
        sys.exit(1)
        
    console.print("[bold blue]TEST:[/bold blue] Cargando datos para prueba de drift...")
    df = pd.read_csv(data_path)
    
    console.print("[INFO] Dividiendo dataset 50/50...")
    ref = df.iloc[:len(df)//2]
    cur = df.iloc[len(df)//2:]
    
    console.print("[INFO] Ejecutando análisis...")
    drift_results = run_drift_analysis(ref, cur)
    
    console.print("\n[bold green]RESULTADOS DETECTADOS:[/bold green]")
    print(drift_results)