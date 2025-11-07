# german_credit_ml/drift_utils.py
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Tuple, Dict, List

def calculate_psi(base_series: pd.Series, current_series: pd.Series, bins: int = 10) -> float:
    """Calcula el Índice de Estabilidad de Población (PSI) para una variable."""
    
    # Asegurarse de que ambas series sean tratadas como categóricas si no son numéricas
    if not pd.api.types.is_numeric_dtype(base_series):
        base_series = base_series.astype('category')
        current_series = current_series.astype('category')

    # Si es numérica, discretizar en 'bins' (contenedores)
    if pd.api.types.is_numeric_dtype(base_series):
        # Usar los 'bins' de la serie base para ambas
        _, bin_edges = pd.qcut(base_series, bins, retbins=True, duplicates='drop')
        # Asegurar que los bordes sean únicos
        bin_edges = np.unique(bin_edges)
        
        base_binned = pd.cut(base_series, bins=bin_edges, include_lowest=True)
        current_binned = pd.cut(current_series, bins=bin_edges, include_lowest=True)
        
        base_dist = base_binned.value_counts(normalize=True, dropna=False)
        current_dist = current_binned.value_counts(normalize=True, dropna=False)
    else:
        # Si es categórica, solo contar valores
        base_dist = base_series.value_counts(normalize=True, dropna=False)
        current_dist = current_series.value_counts(normalize=True, dropna=False)

    # Alinear los índices (categorías/bins) para la resta
    all_categories = set(base_dist.index) | set(current_dist.index)
    base_dist = base_dist.reindex(all_categories, fill_value=0)
    current_dist = current_dist.reindex(all_categories, fill_value=0)

    # Calcular PSI (reemplazando 0s con un valor muy pequeño para evitar división por cero)
    base_dist = base_dist.replace(0, 0.0001)
    current_dist = current_dist.replace(0, 0.0001)

    psi_values = (current_dist - base_dist) * np.log(current_dist / base_dist)
    psi = np.sum(psi_values)
    
    return psi

def calculate_ks_test(base_series: pd.Series, current_series: pd.Series) -> Tuple[float, float]:
    """Ejecuta el Test KS de 2 muestras."""
    if not pd.api.types.is_numeric_dtype(base_series):
        return 0.0, 1.0 # No aplicable a categóricas
        
    ks_stat, p_value = ks_2samp(base_series.dropna(), current_series.dropna())
    return ks_stat, p_value

def run_drift_analysis(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta el análisis de deriva en todas las columnas."""
    
    results = []
    target_col = 'credit_risk' # Asumimos que esta es tu columna objetivo
    
    for col in reference_df.columns:
        if col == target_col:
            continue
            
        base_series = reference_df[col]
        current_series = current_df[col]
        
        if pd.api.types.is_numeric_dtype(base_series):
            # Para numéricas, usamos KS
            ks_stat, p_value = calculate_ks_test(base_series, current_series)
            drift_detected = p_value < 0.05 # Hipótesis nula: son de la misma distribución
            results.append({
                "Columna": col,
                "Tipo": "Numérica",
                "Métrica": "KS Test (p-value)",
                "Valor": f"{p_value:.4f}",
                "Drift Detectado": "🚨 SÍ" if drift_detected else "✅ No"
            })
        else:
            # Para categóricas, usamos PSI
            psi = calculate_psi(base_series, current_series)
            # Reglas comunes de PSI: < 0.1 (sin drift), 0.1-0.2 (advertencia), > 0.2 (drift)
            drift_detected = psi > 0.2
            results.append({
                "Columna": col,
                "Tipo": "Categórica",
                "Métrica": "PSI",
                "Valor": f"{psi:.4f}",
                "Drift Detectado": "🚨 SÍ" if drift_detected else "✅ No"
            })
            
    return pd.DataFrame(results)