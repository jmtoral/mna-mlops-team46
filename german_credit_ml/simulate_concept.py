# simulate_concept.py
import pandas as pd
import numpy as np
from pathlib import Path
from german_credit_ml.utils import console

# --- Configuración ---
CLEAN_DATA_PATH = Path("data/processed/german_credit_clean.csv")
# NUEVO ARCHIVO
CONCEPT_DATA_PATH = Path("data/processed/german_credit_concept.csv")
SPLIT_POINT = 0.7

console.print(f"[bold magenta]🧠 INICIANDO SIMULACIÓN DE CONCEPT DRIFT (Multivariado) 🧠[/bold magenta]")
console.rule("[bold magenta]PRESERVANDO MARGINALES, ROMPIENDO CORRELACIONES[/bold magenta]")

if not CLEAN_DATA_PATH.exists():
    console.print(f"[bold red]ERROR:[/bold red] Falta {CLEAN_DATA_PATH}")
    exit()

df = pd.read_csv(CLEAN_DATA_PATH)
split_idx = int(len(df) * SPLIT_POINT)
df_cur = df.iloc[split_idx:].copy()

console.print(f"[INFO] Modificando {len(df_cur)} filas. Se mantendrán los histogramas exactos.")

# ESTRATEGIA: Barajar (Shuffle) las columnas clave independientemente.
# Esto mantiene la distribución univariada intacta (Data Drift = 0)
# Pero destruye la relación entre variables, confundiendo al modelo (Prediction Drift > 0)

cols_to_shuffle = ['status', 'duration', 'amount', 'age', 'credit_history', 'savings']

for col in cols_to_shuffle:
    # Permutación aleatoria de los valores de la columna
    # (Los valores son los mismos, solo cambian de fila/cliente)
    shuffled_values = np.random.permutation(df_cur[col].values)
    df_cur[col] = shuffled_values
    console.print(f"  🔀 [cyan]{col}[/cyan]: Valores barajados (Distribución intacta, relación rota).")

# Guardar
df_cur.to_csv(CONCEPT_DATA_PATH, index=False)
console.print(f"\n[bold bright_green][SUCCESS][/bold bright_green] Dataset generado: [cyan]{CONCEPT_DATA_PATH}[/cyan]")
console.print("[INFO] Data Drift debería ser [green]VERDE[/green]. Prediction Drift debería ser [red]ROJO[/red].")