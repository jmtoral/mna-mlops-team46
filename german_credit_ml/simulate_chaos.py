# simulate_chaos.py
import pandas as pd
import numpy as np
from pathlib import Path
from german_credit_ml.utils import console

# --- Configuración ---
CLEAN_DATA_PATH = Path("data/processed/german_credit_clean.csv")
# NUEVO ARCHIVO DE SALIDA
CHAOS_DATA_PATH = Path("data/processed/german_credit_chaos.csv")
SPLIT_POINT = 0.7 

console.print(f"[bold magenta]🔮 INICIANDO SIMULACIÓN DE CAOS EN PREDICCIONES 🔮[/bold magenta]")
console.rule("[bold magenta]GENERADOR DE RUIDO ADVERSARIO[/bold magenta]")

if not CLEAN_DATA_PATH.exists():
    console.print(f"[bold red]ERROR:[/bold red] Falta {CLEAN_DATA_PATH}.")
    exit()

df = pd.read_csv(CLEAN_DATA_PATH)
split_index = int(len(df) * SPLIT_POINT)
df_cur = df.iloc[split_index:].copy()

console.print(f"[INFO] Generando caos sobre {len(df_cur)} filas (Datos Actuales)...")

# ESTRATEGIA DE CAOS: Romper las correlaciones que el modelo aprendió
# El modelo XGBoost depende mucho de 'status', 'duration' y 'history'.
# Vamos a aleatorizarlas completamente para que las predicciones sean un disparate.

# 1. Ruido Total en 'status' (La variable más importante)
# Al asignar valores aleatorios uniformes, rompemos la capacidad predictiva principal.
df_cur['status'] = np.random.randint(1, 5, size=len(df_cur))
console.print("  🎲 [bold magenta]Aleatorización:[/bold magenta] 'status' reasignado con ruido uniforme (1-4).")

# 2. Inversión de 'credit_history'
# Hacemos que los buenos parezcan malos y viceversa.
# Mapeo: 0->4, 1->3, 2->2, 3->1, 4->0
console.print("  🔄 [bold magenta]Inversión:[/bold magenta] 'credit_history' invertido (buenos <-> malos).")
df_cur['credit_history'] = df_cur['credit_history'].map({0:4, 1:3, 2:2, 3:1, 4:0})

# 3. Extremos en 'duration'
# Forzamos a que todos los créditos sean o muy cortos o muy largos, sin puntos medios.
# Esto polarizará las predicciones.
extreme_duration = np.random.choice([4, 72], size=len(df_cur))
df_cur['duration'] = extreme_duration
console.print("  ⚡ [bold magenta]Polarización:[/bold magenta] 'duration' forzada a extremos (4 o 72 meses).")

# 4. Montos Aleatorios ('amount')
# Desvinculamos el monto de la realidad del cliente.
df_cur['amount'] = np.random.randint(250, 20000, size=len(df_cur))
console.print("  💸 [bold magenta]Desvinculación:[/bold magenta] 'amount' generado aleatoriamente.")

# Guardar
df_cur.to_csv(CHAOS_DATA_PATH, index=False)

console.print(f"\n[bold bright_green][SUCCESS][/bold bright_green] Dataset de CAOS guardado en: [cyan]{CHAOS_DATA_PATH}[/cyan]")