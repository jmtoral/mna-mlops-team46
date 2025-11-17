# simulate_drift.py
import pandas as pd
import numpy as np
from pathlib import Path
from german_credit_ml.utils import console

# --- Configuración de la Simulación (MODO APOCALIPSIS) ---
CLEAN_DATA_PATH = Path("data/processed/german_credit_clean.csv")
DRIFTED_DATA_PATH = Path("data/processed/german_credit_drifted.csv")

SPLIT_POINT = 0.7 

# Factores de Intensidad
AMOUNT_MULTIPLIER_MIN = 5.0   
AMOUNT_MULTIPLIER_MAX = 10.0  
AGE_SHIFT = 15                

console.print(f"[bold red]⚠️ INICIANDO SIMULACIÓN DE ESCENARIO: COLAPSO ECONÓMICO TOTAL ⚠️[/bold red]")
console.rule("[bold red]ALERTA DE DRIFT[/bold red]")

# 1. Cargar datos limpios
if not CLEAN_DATA_PATH.exists():
    console.print(f"[bold red]ERROR:[/bold red] No se encontró el archivo [cyan]{CLEAN_DATA_PATH}[/cyan].")
    exit()
    
df = pd.read_csv(CLEAN_DATA_PATH)
console.print(f"[INFO] Datos base cargados ({len(df)} filas).")

# 2. Dividir en referencia y actual
split_index = int(len(df) * SPLIT_POINT)
df_ref = df.iloc[:split_index].copy()
df_cur = df.iloc[split_index:].copy()
console.print(f"[INFO] Referencia: {len(df_ref)} filas | Actuales (a destruir): {len(df_cur)} filas.")

# 3. Simular Devaluación, Desempleo y Pérdidas
console.print("\n[bold yellow]>>> Aplicando alteraciones extremas...[/bold yellow]")

# a) Hiperinflación en 'amount'
multipliers = np.random.uniform(AMOUNT_MULTIPLIER_MIN, AMOUNT_MULTIPLIER_MAX, size=len(df_cur))
df_cur['amount'] = (df_cur['amount'] * multipliers).round().astype(int)
console.print(f"  🔥 [bold red]Hiperinflación:[/bold red] 'amount' multiplicado por {AMOUNT_MULTIPLIER_MIN}x - {AMOUNT_MULTIPLIER_MAX}x")

# b) Colapso del Ahorro ('savings')
# 1 (<100) se mantiene, ricos (3, 4) bajan a 1 o 2
df_cur['savings'] = df_cur['savings'].replace({3: 1, 4: 1, 5: 1}) 
console.print("  📉 [bold red]Pánico Bancario:[/bold red] Ahorros masivos eliminados ('savings' -> 1).")

# c) Deterioro del Estado de Cuenta ('status')
# 50% forzado a deuda/sobregiro (1)
mask_crisis = np.random.rand(len(df_cur)) < 0.5
df_cur.loc[mask_crisis, 'status'] = 1
console.print("  💸 [bold red]Crisis de Liquidez:[/bold red] 50% de cuentas en números rojos ('status' = 1).")

# d) Cambio Demográfico ('age')
df_cur['age'] = df_cur['age'] + AGE_SHIFT
console.print(f"  👴 [bold red]Envejecimiento:[/bold red] Edad promedio desplazada +{AGE_SHIFT} años.")

# --- NUEVAS ALTERACIONES ---

# e) Desempleo Masivo ('employment_duration')
# En German Credit: 1 = Desempleado, 2 = <1 año, 3 = 1-4 años, 4 = 4-7 años, 5 = >=7 años
# Simulamos que el 60% de la fuerza laboral pierde su empleo y pasa a categoría 1
mask_unemployment = np.random.rand(len(df_cur)) < 0.6
df_cur.loc[mask_unemployment, 'employment_duration'] = 1
console.print("  🚫 [bold red]Desempleo Masivo:[/bold red] 60% de los solicitantes ahora están desempleados ('employment_duration' = 1).")

# f) Pérdida de Propiedades ('property')
# En German Credit: 1 = Bienes Raíces, 2 = Seguros/Ahorros vivienda, 3 = Auto/Otros, 4 = Desconocido/Sin propiedad
# Simulamos que el 70% pierde sus bienes o hipotecas (ejecuciones hipotecarias), pasando a 4
mask_property_loss = np.random.rand(len(df_cur)) < 0.7
df_cur.loc[mask_property_loss, 'property'] = 4
console.print("  🏠 [bold red]Colapso Inmobiliario:[/bold red] 70% pierde sus propiedades ('property' -> 4).")

# 4. Guardar archivo simulado
df_cur.to_csv(DRIFTED_DATA_PATH, index=False)

console.print(f"\n[bold bright_green][SUCCESS][/bold bright_green] Datos catastróficos guardados en: [cyan]{DRIFTED_DATA_PATH}[/cyan]")
console.print("[INFO] Ejecuta ahora [bold]streamlit run monitor_app.py[/bold] para ver el desastre.")