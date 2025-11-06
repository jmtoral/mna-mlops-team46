# german_credit_ml/check_drift.py

import argparse
from pathlib import Path
import warnings
from dataclasses import dataclass
from typing import Tuple

import pandas as pd

# Importaciones de Evidently (con la ruta corregida)
from evidently.base_metric import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# --- Importaciones de Rich y utilidades ---
from german_credit_ml.utils import console, print_header # Importar consola y header

# Ignorar advertencias futuras
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Clases de Configuración ---

@dataclass(frozen=True)
class Paths:
    """Almacena las rutas necesarias para este script."""
    input_data: Path
    report_output: Path

@dataclass(frozen=True)
class DriftConfig:
    """Configuración para el chequeo de deriva."""
    split_point: float = 0.7 # Porcentaje para datos de referencia

# --- Clases de Lógica ---

class DriftDataModule:
    """Clase para cargar y dividir los datos para el análisis de deriva."""
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        console.print(f"[bold green][INFO][/bold green] Cargando datos desde: [cyan]{self.csv_path}[/cyan]")
        df = pd.read_csv(self.csv_path)
        console.print(f"[bold bright_green][SUCCESS][/bold bright_green] Datos cargados: {df.shape[0]} filas.")
        return df

    def split(self, df: pd.DataFrame, cfg: DriftConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide el dataframe en referencia y actual."""
        console.print(f"[INFO] Dividiendo datos en referencia ({cfg.split_point*100}%) y actuales...")
        split_index = int(len(df) * cfg.split_point)
        reference_df = df.iloc[:split_index].copy()
        current_df = df.iloc[split_index:].copy()
        console.print(f"[SUCCESS] División completa. Referencia: {len(reference_df)} filas, Actuales: {len(current_df)} filas.")
        return reference_df, current_df

class DriftReportRunner:
    """Clase principal que orquesta el análisis de deriva."""
    def __init__(self, paths: Paths, cfg: DriftConfig):
        self.paths = paths
        self.cfg = cfg

    def run(self):
        """Ejecuta todo el flujo de análisis de deriva."""
        print_header() # Imprime encabezado ASCII
        console.print("\n" + "="*50, style="bold dim")
        console.print(f" [bold yellow]Iniciando Chequeo de Deriva de Datos[/bold yellow] ".center(50, "="), style="bold dim")
        console.print("="*50, style="bold dim")

        # Cargar y dividir datos
        data_module = DriftDataModule(self.paths.input_data)
        df = data_module.load()
        reference_df, current_df = data_module.split(df, self.cfg)

        # Definir el reporte de Evidently
        column_mapping = ColumnMapping()
        drift_report = Report(metrics=[DataDriftPreset()])
        
        console.print("\n[bold green][INFO][/bold green] Calculando métricas de deriva... (esto puede tardar)")
        drift_report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
        
        # Guardar el reporte HTML
        self.paths.report_output.parent.mkdir(parents=True, exist_ok=True)
        drift_report.save_html(str(self.paths.report_output))
        
        console.print(f"\n[bold bright_green][SUCCESS][/bold bright_green] Reporte de deriva guardado en: [cyan]{self.paths.report_output}[/cyan]")
        
        # Opcional: Imprimir un resumen en la consola
        try:
            report_dict = drift_report.as_dict()
            drift_details = report_dict['metrics'][0]['result']
            num_drifted = drift_details['number_of_drifted_columns']
            console.print(f"[INFO] Resumen de Deriva: [bold magenta]{num_drifted}[/bold magenta] de {drift_details['number_of_columns']} columnas presentan deriva.")
        except Exception:
            pass # No es crítico si el resumen falla

        console.print("\n[bold bright_green][SUCCESS][/bold bright_green] Chequeo de deriva finalizado.")

# --- Bloque de ejecución principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar análisis de deriva de datos con Evidently.")
    parser.add_argument("--input-data", type=Path, required=True, help="Ruta al CSV de datos limpios.")
    parser.add_argument("--report-output", type=Path, required=True, help="Ruta para guardar el reporte HTML.")
    
    args = parser.parse_args()

    # Crear instancias de configuración
    paths = Paths(
        input_data=args.input_data,
        report_output=args.report_output
    )
    # Usar valores por defecto para la configuración
    cfg = DriftConfig()

    # Crear e iniciar el runner
    runner = DriftReportRunner(paths, cfg)
    runner.run()