# german_credit_ml/eda.py

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDAVisualizer:
    """Clase para generar visualizaciones EDA de un dataset limpio."""

    def __init__(self, input_path: Path, output_dir: Path):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None

    def load_data(self):
        """Carga los datos desde el archivo CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cargando datos limpios desde {self.input_path}...")
        self.df = pd.read_csv(self.input_path)

    def plot_histograms(self):
        """Genera histogramas de columnas numéricas."""
        numeric_cols = ['age', 'amount', 'duration']
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribución de {col.capitalize()}', fontsize=16)
            plt.xlabel(col.capitalize())
            plt.ylabel('Frecuencia')
            plt.tight_layout()

            output_file = self.output_dir / f"hist_{col}.png"
            plt.savefig(output_file)
            plt.close()
            print(f" -> Gráfica guardada en: {output_file}")

    def plot_target_distribution(self):
        """Genera una gráfica de barras para la variable objetivo."""
        plt.figure(figsize=(8, 6))
        sns.countplot(x='credit_risk', data=self.df)
        plt.title('Distribución del Riesgo Crediticio (Target)', fontsize=16)
        plt.xlabel('Riesgo Crediticio (0 = Malo, 1 = Bueno)')
        plt.ylabel('Conteo')
        plt.tight_layout()

        output_file = self.output_dir / "bar_credit_risk.png"
        plt.savefig(output_file)
        plt.close()
        print(f" -> Gráfica guardada en: {output_file}")

    def plot_correlation_heatmap(self):
        """Genera un mapa de calor de correlaciones numéricas."""
        plt.figure(figsize=(16, 12))
        df_numeric = self.df.select_dtypes(include='number')
        corr = df_numeric.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
        plt.title('Mapa de Calor de Correlaciones', fontsize=18)
        plt.tight_layout()

        output_file = self.output_dir / "heatmap_correlation.png"
        plt.savefig(output_file)
        plt.close()
        print(f" -> Gráfica guardada en: {output_file}")

    def run(self):
        """Ejecuta todo el flujo de EDA."""
        self.load_data()
        print("Generando visualizaciones...")
        self.plot_histograms()
        self.plot_target_distribution()
        self.plot_correlation_heatmap()
        print("\nAnálisis EDA completado.")


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar Análisis Exploratorio de Datos (EDA)."
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Ruta al archivo CSV de datos limpios."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directorio donde se guardarán las gráficas."
    )
    args = parser.parse_args()

    eda = EDAVisualizer(Path(args.input_data), Path(args.output_dir))
    eda.run()


if __name__ == "__main__":
    main()
