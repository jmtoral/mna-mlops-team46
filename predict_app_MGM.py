import mlflow
import mlflow.sklearn


# @st.cache_resource
def load_model():
    """Carga el modelo desde MLflow local."""
    try:
        # Conecta al tracking local
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(
            "fase1_modelado_equipo46"
        )  # Deberá actualizarse si se promueve el Model Registry

        # Carga el modelo más reciente del experimento
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("fase1_modelado_equipo46")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None, "No se encontró ningún modelo registrado en MLflow."

        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"

        model = mlflow.sklearn.load_model(model_uri)
        return model, f"Modelo cargado desde MLflow (run_id={run_id})"

    except Exception as e:
        return None, f"⚠️ Error al cargar desde MLflow: {str(e)}"
