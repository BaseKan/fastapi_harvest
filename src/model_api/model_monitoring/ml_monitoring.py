import tensorflow as tf
import mlflow

from model_api.mlops.model_serving import load_registered_predictor_model, load_registered_retrieval_model
from model_api.dataloaders import DataLoader
from model_api.models.embedding_models import get_vocabulary_datasets
from model_api.constants import MODEL_NAME
from model_api.mlops.utils import get_latest_registered_model


def monitor_performance(model_name: str, experiment_name: str, stage: str = "Production"):

    recommender_model = load_registered_retrieval_model(model_name=model_name, stage=stage)

    # Create experiment
    mlflow.set_experiment(experiment_name)

    # Search for latest registered model and order by creation timestamp
    registered_model = get_latest_registered_model(model_name=MODEL_NAME, stage=stage)
    run_id_current_model = registered_model.run_id

    # Set experiment batches dates
    experiment_batches = [
        (0, 50000),
        (0, 60000),
        (0, 70000),
        (0, 80000),
        (0, 90000),
    ]

    for batch in experiment_batches:
        with mlflow.start_run(run_name="model monitoring run") as run:
            # TODO: Log parameter to MLFlow
            mlflow.log_param("model_run_id", ...)
            ...

            # TODO: Evaluate model
            # ---------------------------------------------------------------------------------------------------

            mlflow.log_metric("top_k_100", ...)


if __name__ == "__main__":
    monitor_performance(model_name=MODEL_NAME, experiment_name="Model monitoring")
