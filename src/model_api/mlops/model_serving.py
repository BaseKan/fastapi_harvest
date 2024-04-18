import os
import logging

from mlflow.tracking import MlflowClient
import mlflow
import mlflow.keras
import tensorflow as tf

from model_api.dataloaders import DataLoader
from model_api.models.retrieval_model import init_retrieval_model_checkpoint
from model_api.constants import MODEL_NAME
from model_api.utils.restart_api import restart_api
from model_api.predictors.tensorflow_predictor import TensorflowPredictor
from model_api.mlops.utils import get_latest_registered_model


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def register_best_model(model_name: str, experiment_name: str, metric: str, stage: str = "Production"):
    client = MlflowClient()

    # Search for all the runs in the experiment with the given experiment ID
    current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id=current_experiment['experiment_id']

    df = mlflow.search_runs([experiment_id], order_by=[f"metrics.{metric} DESC"])

    best_run_id = df.at[0, "run_id"]

    mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name=model_name
        )

    # Get latest version
    latest_versions = client.get_latest_versions(name=model_name)

    for version in latest_versions:
        logger.info(f"version: {version.version}, stage: {version.current_stage}")

    # Move new model into production stage and move old model to archived stage
    client.transition_model_version_stage(
        name=model_name,
        stage=stage,
        version=latest_versions[0].version,
        archive_existing_versions=True
    )
    restart_api()


def register_model(model_name: str, run_id: str, stage: str = "Production"):
    client = MlflowClient()

    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name
        )

    # Get latest version
    latest_versions = client.get_latest_versions(name=model_name)
    logger.info(f"Latest version of {model_name} model: {latest_versions[0].version}")

    for version in latest_versions:
        logger.info(f"version: {version.version}, stage: {version.current_stage}")

    # Move new model into production stage and move old model to archived stage
    client.transition_model_version_stage(
        name=model_name,
        stage=stage,
        version=latest_versions[0].version,
        archive_existing_versions=True
    )
    restart_api()


def load_registered_retrieval_model(model_name: str, stage: str = "Production",
                                    data_loader: DataLoader | None = None) -> tf.keras.Model:
    client = MlflowClient()

    # Search for latest registered model and order by creation timestamp
    registered_model = get_latest_registered_model(model_name=MODEL_NAME, stage=stage)

    logger.info(f"Source path of retrieval model: {registered_model.source}")
    source_path = f"./mlruns{registered_model.source.split('mlruns', 1)[1]}"

    logger.info(f"Latest version of {model_name} model in Production stage: {registered_model.version}")

    logged_model_base_path = source_path.rsplit("/model", 1)[0]

    run_id = registered_model.run_id

    logger.info(f"Run id of registered model: {run_id}")
    logger.info(f"Source path of registered model: {source_path}")
    
    # extract params/metrics data for run_id in a single dict 
    run_data_dict = client.get_run(run_id).data.to_dictionary()
    # Get learning rate and embedding dimension
    learning_rate = run_data_dict["params"]["learning_rate"]
    embedding_dimension = run_data_dict["params"]["embedding_dim"]

    if not data_loader:
        data_loader = DataLoader()

    retrieval_model = init_retrieval_model_checkpoint(model_base_path=logged_model_base_path,
                                                      data_loader=data_loader,
                                                      embedding_dimension=embedding_dimension,
                                                      learning_rate=learning_rate)

    return retrieval_model


def load_registered_predictor_model(model_name: str, stage: str = "Production") -> TensorflowPredictor:

    # Search for latest registered model and order by creation timestamp
    registered_model = get_latest_registered_model(model_name=MODEL_NAME, stage=stage)

    logger.info(f"Registered model info: {registered_model}")

    source_path = f"./mlruns{registered_model.source.split('mlruns', 1)[1]}/data/model"
    logger.info(f"Source path of registered model: {source_path}")
    logger.info(f"Latest version of {model_name} model in Production stage: {registered_model.version}")

    return TensorflowPredictor(model_path=source_path)


if __name__ == "__main__":
    load_registered_predictor_model(model_name=MODEL_NAME)