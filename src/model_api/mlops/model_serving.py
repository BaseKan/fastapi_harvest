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

    # TODO: Search for the best experiment and register the model
    # Search for all the runs in the experiment with the given experiment ID
    current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id=current_experiment['experiment_id']

    df = mlflow.search_runs([experiment_id], order_by=[f"metrics.{metric} DESC"])
    
    # TODO: Get latest version for the model
    # latest_versions = client.get_latest_versions(name=model_name)

    # TODO: Move new model into production stage and move old model to archived stage
    client.transition_model_version_stage(...)

    # We restart the api to trigger reloading of the model
    restart_api()


def register_model(model_name: str, run_id: str, stage: str = "Production"):
    client = MlflowClient()

    mlflow.register_model(...)
    
    latest_versions = client.get_latest_versions(name=model_name)
    
    client.transition_model_version_stage(...)

    restart_api()


def load_registered_retrieval_model(model_name: str, stage: str = "Production",
                                    data_loader: DataLoader | None = None) -> tf.keras.Model:
    raise NotImplementedError()

    client = MlflowClient()

    # search for latest registered model and order by creation timestamp
    registered_model = get_latest_registered_model(model_name=MODEL_NAME, stage=stage)

    logged_model_base_path = ...

    run_id = registered_model.run_id
    
    # TODO: extract params/metrics data for run_id in a single dict
    run_data_dict = client.get_run(run_id).data.to_dictionary()
    # TODO: Get learning rate and embedding dimension
    learning_rate = run_data_dict["params"]["learning_rate"]
    embedding_dimension = ...

    if not data_loader:
        data_loader = DataLoader()

    retrieval_model = init_retrieval_model_checkpoint(model_base_path=logged_model_base_path,
                                                      data_loader=data_loader,
                                                      embedding_dimension=embedding_dimension,
                                                      learning_rate=learning_rate)

    return retrieval_model


def load_registered_predictor_model(model_name: str, stage: str = "Production") -> TensorflowPredictor:
    raise NotImplementedError()
    # Search for latest registered model and order by creation timestamp
    registered_model = get_latest_registered_model(model_name=MODEL_NAME, stage=stage)

    source_path = ...

    return TensorflowPredictor(model_path=source_path)


if __name__ == "__main__":
    load_registered_predictor_model(model_name=MODEL_NAME)
