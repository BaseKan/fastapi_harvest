from mlflow import MlflowClient
import mlflow
import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from model_api.models.retrieval_model import RetrievalModel
import logging

from model_api.mlops.model_serving import load_registered_retrieval_model
from model_api.models.embedding_models import get_vocabulary_datasets, process_training_data, create_embedding_models
from model_api.dependencies import data_loader
from model_api.constants import (RETRIEVAL_CHECKPOINT_PATH, 
                                 RETRAINING_EXPERIMENT_NAME, 
                                 MONITORING_EXPERIMENT_NAME, 
                                 MODEL_NAME)
from model_api.mlops.model_serving import register_model
from model_api.mlops.utils import get_latest_registered_model


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def check_for_retraining(experiment_name_monitoring: str, 
                         current_model_name: str, 
                         threshold: float, 
                         dataset_first_rating_id: int,
                         dataset_last_rating_id: int,
                         new_experiment_name: str, 
                         epochs: int):

    current_monitoring_experiment = dict(mlflow.get_experiment_by_name(experiment_name_monitoring))
    monitoring_experiment_id = current_monitoring_experiment['experiment_id']

    # TODO: Get worst performing batch in the monitoring experiment
    current_performance_run = ...

    if current_performance_run["metrics.top_k_100"] <= threshold:
        # TODO: Search for latest registered model and order by creation timestamp
        # Get learning rate and embedding dimension
        learning_rate = ...
        embedding_dimension = ...

        # Load current model including weights
        current_retrieval_model = load_registered_retrieval_model(model_name=current_model_name)

        mlflow.set_experiment(new_experiment_name)

        with mlflow.start_run() as run:

            run_id = run.info.run_id
            experiment_id = run.info.experiment_id

            # Log parameters
            mlflow.log_params({
                "embedding_dim": embedding_dimension,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "dataset_first_rating_id": dataset_first_rating_id,
                "dataset_last_rating_id": dataset_last_rating_id
            })

            model_version_base_path = f"mlruns/{experiment_id}/{run_id}/artifacts"
            # TODO: Create and compile model

            # TODO: Fit and evaluate new retrieval model with stateful weights

            # TODO: Also evaluate the current best model
            new_evaluation_result = ...
            current_evaluation_result = ...

            # TODO: Register and replace model if performance is better than current model on the same test set
            if new_evaluation_result["factorized_top_k/top_100_categorical_accuracy"] > current_evaluation_result["factorized_top_k/top_100_categorical_accuracy"]:
                # Remove prefix from evaluation results 
                evaluation_results_new = {}
                prefix_to_remove = 'factorized_top_k/'

                for key, value in new_evaluation_result.items():
                    if key.startswith(prefix_to_remove):
                        new_key = key[len(prefix_to_remove):]
                    else:
                        new_key = key
                    evaluation_results_new[new_key] = value
                # TODO: Log metrics

                # TODO: Initialize index

                # TODO: Log model

                # TODO: Register new trained model
                register_model(model_name=current_model_name, run_id=run_id)


if __name__ == "__main__":
    check_for_retraining(experiment_name_monitoring=MONITORING_EXPERIMENT_NAME, 
                         current_model_name=MODEL_NAME, 
                         threshold=0.5, 
                         dataset_first_rating_id=0,
                         dataset_last_rating_id=90000,
                         new_experiment_name=RETRAINING_EXPERIMENT_NAME,
                         epochs=5)
    