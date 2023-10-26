import itertools
import os

import mlflow
from mlflow.tracking import MlflowClient
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

from model_api.models.embedding_models import get_vocabulary_datasets, process_training_data, create_embedding_models
from model_api.models.retrieval_model import RetrievalModel
from model_api.constants import RETRIEVAL_CHECKPOINT_PATH
from model_api.dependencies import data_loader


client = MlflowClient()

# Create experiment
mlflow.set_experiment("Recommender hyperparameter tuning")


def tune_hyperparams(dataset_first_rating_id, dataset_last_rating_id, epochs):
    with mlflow.start_run() as run:

        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        # Log parameters
        mlflow.log_params({
            "embedding_dim": ...,
            "learning_rate": ...,
            "epochs": epochs,
            "dataset_first_rating_id": dataset_first_rating_id,
            "dataset_last_rating_id": dataset_last_rating_id
        })

        model_version_base_path = f"mlruns/{experiment_id}/{run_id}/artifacts"
        # TODO: Initialise the model and datasets

        # TODO: Train and evaluate the model

        evaluation_result = {}
        # Remove prefix from evaluation results
        evaluation_results_new = {}
        prefix_to_remove = 'factorized_top_k/'

        for key, value in evaluation_result.items():
            if key.startswith(prefix_to_remove):
                new_key = key[len(prefix_to_remove):]
            else:
                new_key = key
            evaluation_results_new[new_key] = value

        # TODO: Log metrics

        # TODO: Initialize model index

        # TODO: Log model


if __name__ == '__main__':
    tune_hyperparams(dataset_first_rating_id=0, 
                     dataset_last_rating_id=50000, 
                     epochs=5)
