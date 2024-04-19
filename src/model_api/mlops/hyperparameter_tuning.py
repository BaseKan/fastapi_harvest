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

# Hyperparameter search space
embedding_dimension = [16, 32, 64, 128]
learning_rate = [0.001, 0.01, 0.1]

# Create Cartesian product of hyperparams
hyperparameter_combinations = list(itertools.product(embedding_dimension, learning_rate))
hyperparameters = [{"embedding_dim": emb_dim, "learning_rate": lr} for emb_dim, lr in hyperparameter_combinations]


def tune_hyperparams(dataset_first_rating_id, dataset_last_rating_id, epochs):
    for hyperparams in hyperparameters:
        with mlflow.start_run() as run:

            run_id = run.info.run_id
            experiment_id = run.info.experiment_id

            # Log parameters
            mlflow.log_params({
                "embedding_dim": hyperparams["embedding_dim"],
                "learning_rate": hyperparams["learning_rate"],
                "epochs": epochs,
                "dataset_first_rating_id": dataset_first_rating_id,
                "dataset_last_rating_id": dataset_last_rating_id
            })
             # TODO: Initialise the model and datasets
            # users, movies = get_vocabulary_datasets(data_loader=data_loader)
            # user_model, movie_model = create_embedding_models(users, movies, embedding_dimension=hyperparams["embedding_dim"])
            
            # Hint: implement process_training_data from retrain_models.py 

             # TODO: Train and evaluate the model
            # Create and compile model
            # retrieval_model = RetrievalModel(...)
            # Compile model

            # cached_train = ratings_train.shuffle(100_000).batch(8192).cache()
            # cached_test = ratings_test.batch(4096).cache()

            model_version_base_path = f"mlruns/{experiment_id}/{run_id}/artifacts"
            
            # Fit retrieval model with cached training data and use callbacks

            evaluation_result = ...

            # Remove prefix from evaluation results 
            evaluation_results_new = {}
            prefix_to_remove = 'factorized_top_k/'

            for key, value in evaluation_result.items():
                if key.startswith(prefix_to_remove):
                    new_key = key[len(prefix_to_remove):]
                else:
                    new_key = key
                evaluation_results_new[new_key] = value
                
            #TODO: Log metrics with mlflow.log_metrics()

            #TODO: Initialize model index
            # index = tfrs.layers.factorized_top_k.BruteForce(retrieval_model.user_model) 
            # Recommends movies out of the entire movies dataset.

            # call once otherwise it cannot be saved
            # input_features = {'user_id': tf.convert_to_tensor(
            #     [[x for x in ratings_test.take(1)][0]['user_id'].numpy()])}

            # _, titles = index({k: np.array(v) for k, v in input_features.items()})
            # print(titles)

            #TODO: Log model
            mlflow.tensorflow.log_model(...)


if __name__ == '__main__':
    tune_hyperparams(dataset_first_rating_id=0, 
                     dataset_last_rating_id=50000, 
                     epochs=5)
