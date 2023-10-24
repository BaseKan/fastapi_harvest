from mlflow import MlflowClient
import mlflow
import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

from model_api.mlflow.model_serving import load_registered_retrieval_model
from model_api.models.embedding_models import get_vocabulary_datasets, process_training_data, create_embedding_models
from model_api.dependencies import data_loader
from model_api.constants import RETRIEVAL_CHECKPOINT_PATH


def check_for_retraining(experiment_name_monitoring: str, 
                         current_model_name: str, 
                         threshold: float, 
                         dataset_first_rating_id: int,
                         dataset_last_rating_id: int,
                         new_experiment_name: str, 
                         epochs: int):

    current_experiment=dict(mlflow.get_experiment_by_name(experiment_name_monitoring))
    experiment_id=current_experiment['experiment_id']

    current_performance_run = mlflow.search_runs(
    experiment_ids=experiment_id,
    order_by=["metrics.top_k_100 ASC"]
    ).loc[0]

    if current_performance_run["metrics.top_k_100"] <= threshold:
        # last_rating_id = run["run_id"]

        # List all registered models
        filtered_string = f"name='{current_model_name}'"

        # Search for latest registered model and order by creation timestamp
        registered_models = mlflow.search_registered_models(filter_string=filtered_string, 
                                                            order_by=["creation_timestamp"])

        # Filter for models with a current stage of "Production"
        production_models = [model for model in registered_models if model.latest_versions[0].current_stage == "Production"]
        run_id_current_model = production_models[0].latest_versions[0].run_id

        # Load current model including weights
        retrieval_model = load_registered_retrieval_model(model_name=current_model_name)

        mlflow.set_experiment(new_experiment_name)

        with mlflow.start_run() as run:

            run_id = run.info.run_id
            experiment_id = run.info.experiment_id

            users, movies = get_vocabulary_datasets(data_loader=data_loader)
            # user_model, movie_model = create_embedding_models(users, movies, embedding_dimension=embedding_dimension)

            ratings_train, ratings_test, movies_ds = process_training_data(data_loader=data_loader, movies=movies,
                                                                           dataset_first_rating_id=dataset_first_rating_id,
                                                                           dataset_last_rating_id=dataset_last_rating_id)
            
            cached_train = ratings_train.shuffle(100_000).batch(8192).cache()
            cached_test = ratings_test.batch(4096).cache()

            # in map data van artifacts een folder aanmaken genaamd 'retrieval_model'
            model_version_base_path = f"mlruns/{experiment_id}/{run_id}/artifacts"

            retrieval_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_version_base_path, RETRIEVAL_CHECKPOINT_PATH),
                save_weights_only=True,
                monitor='loss',
                mode='min',
                save_best_only=True)
            
            retrieval_model.fit(cached_train, epochs=epochs, callbacks=retrieval_checkpoint_callback)

            evaluation_result = retrieval_model.evaluate(cached_test, return_dict=True)
            print(evaluation_result)

            # Remove prefix from evaluation results 
            evaluation_results_new = {}
            prefix_to_remove = 'factorized_top_k/'

            for key, value in evaluation_result.items():
                if key.startswith(prefix_to_remove):
                    new_key = key[len(prefix_to_remove):]
                else:
                    new_key = key
                evaluation_results_new[new_key] = value
            # Log metrics
            mlflow.log_metrics(evaluation_results_new)

            index = tfrs.layers.factorized_top_k.BruteForce(retrieval_model.user_model) 
            # recommends movies out of the entire movies dataset.
            index.index_from_dataset(
                tf.data.Dataset.zip((movies_ds.batch(100), movies_ds.batch(100).map(retrieval_model.movie_model)))
            )

            # call once otherwise it cannot be saved
            input_features = {'user_id': tf.convert_to_tensor(
                [[x for x in ratings_test.take(1)][0]['user_id'].numpy()])}

            _, titles = index({k: np.array(v) for k, v in input_features.items()})
            print(titles)

            # Log model
            mlflow.tensorflow.log_model(model=index, artifact_path="model")

            # Register snd replace model if performance is better than current model on the same test set
            print(f"Current model performance: {current_performance_run['metrics.top_k_100']}")


if __name__ == "__main__":
    check_for_retraining(experiment_name_monitoring="model monitoring", 
                         current_model_name="harvest_recommender", 
                         threshold=0.35, 
                         dataset_first_rating_id=0,
                         dataset_last_rating_id=80000,
                         new_experiment_name="retraining experiment",
                         epochs=4)
    