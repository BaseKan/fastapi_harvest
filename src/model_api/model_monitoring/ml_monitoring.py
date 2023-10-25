import tensorflow as tf
import mlflow

from model_api.mlflow.model_serving import load_registered_predictor_model, load_registered_retrieval_model
from model_api.dataloaders import DataLoader
from model_api.models.embedding_models import get_vocabulary_datasets
from model_api.constants import MODEL_NAME
from model_api.mlflow.utils import get_latest_registered_model


def monitor_performance(model_name: str, experiment_name: str, stage: str = "Production"):

    recommender_model = load_registered_retrieval_model(model_name=model_name, stage=stage)

    # Create experiment
    mlflow.set_experiment(experiment_name)

    # Search for latest registered model and order by creation timestamp
    registered_model = get_latest_registered_model(model_name=MODEL_NAME, stage=stage)
    run_id_current_model = registered_model.run_id

    # Set experiment batches dates
    experiment_batches = [
        (0,50000),
        (0,60000),
        (0,70000),
        (0,80000),
        (0,90000),  
    ]

    for batch in experiment_batches:
        with mlflow.start_run(run_name="model monitoring run") as run:

            mlflow.log_param("model_run_id", run_id_current_model)
            mlflow.log_param("first_rating_id", batch[0])
            mlflow.log_param("last_rating_id", batch[1])
            mlflow.log_param("registered_model_name", model_name)

            data = DataLoader()
            vocabulary = get_vocabulary_datasets(data_loader=data)
            _, movies = vocabulary

            # Get performance
            # ---------------------------------------------------------------------------------------------------
            ratings_df = (data.get_ratings_id_range(first_id=batch[0],
                                                    last_id=batch[1])
                            .merge(movies, on='movie_id')
                            .loc[:, ['movie_title', 'user_id']]
                            .astype({'user_id': str}))
            
            ratings_ds = tf.data.Dataset.from_tensor_slices(dict(ratings_df)).batch(4096).cache()

            model_performance = recommender_model.evaluate(ratings_ds, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]

            mlflow.log_metric("top_k_100", model_performance, batch[1])


if __name__ == "__main__":

    monitor_performance(model_name=MODEL_NAME, experiment_name="Model monitoring")
