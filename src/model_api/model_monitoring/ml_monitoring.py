import tensorflow as tf
import mlflow

from model_api.mlflow.model_serving import load_registered_predictor_model, load_registered_retrieval_model
from model_api.dataloaders import DataLoader
from model_api.models.embedding_models import get_vocabulary_datasets


def monitor_performance(model_name: str, experiment_name: str):

    recommender_model = load_registered_retrieval_model(model_name="harvest_recommender", stage="Production")

    # Create experiment
    mlflow.set_experiment(experiment_name)

    # List all registered models
    filtered_string = f"name='{model_name}'"

    # Search for latest registered model and order by creation timestamp
    registered_models = mlflow.search_registered_models(filter_string=filtered_string, 
                                                        order_by=["creation_timestamp"])

    # Filter for models with a current stage of "Production"
    production_models = [model for model in registered_models if model.latest_versions[0].current_stage == "Production"]
    run_id = production_models[0].latest_versions[0].run_id

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

            mlflow.log_param("model_run_id", run_id)
            mlflow.log_param("batches", batch[1])

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
    monitor_performance(model_name="harvest_recommender", experiment_name="model monitoring")
