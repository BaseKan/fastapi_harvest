import os

from mlflow.tracking import MlflowClient
import mlflow
import mlflow.keras
import tensorflow as tf

from model_api.dataloaders import DataLoader
from model_api.models.embedding_models import get_vocabulary_datasets, create_embedding_models
from model_api.models.retrieval_model import RetrievalModel
from model_api.constants import RETRIEVAL_CHECKPOINT_PATH
from model_api.predictors.tensorflow_predictor import TensorflowPredictor



def register_best_model(model_name: str, experiment_name: str, metric: str, stage: str = "Production"):
    client = MlflowClient()

    # Search for all the runs in the experiment with the given experiment ID
    current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id=current_experiment['experiment_id']

    df = mlflow.search_runs([experiment_id], order_by=[f"metrics.{metric} DESC"])
    print(df[[f"metrics.{metric}","run_id"]])

    best_run_id = df.at[0, "run_id"]

    mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name=model_name
        )

    # Get latest version
    latest_versions = client.get_latest_versions(name=model_name)

    for version in latest_versions:
        print(f"version: {version.version}, stage: {version.current_stage}")

    # Move new model into production stage and move old model to archived stage
    client.transition_model_version_stage(
        name=model_name,
        stage=stage,
        version=version.version,
        archive_existing_versions=True
    )


def load_registered_retrieval_model(model_name: str, stage: str = "Production") -> tf.keras.Model:
    client = MlflowClient()

    # List all registered models
    filtered_string = f"name='{model_name}'"

    # Search for latest registered model and order by creation timestamp
    registered_models = mlflow.search_registered_models(filter_string=filtered_string, 
                                                        order_by=["creation_timestamp"])

    # Filter for models with a current stage of "Production"
    production_models = [model for model in registered_models if model.latest_versions[0].current_stage == stage]
    source_path = f"./mlruns{production_models[0].latest_versions[0].source.split('mlruns', 1)[1]}"
    logged_model_base_path = source_path.rsplit("/model", 1)[0]

    run_id = production_models[0].latest_versions[0].run_id

    print(f"Source path={source_path}")
    print(f"Run id={run_id}")
    
    # extract params/metrics data for run_id in a single dict 
    run_data_dict = client.get_run(run_id).data.to_dictionary()
    # Get learning rate
    learning_rate = run_data_dict["params"]["learning_rate"]
    embedding_dimension = run_data_dict["params"]["embedding_dim"]

    data = DataLoader()
    vocabulary = get_vocabulary_datasets(data_loader=data)

    users, movies = vocabulary
    user_model, movie_model = create_embedding_models(users, movies, embedding_dimension=int(embedding_dimension))
    movies_ds = tf.data.Dataset.from_tensor_slices(dict(movies)).map(lambda x: x['movie_title'])
    retrieval_model = RetrievalModel(user_model, movie_model, movies_ds)

    retrieval_model.load_weights(filepath=os.path.join(logged_model_base_path, RETRIEVAL_CHECKPOINT_PATH))
    retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=float(learning_rate)))

    return retrieval_model


def load_registered_predictor_model(model_name: str, stage: str = "Production") -> tf.keras.Model:
    # List all registered models
    filtered_string = f"name='{model_name}'"

    # Search for latest registered model and order by creation timestamp
    registered_models = mlflow.search_registered_models(filter_string=filtered_string, 
                                                        order_by=["creation_timestamp"])

    # Filter for models with a current stage of "Production"
    production_models = [model for model in registered_models if model.latest_versions[0].current_stage == stage]
    source_path = f"./mlruns{production_models[0].latest_versions[0].source.split('mlruns', 1)[1]}/data/model"

    predictor = TensorflowPredictor(model_path=source_path)

    print(predictor.predict({"user_id": ["151"]}))









if __name__ == "__main__":
    # register_best_model(model_name="harvest_recommender", 
    #                     experiment_name="Recommender hyperparameter tuning", 
    #                     metric="top_100_categorical_accuracy"
    #                     )

    load_registered_predictor_model(model_name="harvest_recommender", 
            #    experiment_name="Recommender hyperparameter tuning", 
               stage="Production"
               )
    