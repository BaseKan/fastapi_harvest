# def get_best_model()

# Register model

# Serve model as REST API
from mlflow.tracking import MlflowClient
import mlflow

if __name__ == "__main__":

    client = MlflowClient()
    model_name = "NIELS"
    # Search for all the runs in the experiment with the given experiment ID
    current_experiment=dict(mlflow.get_experiment_by_name("Recommender hyperparameter tuning"))
    experiment_id=current_experiment['experiment_id']

    df = mlflow.search_runs([experiment_id], order_by=["metrics.top_100_categorical_accuracy DESC"])
    print(df[["metrics.top_100_categorical_accuracy","run_id"]])
    best_run_id = df.at[0, "run_id"]

    result = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name=model_name
    )

    latest_versions = client.get_latest_versions(name=model_name)

    for version in latest_versions:
        print(f"version: {version.version}, stage: {version.current_stage}")

    # Move new model into production stage and move old model to archived stage
    client.transition_model_version_stage(
        name=model_name,
        version=4,
        stage="Production",
        archive_existing_versions=True
    )




    