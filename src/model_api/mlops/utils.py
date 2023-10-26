import mlflow
from typing import List
import logging
from mlflow.entities.model_registry.model_version import ModelVersion


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def get_latest_registered_model(model_name: str, stage: str = "Production") -> ModelVersion:
    # List all registered models
    filtered_string = f"name='{model_name}'"

    # Search for latest registered model and order by creation timestamp
    registered_models = mlflow.search_registered_models(filter_string=filtered_string, 
                                                        order_by=["creation_timestamp"])

    # Filter for models with a current stage of "Production"
    production_model = [model for model in registered_models[0].latest_versions if model.current_stage == stage]
    logger.info(f"Found: {production_model[0]}")

    return production_model[0]
