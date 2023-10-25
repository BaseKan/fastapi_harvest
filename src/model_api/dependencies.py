import json
import os.path
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI

from model_api.models.embedding_models import get_vocabulary_datasets, create_embedding_models
from model_api.models.retrieval_model import RetrievalModel
from model_api.dataloaders import DataLoader
from model_api.predictors import TensorflowPredictor
from model_api.mlflow.model_serving import load_registered_predictor_model, load_registered_retrieval_model
from model_api.constants import MODEL_VERSION, MODEL_DIR, RETRIEVAL_CHECKPOINT_PATH, MODEL_NAME
import logging


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

models = {"predictor": None, "retrieval_model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    try:
        models["predictor"] = load_registered_predictor_model(model_name=MODEL_NAME)
        logger.info(f"Successfully loaded registered predictor model: {MODEL_NAME}")

        models["retrieval_model"] = load_registered_retrieval_model(model_name=MODEL_NAME)
        logger.info(f"Successfully loaded retrieval model: {MODEL_NAME}")

    except:
        logger.info(f"Unsuccessfully loaded registered model: {MODEL_NAME}")
        pass
    yield
    # Any cleanup code here

data_loader = DataLoader()


async def get_predictor() -> TensorflowPredictor:
    return models.get('predictor')


async def get_data() -> DataLoader:
    return data_loader


async def get_vocabulary_dependency(data: DataLoader = Depends(get_data)) -> (pd.DataFrame, pd.DataFrame):
    return get_vocabulary_datasets(data_loader=data)


async def get_full_model(vocabulary: (pd.DataFrame, pd.DataFrame) = Depends(get_vocabulary_dependency)) -> tfrs.Model:
    model_version_base_path = os.path.join(MODEL_DIR, str(MODEL_VERSION))
    with open(os.path.join(model_version_base_path, 'training_parameters.json')) as file:
        training_parameters = json.load(file)

    embedding_dimension = training_parameters.get('embedding_dimension', 32)

    users, movies = vocabulary
    user_model, movie_model = create_embedding_models(users, movies, embedding_dimension=embedding_dimension)
    movies_ds = tf.data.Dataset.from_tensor_slices(dict(movies)).map(lambda x: x['movie_title'])
    retrieval_model = RetrievalModel(user_model, movie_model, movies_ds)
    retrieval_model.load_weights(filepath=os.path.join(model_version_base_path, RETRIEVAL_CHECKPOINT_PATH))
    retrieval_model.compile()

    return retrieval_model
