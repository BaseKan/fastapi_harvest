import json
import os.path
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI

from model_api.models.embedding_models import get_vocabulary_datasets
from model_api.models.retrieval_model import load_retrieval_model_checkpoint
from model_api.dataloaders import DataLoader
from model_api.predictors import TensorflowPredictor
from model_api.mlops.model_serving import load_registered_predictor_model, load_registered_retrieval_model
from model_api.constants import MODEL_DIR, MODEL_NAME, DEFAULT_MODEL_VERSION
import logging


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

data_loader = DataLoader()

models = {"predictor": None, "retrieval_model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    try:
        models["predictor"] = load_registered_predictor_model(model_name=MODEL_NAME)
        logger.info(f"Successfully loaded registered predictor model: {MODEL_NAME}")

        models["retrieval_model"] = load_registered_retrieval_model(model_name=MODEL_NAME, data_loader=data_loader)
        logger.info(f"Successfully loaded retrieval model: {MODEL_NAME}")

    except:
        logger.info(f"Unsuccessfully loaded registered model: {MODEL_NAME}. Using default models.")
        models["predictor"] = TensorflowPredictor(model_path=os.path.join(MODEL_DIR,
                                                                          str(DEFAULT_MODEL_VERSION), 'index'))
        models["retrieval_model"] = load_retrieval_model_checkpoint(data_loader=data_loader, model_version=0)
    yield
    # Any cleanup code here


async def get_predictor() -> TensorflowPredictor:
    return models.get('predictor')


async def get_data() -> DataLoader:
    return data_loader


async def get_vocabulary_dependency(data: DataLoader = Depends(get_data)) -> (pd.DataFrame, pd.DataFrame):
    return get_vocabulary_datasets(data_loader=data)


async def get_full_model() -> tfrs.Model:
    return models.get('retrieval_model')
