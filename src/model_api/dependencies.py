import json
import os.path
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from fastapi import Depends

from model_api.models.embedding_models import get_vocabulary_datasets, create_embedding_models
from model_api.models.retrieval_model import RetrievalModel
from model_api.retrain_models import MODEL_DIR, RETRIEVAL_CHECKPOINT_PATH
from model_api.dataloaders import DataLoader
from model_api.predictors import TensorflowPredictor

# TODO: MUST BE DYNAMIC
MODEL_VERSION = 0

predictor = TensorflowPredictor(model_path=os.path.join('./model', str(MODEL_VERSION), 'index'))
data_loader = DataLoader()


async def get_predictor() -> TensorflowPredictor:
    return predictor


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
