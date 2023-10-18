import os.path

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
