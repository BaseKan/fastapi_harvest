import os.path

from model_api.dataloaders import DataLoader
from model_api.predictors import TensorflowPredictor
from model_api.constants import MODEL_VERSION, MODEL_DIR

predictor = TensorflowPredictor(model_path=os.path.join(MODEL_DIR, str(MODEL_VERSION), 'index'))

data_loader = DataLoader()


async def get_predictor() -> TensorflowPredictor:
    return predictor


async def get_data() -> DataLoader:
    return data_loader
