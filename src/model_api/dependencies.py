from model_api.dataloaders import DataLoader
from model_api.predictors import TensorflowPredictor


predictor = TensorflowPredictor(model_path="./model/index")
data_loader = DataLoader()


async def get_predictor() -> TensorflowPredictor:
    return predictor


async def get_data() -> DataLoader:
    return data_loader
