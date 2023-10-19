import numpy as np
import tensorflow as tf

from model_api.tensorflow_base_classes import TensorflowPredictorBase


class TensorflowPredictor(TensorflowPredictorBase):

    def __init__(self, **data):
        super().__init__(**data)
        self._model = self.load_model(self.model_path)

    def load_model(self, path: str) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def predict(self, input_features: dict[str, list]) -> list[list[str]]:
        print(self._model({k: np.array(v) for k, v in input_features.items()}))
        predictions = self._model({k: np.array(v) for k, v in input_features.items()})[1].numpy()
        return [[tf.compat.as_str_any(result) for result in prediction] for prediction in predictions]

    def reload_model(self):
        self._model = self.load_model(self.model_path)
