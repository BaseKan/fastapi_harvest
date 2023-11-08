import abc

import tensorflow as tf

from model_api.base_classes import PredictorBase


class TensorflowPredictorBase(PredictorBase):
    model_path: str
    _model: tf.keras.Model

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def load_model(self, path: str) -> tf.keras.Model:
        pass

    @abc.abstractmethod
    def predict(self, input_features: dict[str, list]) -> tf.Tensor:
        pass

    @abc.abstractmethod
    def reload_model(self):
        pass

    class Config:
        arbitrary_types_allowed = True
