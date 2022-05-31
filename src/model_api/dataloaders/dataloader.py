from typing import Any, Optional
from random import sample

import pandas as pd

from model_api.base_classes import DataLoaderBase
from model_api.api_classes import ViewResponseModel
from model_api.tensorflow_base_classes import TensorflowPredictorBase


class DataLoader(DataLoaderBase):
    data_path: Optional[str] = ""
    _all_users: list[str]
    _all_movies: list[str]
    _user_view_data: pd.DataFrame

    def __init__(self, predictor_model: TensorflowPredictorBase, **data):
        super().__init__(**data)
        self._all_users = predictor_model.model.query_model.get_layer(
            "embedding_model").get_layer("sequential").get_layer("string_lookup").get_vocabulary()[1:]
        self._all_movies = list(predictor_model.model.identifiers.numpy())
        self._user_view_data = pd.DataFrame({"user": ["1", "2"],
                                             "movie": ["Winnie the Pooh and the Blustery Day (1968)",
                                                       "Home Alone (1990)"]})

    def get_data(self) -> pd.DataFrame:
        return self._user_view_data

    def get_users(self) -> list[str]:
        return self._all_users

    def get_movies(self) -> list[str]:
        return self._all_movies

    def get_random_movies(self, n: int) -> list[str]:
        return sample(self._all_movies, n)

    def post_data(self, new_data: ViewResponseModel):
        self._user_view_data = self._user_view_data.append(pd.DataFrame(new_data.dict(), index=[0]))

    def set_data(self, key: int, col: str, value: Any):
        self._user_view_data.loc[self._user_view_data["user"] == key, col] = value

    def query_data(self, **kwargs) -> pd.DataFrame:
        return self._user_view_data.query(
            " and ".join(k + "==" + ("'" + v + "'" if type(v) == str else str(v)) for k, v in kwargs.items()))

    def delete_data(self, key):
        self._user_view_data = self._user_view_data.loc[self._user_view_data["user"] != key]
