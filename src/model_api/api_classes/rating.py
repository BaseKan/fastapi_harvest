from pydantic import BaseModel, validator
from model_api.dependencies import data_loader


class RatingResponseModel(BaseModel):
    user_id: int
    movie_id: int
    user_rating: float

    @validator('user_id')
    def validate_user_id(cls, v):
        if data_loader.query_on_col_value(table='users', col_name='user_id', col_value=str(v)).empty:
            raise ValueError('Invalid user_id.')

        return v

    @validator('movie_id')
    def validate_movie_id(cls, v):
        if data_loader.query_on_col_value(table='movies', col_name='movie_id', col_value=str(v)).empty:
            raise ValueError('Invalid movie_id.')

        return v

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "movie_id": 4,
                "user_rating": 3.5,
            }
        }
