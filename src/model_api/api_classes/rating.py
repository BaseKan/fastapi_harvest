from pydantic import BaseModel


class RatingResponseModel(BaseModel):
    user_id: int
    movie_id: int
    user_rating: float

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "movie_id": 4,
                "user_rating": 3.5,
            }
        }
