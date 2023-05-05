from pydantic import BaseModel


class RecommendationsResponseModel(BaseModel):
    user_id: str
    movies: list[str]

    class Config:
        schema_extra = {
            "example": {
                "user_id": "1",
                "movies": ["Home Alone (1990)", "Titanic (1997)"]
            }
        }
