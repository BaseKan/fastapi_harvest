from enum import Enum

from fastapi import APIRouter, Depends, Path, Query

from model_api.api_classes import RecommendationsResponseModel
from model_api.dataloaders import DataLoader
from model_api.predictors import TensorflowPredictor
from model_api.dependencies import get_data, get_predictor, data_loader


router = APIRouter(prefix="/movies",
                   tags=["movies"],
                   responses={404: {"description": "Not Found"}})

UserEnum = Enum("UserEnum", {str(value): str(value)
                             for value in list(data_loader.get_full_table(table='users').user_id)},
                type=str)


@router.get("/")
async def get_movies(data: DataLoader = Depends(get_data)):
    return {"message": data.get_full_table(table='movies').to_dict("records")}


@router.get("/random/{n}")
async def get_random_movies(n: int = Path(gt=0), data: DataLoader = Depends(get_data)):
    return {"message": data.random_sample_table(table='movies', n=n).to_dict("records")}


@router.get("/ratings/")
async def query_movies(min_rating: float = 0.0, max_rating: float = 5.0, data: DataLoader = Depends(get_data)):
    df = data.query_data(
        query="""select M.movie_title, avg(R.user_rating) as rating 
                 from movies M join ratings R 
                 on M.movie_id=R.movie_id 
                 group by M.movie_title
                 having rating >= ? and rating <= ?
                 """,
        params=[min_rating, max_rating]
    )
    return df.to_dict("records")


@router.get("/recommend/{user}/{n}", responses={422: {"description": "The user id was invalid."}})
async def recommend_movies(user: UserEnum, n: int, predictor: TensorflowPredictor = Depends(get_predictor)):
    print(predictor.predict({"user_id": [user.value]}))
    return {"message": predictor.predict({"user_id": [user.value]})[0][0:n]}


@router.get("recommend/batch/{n}", response_model=list[RecommendationsResponseModel])
async def recommend_movies_batch(n: int, users: list[UserEnum] = Query(),
                                 predictor: TensorflowPredictor = Depends(get_predictor)):
    predictions = predictor.predict({"user_id": [user.value for user in users]})
    return [{"user_id": users[i].value, "movies": predictions[i][0:n]} for i in range(len(users))]
