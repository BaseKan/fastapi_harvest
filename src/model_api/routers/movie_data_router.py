from enum import Enum

from fastapi import APIRouter, Depends, Path, Query

from model_api.api_classes import RecommendationsResponseModel
from model_api.dataloaders import DataLoader
from model_api.predictors import TensorflowPredictor
from model_api.dependencies import get_data, get_predictor, data_loader


router = APIRouter(prefix="/movies",
                   tags=["movies"],
                   dependencies=[Depends(get_data)],
                   responses={404: {"description": "Not Found"}})

UserEnum = Enum("UserEnum", {value: value for value in data_loader.get_users()}, type=str)


@router.get("/")
async def get_movies(data: DataLoader = Depends(get_data)):
    return {"message": data.get_movies()}


@router.get("/random/{n}")
async def get_random_movies(n: int = Path(gt=0), data: DataLoader = Depends(get_data)):
    return {"message": data.get_random_movies(n)}


@router.get("/recommend/{user}/{n}", responses={422: {"description": "The user id was invalid."}})
async def recommend_movies(user: UserEnum, n: int, predictor: TensorflowPredictor = Depends(get_predictor)):
    return {"message": predictor.predict({"user_id": [user]})[0][0:n]}


@router.get("recommend/batch/{n}", response_model=list[RecommendationsResponseModel])
async def recommend_movies_batch(n: int, users: list[UserEnum] = Query(),
                                 predictor: TensorflowPredictor = Depends(get_predictor)):
    predictions = predictor.predict({"user_id": [user.value for user in users]})
    return [{"user_id": users[i].value, "movies": predictions[i][0:n]} for i in range(len(users))]
