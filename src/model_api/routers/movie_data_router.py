from fastapi import APIRouter, Depends, Path

from model_api.dataloaders import DataLoader
from model_api.predictors import TensorflowPredictor
from model_api.dependencies import get_data, get_predictor

router = APIRouter(prefix="/movies",
                   tags=["movies"],
                   dependencies=[Depends(get_data)],
                   responses={404: {"description": "Not Found"}})


@router.get("/")
async def get_movies(data: DataLoader = Depends(get_data)):
    return {"message": data.get_movies()}


@router.get("/random/{n}")
async def get_random_movies(n: int = Path(gt=0), data: DataLoader = Depends(get_data)):
    return {"message": data.get_random_movies(n)}


@router.get("/recommend/{user}{n}")
async def recommend_movies(user: str, n: int, predictor: TensorflowPredictor = Depends(get_predictor)):
    print(predictor.predict({"user_id": [user]})[0][0:n])
    return {"message": predictor.predict({"user_id": [user]})[0][0:n]}
