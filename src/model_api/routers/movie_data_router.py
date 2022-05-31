from fastapi import APIRouter, Depends, Path

from model_api.dataloaders import DataLoader
from model_api.dependencies import get_data

router = APIRouter(prefix="/movies",
                   tags=["movies"],
                   dependencies=[Depends(get_data)],
                   responses={404: {"description": "Not Found"}})


@router.get("/")
async def get_movies(data: DataLoader = Depends(get_data)):
    return {"message": data.get_movies()}


@router.get("/random/{n}")
async def get_random(n: int = Path(gt=0)):
    pass