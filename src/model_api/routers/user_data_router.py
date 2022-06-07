from fastapi import APIRouter, Depends, Path, HTTPException

from model_api.dataloaders import DataLoader
from model_api.dependencies import get_data
from model_api.api_classes import ViewResponseModel

router = APIRouter(prefix="/users",
                   tags=["users"],
                   dependencies=[Depends(get_data)],
                   responses={404: {"description": "Not Found"}})


@router.get("/")
async def get_users(data: DataLoader = Depends(get_data)):
    return {"message": data.get_users()}


@router.get("/{user}", response_model=list[ViewResponseModel])
async def get_user_view_data(user: int = Path(..., description="The user ID", ge=1),
                             data: DataLoader = Depends(get_data)):
    return data.query_data(user=str(user)).to_dict("records")


@router.post("/add_view/", responses={404: {"description": "One or more users or movies were invalid."}})
async def add_user_view_data(view_data: list[ViewResponseModel], data: DataLoader = Depends(get_data)):
    users = data.get_users()
    movies = data.get_movies()
    for view in view_data:
        if view.user not in users or view.movie not in movies:
            raise HTTPException(status_code=404, detail="One or more users or movies were invalid.")

    for view in view_data:
        data.post_data(view)

    return {"message": "Data successfully added!"}

