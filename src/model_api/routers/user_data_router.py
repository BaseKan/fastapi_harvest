from fastapi import APIRouter, Depends, Path

from model_api.dataloaders import DataLoader
from model_api.dependencies import get_data
from model_api.api_classes import UserResponseModel, RatingResponseModel

router = APIRouter(prefix="/users",
                   tags=["users"],
                   responses={404: {"description": "Not Found"}})


@router.get("/")
async def get_users(data: DataLoader = Depends(get_data)):
    return {"message": data.get_full_table(table='users').to_dict("records")}


@router.get("/{user}", response_model=list[UserResponseModel])
async def get_user_data(user: int = Path(..., description="The user ID", ge=1),
                        data: DataLoader = Depends(get_data)):
    return data.query_on_col_value(table='users', col_name='user_id', col_value=str(user)).to_dict("records")


@router.post("/add_user_rating/", responses={404: {"description": "One or more users or movies were invalid."}})
async def add_user_rating(user_ratings: list[RatingResponseModel], data: DataLoader = Depends(get_data)):
    new_rating_id = int(data.query_data('select max(rating_id) from ratings').loc[0, :][0])
    for user_rating in user_ratings:
        new_rating_id += 1
        data_dict = user_rating.dict()
        data_dict['rating_id'] = new_rating_id
        data.insert_data(table='ratings', data=data_dict)

    return {"message": "Data successfully added!"}
