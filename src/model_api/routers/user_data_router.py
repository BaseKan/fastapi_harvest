from datetime import timedelta
from typing import Any, Annotated

from fastapi import APIRouter, Depends, Path, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import status
from fastapi.encoders import jsonable_encoder

from model_api.dataloaders import DataLoader
from model_api.dependencies import get_data
from model_api.api_classes import UserResponseModel, RatingResponseModel
from model_api.utils.authentication import get_password_hash
from model_api.utils.authentication import authenticate_user
from model_api.utils.authentication import token_exception
from model_api.utils.authentication import create_access_token
from model_api.utils.authentication import get_current_user
from model_api.utils.authentication import Token


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
<<<<<<< HEAD


@router.post("/add_user_rating/", responses={404: {"description": "One or more users or movies were invalid."}})
async def add_user_rating(user_ratings: list[RatingResponseModel], data: DataLoader = Depends(get_data)):
    new_rating_id = int(data.query_data('select max(rating_id) from ratings').loc[0, :][0])
    print(new_rating_id)
    for user_rating in user_ratings:
        new_rating_id += 1
        data_dict = user_rating.dict()
        data_dict['rating_id'] = new_rating_id
        data.insert_data(table='ratings', data=data_dict)

    return {"message": "Data successfully added!"}


@router.post("/update_password")
async def update_password(user_id: int, password: str, data: DataLoader = Depends(get_data)) -> JSONResponse:
    "Update a hashed password in the database by providing a user_id and password."
    user = data.query_on_col_value(table='users', col_name='user_id', col_value=user_id)
    hashed_password = get_password_hash(password)

    password_to_update = {"hashed_passwords": hashed_password}
    user_tuple = ("user_id", user_id)

    _ = data.update_data("users", user_tuple, password_to_update)

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"Message": "Created user successfully",
                 "User details": jsonable_encoder(user.to_dict('records'))
                 }
    )


@router.post("/token", response_model=Token)
async def login_for_access_token(data: Annotated[DataLoader, Depends(get_data)],
                                 form_data: OAuth2PasswordRequestForm = Depends()) -> dict[str, str]:
    "Creates and returns an access token (JWT) when user is authenticated."

    user = authenticate_user(user_id=int(form_data.username),
                             password=form_data.password,
                             dataloader=data)

    if user.empty:
        raise token_exception()
    
    token_expires = timedelta(minutes=20)
    token = create_access_token(data={'sub': str(user.user_id[0])},
                                expires_delta=token_expires)
    
    return {"access_token": token, "token_type": "bearer"}


@router.get("/secure_ratings/")
async def read_ratings_by_user(user_id: int = Depends(get_current_user),
                               data: DataLoader = Depends(get_data)
                               ) -> JSONResponse:
    "Returns ratings by the current user."
    if user_id is None:
        raise HTTPException(status_code=404, detail="User not found")

    df = data.query_data(
            query=f"""select R.movie_id, R.user_rating
                     from ratings R
                     where R.user_id = {user_id}
                     """
                     )
    
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"Message": f"Recommendations for user {user_id}",
                 "Recommendations": jsonable_encoder(df.to_dict("records")[:5])
                 }
    )
=======
>>>>>>> master
