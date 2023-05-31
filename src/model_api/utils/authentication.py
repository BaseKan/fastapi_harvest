from typing import Annotated, Optional
from pydantic import BaseModel
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import status
from fastapi import HTTPException

from model_api.dataloaders import DataLoader


SECRET_KEY = "FNE#NG#34tSNGFEWG3r323r234tfnfweGFWEF!"
ALGORITHM = "HS256"

bcrypt_context = CryptContext(schemes=["bcrypt"])

oauth2_bearer = OAuth2PasswordBearer(tokenUrl="/users/token")


class Token(BaseModel):
    access_token: str
    token_type: str


def get_password_hash(password: str) -> str:
    "Get a hashed password by providing a normal password."
    return bcrypt_context.hash(password)


def verify_password(password: str, hashed_password: str):
    "Verifies a password by checking the password and the hashed password."
    return bcrypt_context.verify(password, hashed_password)


def authenticate_user(user_id: int, password: str, dataloader: DataLoader):
    "Authenticates a user by check if the user exists and if the user - password combination is correct."
    user = dataloader.query_on_col_value(table='users', col_name='user_id', col_value=user_id)

    if user.empty:
        return False
    
    if not verify_password(password, user.hashed_passwords[0]):
        return False

    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    "Creates a JWT (access token) based on a user_id and expiration timedelta."
    encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta

    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    encode.update({"exp": expire})

    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)


async def token_exception() -> HTTPException:
    "Rasies a custom 401 status code in case the username or password is incorrect."
    token_exception_response = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"}
    )

    return token_exception_response


async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)]):
    "Gets the current user by providing a JSON Web token."
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(payload)
        user_id: int = payload.get("sub")

        if user_id is None:
            raise HTTPException(status_code=404, detail="User not found or login session expired.")
        
        return user_id
    
    except JWTError:
        raise HTTPException(status_code=404, detail="User not found or login session expired.")
    