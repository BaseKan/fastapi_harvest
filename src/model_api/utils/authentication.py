from typing import Annotated
from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt, JWTError

from model_api.dependencies import get_data
from model_api.dataloaders import DataLoader


SECRET_KEY = "FNE#NG#34tSNGFEWG3r323r234tfnfweGFWEF!"
ALGORITHM = "HS256"

bcrypt_context = CryptContext(schemes=["bcrypt"])

oauth2_bearer = OAuth2PasswordBearer(tokenUrl="token")


def get_password_hash(password: str) -> str:
    "Get a hashed password by providing a normal password."
    return bcrypt_context.hash(password)


def verify_password(password: str, hashed_password: str):
    "Verifies a password by checking the password and the hashed password."
    return bcrypt_context.verify(password, hashed_password)


def authenticate_user(user_id: int, password: str, db: Annotated[DataLoader, Depends(get_data)]):
    user = db.query_on_col_value(table='users', col_name='user_id', col_value=user_id)

    if user.empty:
        #TODO: custom error raise
        return False
    
    if not verify_password(password, user.hashed_password):
        return False

    return user