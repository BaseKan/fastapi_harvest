from pydantic import BaseModel


class UserResponseModel(BaseModel):
    user_id: int
    bucketized_user_age: float
    raw_user_age: float
    user_gender: int # 1 = male, 0 = female
    user_occupation_label: int
    user_occupation_text: str
    user_zip_code: str

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "bucketized_user_age": 20.0,
                "raw_user_age": 21.0,
                "user_gender": 1,
                "user_occupation_label": 4,
                "user_occupation_text": "doctor",
                "user_zip_code": "53211"
            }
        }
