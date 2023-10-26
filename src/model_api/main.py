import os
import logging
import uvicorn

from fastapi import FastAPI

from model_api.routers import user_data_router, movie_data_router, model_monitoring_router
from model_api.dependencies import lifespan

if not os.path.isdir("logs"):
    os.mkdir("logs")

logging.basicConfig(filename="./logs/logfile.log",
                    filemode="a",
                    format="%(levelname)s %(asctime)s - %(message)s",
                    level=logging.INFO)

logger = logging.getLogger()

app = FastAPI(lifespan=lifespan)

app.include_router(user_data_router.router)
app.include_router(movie_data_router.router)
app.include_router(model_monitoring_router.router)


@app.get("/")
async def root():
    return {"message": "Welcome to this simple API."}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
