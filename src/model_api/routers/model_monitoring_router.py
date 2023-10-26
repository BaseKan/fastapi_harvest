import os
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from fastapi import APIRouter, Depends, BackgroundTasks

from model_api.dataloaders import DataLoader
from model_api.dependencies import get_data, get_full_model, get_vocabulary_dependency
from model_api.utils.restart_api import restart_api

router = APIRouter(prefix="/monitor",
                   tags=["monitor"],
                   responses={404: {"description": "Not Found"}})





@router.get("/evaluate")
async def get_model_evaluation(first_rating_id: int, last_rating_id: int,
                               data: DataLoader = Depends(get_data),
                               vocabulary: (pd.DataFrame, pd.DataFrame) = Depends(get_vocabulary_dependency),
                               retrieval_model: tfrs.Model = Depends(get_full_model)):
    _, movies = vocabulary
    ratings_df = (data.get_ratings_id_range(first_id=first_rating_id,
                                            last_id=last_rating_id)
                  .merge(movies, on='movie_id')
                  .loc[:, ['movie_title', 'user_id']]
                  .astype({'user_id': str}))

    ratings_ds = tf.data.Dataset.from_tensor_slices(dict(ratings_df)).batch(4096).cache()

    model_performance = retrieval_model.evaluate(ratings_ds, return_dict=True)

    return {"message": f"Model Performance on test data: {str(model_performance)}"}


@router.get("/reload_models")
async def reload_models(background_tasks: BackgroundTasks):
    background_tasks.add_task(restart_api)
    return {"message": "Reloading models, restarting API."}
