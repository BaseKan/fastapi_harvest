import pandas as pd
import tensorflow as tf
from fastapi import APIRouter, Depends

from model_api.dataloaders import DataLoader
from model_api.dependencies import get_data
from model_api.models.embedding_models import get_vocabulary_datasets, create_embedding_models
from model_api.models.retrieval_model import RetrievalModel
from model_api.retrain_models import RETRIEVAL_CHECKPOINT_PATH

router = APIRouter(prefix="/monitor",
                   tags=["monitor"],
                   responses={404: {"description": "Not Found"}})


@router.get("/evaluate")
async def get_model_evaluation(first_rating_id: int, last_rating_id: int,
                               data: DataLoader = Depends(get_data)):
    movies = pd.DataFrame(data.get_full_table('movies').loc[:, ['movie_id', 'movie_title']])
    ratings_df = (data.get_ratings_id_range(first_id=first_rating_id,
                                            last_id=last_rating_id)
                  .merge(movies, on='movie_id')
                  .loc[:, ['movie_title', 'user_id']]
                  .astype({'user_id': str}))

    ratings_ds = tf.data.Dataset.from_tensor_slices(dict(ratings_df)).batch(4096).cache()

    users, movies = get_vocabulary_datasets()
    user_model, movie_model = create_embedding_models(users, movies)
    movies_ds = tf.data.Dataset.from_tensor_slices(dict(movies)).map(lambda x: x['movie_title'])
    retrieval_model = RetrievalModel(user_model, movie_model, movies_ds)
    retrieval_model.load_weights(filepath=RETRIEVAL_CHECKPOINT_PATH)
    retrieval_model.compile()

    model_performance = retrieval_model.evaluate(ratings_ds, return_dict=True)

    return {"message": f"Model Performance on test data: {str(model_performance)}"}

