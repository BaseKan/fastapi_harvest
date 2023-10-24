from enum import Enum

import tensorflow as tf

from model_api.mlflow.model_serving import load_registered_model
from model_api.dataloaders import DataLoader
from model_api.models.embedding_models import get_vocabulary_datasets

from evidently.metrics import PrecisionTopKMetric
from evidently.report import Report

if __name__ == "__main__":

    recommender_model = load_registered_model(model_name="harvest_recommender", stage="Production")

    #set reference dates
    reference_ratings = (0,50000)

    #set experiment batches dates
    experiment_batches = [
        (0,60000),
        (0,70000),
        (0,80000),
        (0,90000),  
    ]

    data = DataLoader()
    vocabulary = get_vocabulary_datasets(data_loader=data)
    _, movies = vocabulary

    ratings_df = (data.get_ratings_id_range(first_id=0,
                                                last_id=50000)
                    .merge(movies, on='movie_id')
                    .loc[:, ['movie_title', 'user_id']]
                    .astype({'user_id': str}))

    ratings_ds = tf.data.Dataset.from_tensor_slices(dict(ratings_df)).batch(4096).cache()

    predictions = recommender_model.predict({"user_id": [138]})

    print(predictions)
