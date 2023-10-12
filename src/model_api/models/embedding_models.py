import tensorflow as tf
import pandas as pd

from model_api.dataloaders import DataLoader

EMBEDDING_DIMENSION = 32
data_loader = DataLoader()


def train_test_split_ds(ds, train_split=0.9, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)

    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)

    return train_ds, test_ds


def get_vocabulary_datasets():
    users = pd.DataFrame(data_loader.get_full_table('users').loc[:, 'user_id'])
    movies = pd.DataFrame(data_loader.get_full_table('movies').loc[:, ['movie_id', 'movie_title']])

    return users, movies


def process_training_data(movies, dataset_first_rating_id: int = 0,
                          dataset_last_rating_id: int = 50000):
    ratings = (data_loader.get_ratings_id_range(first_id=dataset_first_rating_id,
                                                last_id=dataset_last_rating_id)
               .merge(movies, on='movie_id')
               .loc[:, ['movie_title', 'user_id']]
               .astype({'user_id': str}))

    ratings_ds = tf.data.Dataset.from_tensor_slices(dict(ratings))
    movies_ds = tf.data.Dataset.from_tensor_slices(dict(movies)).map(lambda x: x['movie_title'])

    ratings_train, ratings_test = train_test_split_ds(ds=ratings_ds)

    return ratings_train, ratings_test, movies_ds


def create_embedding_models(users, movies):
    unique_user_ids = [str(i) for i in users.loc[:, 'user_id'].unique()]
    unique_movie_titles = [str(i) for i in movies.loc[:, 'movie_title'].unique()]

    user_model = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
          vocabulary=unique_user_ids, mask_token=None),
      # We add an additional embedding to account for unknown tokens.
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, EMBEDDING_DIMENSION)
    ])

    movie_model = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
          vocabulary=unique_movie_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, EMBEDDING_DIMENSION)
    ])

    return user_model, movie_model
