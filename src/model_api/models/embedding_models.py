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


ratings = data_loader.query_data(
    query="""Select M.movie_title, R.user_id 
             from movies M join ratings R
             on M.movie_id=R.movie_id""").astype({'user_id': str})

movies = pd.DataFrame(data_loader.get_full_table('movies').loc[:, 'movie_title'])

unique_user_ids = [str(i) for i in ratings.loc[:, 'user_id'].unique()]
unique_movie_titles = [str(i) for i in movies.loc[:, 'movie_title'].unique()]

ratings_ds = tf.data.Dataset.from_tensor_slices(dict(ratings))
movies_ds = tf.data.Dataset.from_tensor_slices(dict(movies)).map(lambda x: x['movie_title'])

ratings_train, ratings_test = train_test_split_ds(ds=ratings_ds)

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
