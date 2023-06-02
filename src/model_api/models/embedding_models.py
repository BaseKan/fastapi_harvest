import tensorflow as tf

from model_api.dependencies import data_loader

EMBEDDING_DIMENSION = 32


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

movies = data_loader.get_full_table('movies').loc[:, 'movie_title']

unique_user_ids = [str(i) for i in ratings.loc[:, 'user_id'].unique()]
unique_movie_titles = [str(i) for i in movies.unique()]

ratings_ds = tf.data.Dataset.from_tensor_slices(ratings.to_dict('list'))
movies_ds = tf.data.Dataset.from_tensor_slices(list(movies))

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
