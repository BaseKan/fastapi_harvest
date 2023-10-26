import os
import json
import tensorflow as tf
import tensorflow_recommenders as tfrs

from model_api.dataloaders import DataLoader
from model_api.constants import RETRIEVAL_CHECKPOINT_PATH, MODEL_DIR
from .embedding_models import get_vocabulary_datasets, create_embedding_models


class RetrievalModel(tfrs.Model):
  def __init__(self, user_model, movie_model, movies_ds):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
      metrics=tfrs.metrics.FactorizedTopK(
        candidates=movies_ds.batch(128).map(movie_model)
      )
    )

  def compute_loss(self, features: dict[str, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model,
    # getting embeddings back.
    positive_movie_embeddings = self.movie_model(features["movie_title"])

    # The task computes the loss and the metrics.
    return self.task(user_embeddings, positive_movie_embeddings)


def init_retrieval_model_checkpoint(model_base_path: str, data_loader: DataLoader,
                                    embedding_dimension: int, learning_rate: float) -> tfrs.Model:
    users, movies = get_vocabulary_datasets(data_loader=data_loader)
    user_model, movie_model = create_embedding_models(users, movies, embedding_dimension=int(embedding_dimension))
    movies_ds = tf.data.Dataset.from_tensor_slices(dict(movies)).map(lambda x: x['movie_title'])
    retrieval_model = RetrievalModel(user_model, movie_model, movies_ds)

    retrieval_model.load_weights(filepath=os.path.join(model_base_path, RETRIEVAL_CHECKPOINT_PATH))
    retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=float(learning_rate)))

    return retrieval_model


def load_retrieval_model_checkpoint(data_loader: DataLoader, model_version: int = 0) -> tfrs.Model:
    model_version_base_path = os.path.join(MODEL_DIR, str(model_version))
    with open(os.path.join(model_version_base_path, 'training_parameters.json')) as file:
        training_parameters = json.load(file)

    embedding_dimension = training_parameters.get('embedding_dimension', 32)
    learning_rate = training_parameters.get('learning_rate', 0.1)

    return init_retrieval_model_checkpoint(model_base_path=model_version_base_path,
                                           data_loader=data_loader,
                                           embedding_dimension=embedding_dimension,
                                           learning_rate=learning_rate)
