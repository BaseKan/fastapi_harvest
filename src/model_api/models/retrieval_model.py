import tensorflow as tf
import tensorflow_recommenders as tfrs


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
