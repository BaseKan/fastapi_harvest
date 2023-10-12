import tensorflow as tf
import tensorflow_recommenders as tfrs


class RankingModel(tfrs.Model):
    def __init__(self, query_model: tf.keras.Model, candidate_model: tf.keras.Model, query_feature_names: list[str],
                 candidate_feature_names: list[str], label_feature_name: str) -> None:
        super().__init__()

        self.query_model: tf.keras.Model = query_model
        self.candidate_model: tf.keras.Model = candidate_model
        self.ranking_model = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)
        ])
        self.query_feature_names: list[str] = query_feature_names
        self.candidate_feature_names: list[str] = candidate_feature_names
        self.label_feature_name: str = label_feature_name
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features: dict[str, tf.Tensor]) -> tf.Tensor:
        query_embedding = self.query_model(
            {feature: features[feature] for feature in self.query_feature_names}
        )
        candidate_embedding = self.candidate_model(
            {feature: features[feature] for feature in self.candidate_feature_names}
        )

        return self.ranking_model(
            tf.concat([query_embedding, candidate_embedding], axis=1))

    def compute_loss(self, features: dict[str, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop(self.label_feature_name)

        # noinspection PyCallingNonCallable
        predictions = self(features)

        # The task computes the loss and the metrics.
        # noinspection PyCallingNonCallable
        return self.task(labels=labels, predictions=predictions)
