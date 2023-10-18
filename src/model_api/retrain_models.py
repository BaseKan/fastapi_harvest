import os
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from model_api.models.embedding_models import get_vocabulary_datasets, process_training_data, create_embedding_models
from model_api.models.retrieval_model import RetrievalModel

MODEL_DIR = './model/'
RETRIEVAL_CHECKPOINT_PATH = os.path.join('retrieval_model', 'retrieval_model')

# See: https://www.tensorflow.org/recommenders/examples/basic_retrieval


def retrain(from_checkpoint: bool = True, epochs: int = 3,
            dataset_first_rating_id: int = 0,
            dataset_last_rating_id: int = 50000,
            embedding_dimension: int = 32,
            learning_rate: float = 0.1,
            model_version: int = 0,
            previous_checkpoint_model_version: int = 0):

    model_version_base_path = os.path.join(MODEL_DIR, str(model_version))

    users, movies = get_vocabulary_datasets()
    user_model, movie_model = create_embedding_models(users, movies, embedding_dimension=embedding_dimension)

    ratings_train, ratings_test, movies_ds = process_training_data(movies, dataset_first_rating_id,
                                                                   dataset_last_rating_id)

    retrieval_model = RetrievalModel(user_model, movie_model, movies_ds)
    retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate))

    cached_train = ratings_train.shuffle(100_000).batch(8192).cache()
    cached_test = ratings_test.batch(4096).cache()

    if from_checkpoint:
        retrieval_model.load_weights(filepath=os.path.join(MODEL_DIR, str(previous_checkpoint_model_version),
                                                           RETRIEVAL_CHECKPOINT_PATH))

    retrieval_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_version_base_path, RETRIEVAL_CHECKPOINT_PATH),
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    retrieval_model.fit(cached_train, epochs=epochs, callbacks=retrieval_checkpoint_callback)

    print(retrieval_model.evaluate(cached_test, return_dict=True))

    index = tfrs.layers.factorized_top_k.BruteForce(retrieval_model.user_model)
    # recommends movies out of the entire movies dataset.
    index.index_from_dataset(
        tf.data.Dataset.zip((movies_ds.batch(100), movies_ds.batch(100).map(retrieval_model.movie_model)))
    )
    # call once otherwise it cannot be saved
    input_features = {'user_id': tf.convert_to_tensor(
        [[x for x in ratings_test.take(1)][0]['user_id'].numpy()])}

    _, titles = index({k: np.array(v) for k, v in input_features.items()})
    print(titles)

    index.save(os.path.join(model_version_base_path, 'index'))

    with open(os.path.join(model_version_base_path, 'retrieval_training_parameters.txt'), 'w') as file:
        file.writelines([f'from_checkpoint: {from_checkpoint}\n',
                         f'epochs: {epochs}\n',
                         f'dataset_first_rating_id: {dataset_first_rating_id}\n',
                         f'dataset_last_rating_id: {dataset_last_rating_id}\n',
                         f'embedding_dimension: {embedding_dimension}\n',
                         f'learning_rate: {learning_rate}\n',
                         f'previous_checkpoint_model_version: {previous_checkpoint_model_version}\n'])


if __name__ == '__main__':
    retrain()
