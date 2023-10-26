import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from model_api.models.embedding_models import get_vocabulary_datasets, process_training_data, create_embedding_models
from model_api.models.retrieval_model import RetrievalModel
from model_api.dependencies import data_loader
from model_api.constants import MODEL_DIR, RETRIEVAL_CHECKPOINT_PATH


def retrain(from_checkpoint: bool = True, epochs: int = 3,
            dataset_first_rating_id: int = 0,
            dataset_last_rating_id: int = 50000,
            embedding_dimension: int = 32,
            learning_rate: float = 0.1,
            model_version: int = 0,
            previous_checkpoint_model_version: int = 0):

    model_version_base_path = os.path.join(MODEL_DIR, str(model_version))

    users, movies = get_vocabulary_datasets(data_loader=data_loader)
    user_model, movie_model = create_embedding_models(users=users, movies=movies,
                                                      embedding_dimension=embedding_dimension)

    ratings_train, ratings_test, movies_ds = process_training_data(data_loader=data_loader, movies=movies,
                                                                   dataset_first_rating_id=dataset_first_rating_id,
                                                                   dataset_last_rating_id=dataset_last_rating_id)

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

    evaluation_result = retrieval_model.evaluate(cached_test, return_dict=True)
    print(evaluation_result)

    index = tfrs.layers.factorized_top_k.BruteForce(retrieval_model.user_model, k=100)
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

    training_parameters_dict = {
        'from_checkpoint': from_checkpoint,
        'epochs': epochs,
        'dataset_first_rating_id': dataset_first_rating_id,
        'dataset_last_rating_id': dataset_last_rating_id,
        'embedding_dimension': embedding_dimension,
        'learning_rate': learning_rate,
        'previous_checkpoint_model_version': previous_checkpoint_model_version
    }

    with open(os.path.join(model_version_base_path, 'training_parameters.json'), 'w') as file:
        json.dump(training_parameters_dict, file, indent=4)

    with open(os.path.join(model_version_base_path, 'evaluation_result.json'), 'w') as file:
        json.dump(evaluation_result, file, indent=4)

    return training_parameters_dict, evaluation_result


if __name__ == '__main__':
    retrain()
