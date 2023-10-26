import os


MODEL_DIR = './model_tf/'
RETRIEVAL_CHECKPOINT_PATH = os.path.join('retrieval_model', 'retrieval_model')
DEFAULT_MODEL_VERSION = 0
MODEL_NAME = 'harvest_recommender'
HYPERTUNING_EXPERIMENT_NAME = 'Recommender hyperparameter tuning'
MONITORING_EXPERIMENT_NAME = 'Model monitoring'
RETRAINING_EXPERIMENT_NAME = "Retraining"
MODEL_METRIC = "top_100_categorical_accuracy"
