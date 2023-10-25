import os


MODEL_DIR = './model_tf/'
RETRIEVAL_CHECKPOINT_PATH = os.path.join('retrieval_model', 'retrieval_model')
MODEL_NAME = 'harvest_recommender'
HYPERTUNING_EXPERIMENT_NAME = 'Recommender hyperparameter tuning'
MONITORING_EXPERIMENT_NAME = 'Model monitoring'
RETRAINING_EXPERIMENT_NAME = "Retraining"

# TODO: MUST BE DYNAMIC
MODEL_VERSION = 0
