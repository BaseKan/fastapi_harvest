from model_api.mlops.hyperparameter_tuning import tune_hyperparams
from model_api.mlops.model_serving import register_best_model
from model_api.constants import MODEL_NAME, HYPERTUNING_EXPERIMENT_NAME, MODEL_METRIC


if __name__ == "__main__":

    tune_hyperparams(dataset_first_rating_id=0,
                     dataset_last_rating_id=50000,
                     epochs=5)
    
    register_best_model(model_name=MODEL_NAME, 
                        experiment_name=HYPERTUNING_EXPERIMENT_NAME, 
                        metric=MODEL_METRIC)
    