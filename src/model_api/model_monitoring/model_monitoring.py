from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import mlflow
from mlflow.tracking import MlflowClient
import tensorflow as tf

from model_api.mlflow.model_serving import load_registered_model
from model_api.dataloaders import DataLoader
from model_api.models.embedding_models import get_vocabulary_datasets

from evidently.metrics import PrecisionTopKMetric
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report


def monitor_model(model_name: str):
    recommender_model = load_registered_model(model_name=model_name, stage="Production")

    #set reference dates
    reference_ratings = (0,50000)

    #set experiment batches dates
    experiment_batches = [
        (0,60000),
        (0,70000),
        (0,80000),
        (0,90000),  
    ]

    data = DataLoader()
    vocabulary = get_vocabulary_datasets(data_loader=data)
    _, movies = vocabulary

    # Get performance on reference dataset
    # ---------------------------------------------------------------------------------------------------
    ratings_df_ref = (data.get_ratings_id_range(first_id=0,
                                                last_id=50000)
                    .merge(movies, on='movie_id')
                    .loc[:, ['movie_title', 'user_id']]
                    .astype({'user_id': str}))
    
    ratings_ds_ref = tf.data.Dataset.from_tensor_slices(dict(ratings_df_ref)).batch(4096).cache()
    model_performance_ref = recommender_model.evaluate(ratings_ds_ref, return_dict=True)

    print(f"Model performance on reference dataset: {str(model_performance_ref)}")

    # ---------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    monitor_model(model_name="harvest_recommender")



# #start new run
# for batch in experiment_batches:
#     with mlflow.start_run() as run:
        
#         # Log parameters
#         mlflow.log_param("begin", batch[0])
#         mlflow.log_param("end", batch[1])

#         # Log metrics


#         print(run.info)