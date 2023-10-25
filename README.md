# Installation

```commandline
pip install -r requirements.txt
pip install -e .
```

# Run

```commandline
uvicorn model_api.main:app --reload
```

# Run met docker
Ga naar de directory met dockerfile

Run:
```commandline
docker build -t harvest_api .
```

Run:
```commandline
mkdir logs
docker run -d --name harvest_api_container -p 80:80 -v /path/to/repo/logs:/code/logs harvest_api
```

# Run MLFlow UI
In order to run the MLFlow UI, one needs to specify the gunicorn config file.'
```commandline
mlflow ui --gunicorn-opts "-c gunicorn_mlflow.conf.py"  
```
  
# Serve model 
In order to serve a registred model:
```commandline
mlflow models serve -m runs:/<RUN_ID>/model_name -h 0.0.0.0 -p <PORT> --env-manager=local
mlflow models serve -m runs:/2e9f42abdd1747ef94ac77937d686b58/model -h 0.0.0.0 -p 1234 --no-conda
```

# Steps in MLOps
1. Hypertune model and log with MLFlow (Tracking API)
2. Register best performing model on test set with MLFlow (Model Registry API)
3. Load latest model for the FastAPI from Production Stage with MLFlow
4. Perform model monitoring with MLFlow, looping over different batches
5. Retrain if performance of current model hits a threshold
6. Register new (stateful) trained model if it outperforms the current model on a test set
7. Restart FastAPI