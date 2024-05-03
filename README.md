# Installation

Code has been tested with Python 3.10. Other Python versions might cause problems.

```commandline
pip install -r requirements.txt
pip install -e .
```

# Run

Debugging:
```commandline
uvicorn model_api.main:app --reload
```

Production:
```commandline
sh start_api.sh
```

# Run MLFlow UI
In order to run the MLFlow UI, one needs to specify the gunicorn config file.'
```commandline
mlflow ui --gunicorn-opts "-c gunicorn_mlflow.conf.py"  
```
In case the MLFlow UI server does not start, kill all processes running on port 5000:
```commandline
sudo kill -9 `sudo lsof -t -i:5000`
```

