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
docker run -d --name harvest_api_container -p 80:80 harvest_api
```
