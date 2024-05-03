# Installation

Code has been tested with Python 3.10. Other Python versions might cause problems. 

```commandline
pip install -r requirements.txt
pip install -e .
```

# Run

```commandline
uvicorn model_api.main:app --reload
```
