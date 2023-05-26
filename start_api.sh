#!/bin/bash
cd /code/
if [ -f api.pid ] && ps -p $(cat api.pid) > /dev/null
then
  PID=$(cat api.pid)
  kill -HUP $PID
else
  python -m model_api.init_db
  gunicorn model_api.main:app -c ./gunicorn.conf.py
fi