#!/bin/bash
#cd /code/
PID=$(lsof -t -i:8000)
if [ "$PID" != "" ]
then
kill -9 $PID
sleep 2
echo "" > nohup.out
echo "Restarting FastAPI server"
else
echo "No such process. Starting new FastAPI server"
fi
nohup /Users/basdekan/PycharmProjects/harvest/fastapi_harvest/venv/bin/uvicorn model_api.main:app --reload &


#if [ -f api.pid ] && ps -p $(cat api.pid) > /dev/null
#then
#  PID=$(cat api.pid)
#  kill -HUP $PID
#  sleep 2
#  echo "Restarting Gunicorn server."
#else
#  echo "No running process found, starting Gunicorn server."
#fi
#python -m model_api.init_db
#uvicorn model_api.main:app --reload & disown
#gunicorn model_api.main:app -c ./gunicorn.conf.py & disown
