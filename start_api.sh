#!/bin/bash
#cd /code/
PID=$(ps aux | grep 'uvicorn model_api.main:app --reload' | grep -v grep | awk {'print $2'} | xargs)
if [ "$PID" != "" ]
then
kill -9 $PID
PID=$(ps aux | grep 'venv/bin/python -c from multiprocessing' | grep -v grep | awk {'print $2'} | xargs)
if [ "$PID" != "" ]
then
kill -9 $PID
fi
sleep 2
echo "" > nohup.out
echo "Restarting FastAPI server"
else
echo "No such process. Starting new FastAPI server"
fi
nohup uvicorn model_api.main:app --reload &


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
