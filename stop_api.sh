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
echo "Stopping FastAPI server"
else
echo "No such process."
fi

#if [ -f api.pid ] && ps -p $(cat api.pid) > /dev/null
#then
#  PID=$(cat api.pid)
#  kill -HUP $PID
#  sleep 2
#  echo "Stopping Gunicorn server."
#else
#  echo "No running process found."
#fi
