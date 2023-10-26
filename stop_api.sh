#!/bin/bash
#cd /code/
PID=$(lsof -t -i:8000)
if [ "$PID" != "" ]
then
kill -9 $PID
sleep 2
echo "" > nohup.out
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
