#!/bin/bash
#cd /code/
if [ -f api.pid ] && ps -p $(cat api.pid) > /dev/null
then
  PID=$(cat api.pid)
  kill -HUP $PID
  sleep 2
  echo "Stoppin Gunicorn server."
else
  echo "No running process found."
fi
