#!/bin/sh

./requirements.sh

HOST=$1
PORT=$2
if [ -z "$HOST" ]; then
    HOST=victim
fi

if [ -z "$PORT" ]; then
    PORT=1337
fi

python3 solve.py HOST=$HOST PORT=$PORT
