#!/bin/bash

chmod +x /root/requirements.sh
./requirements.sh

HOST=$1
PORT=$2

if [ -z "$HOST" ]; then
    HOST=victim
fi

if [ -z "$PORT" ]; then
    PORT=1337
fi

python3 solver.py HOST=$HOST PORT=$PORT
