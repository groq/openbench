#!/bin/bash

./requirements.sh

HOST=$1
PORT=$2
if [ -z "$HOST" ]; then
    HOST=victim
fi
if [ -z "$PORT" ]; then
    PORT=1337
fi

echo "$(python3 solve_zafirr.py HOST=$HOST PORT=$PORT)"
