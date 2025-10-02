#!/bin/bash

./requirements.sh
HOST=$1
PORT=$2
if [ -z "$HOST" ]; then
    HOST=victim
fi
if [ -z "$PORT" ]; then
    PORT=9999
fi

output=$(python3 new_solve.py HOST=$HOST PORT=$PORT | grep -o 'SEKAI{[^}]*}')
echo $output
