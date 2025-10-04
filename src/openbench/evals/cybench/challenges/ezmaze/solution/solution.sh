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

for ATTEMPT in {1..10}; do
  if OUTPUT=$(python3 solve.py HOST=$HOST PORT=$PORT); then
    echo "$OUTPUT"
    exit 0
  fi
done
exit 1
