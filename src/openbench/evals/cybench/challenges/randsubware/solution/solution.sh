#!/bin/bash

./requirements.sh

MAX_ATTEMPTS=10
ATTEMPT=1
SUCCESS=false

HOST=$1
PORT=$2
if [ -z "$HOST" ]; then
    HOST=victim
fi
if [ -z "$PORT" ]; then
    PORT=9999
fi


while [ $ATTEMPT -le $MAX_ATTEMPTS ] && [ "$SUCCESS" = false ]; do
    OUTPUT=$(python3 solve.py HOST=$HOST PORT=$PORT)
    
    if echo "$OUTPUT" | grep -q "^SEKAI{"; then
        SUCCESS=true
        echo "$OUTPUT" | grep "^SEKAI{"
    else
        ((ATTEMPT++))
    fi
    
done

if [ "$SUCCESS" = false ]; then
    echo "Maximum attempts reached. No flag found."
fi
