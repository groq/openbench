#!/bin/bash

if [ -f /init/init.sh ]; then
    bash /init/init.sh
fi

bash /evaluator/entrypoint.sh &
EVALUATOR_PID=$!

exec "$@"

wait $EVALUATOR_PID
