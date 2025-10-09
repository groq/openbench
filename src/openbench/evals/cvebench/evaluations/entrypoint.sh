#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

gunicorn -b 0.0.0.0:9091 evaluator:app &
EVALUATOR_PID=$!
gunicorn -b 0.0.0.0:8000 outbound_server:app &
OUTBOUND_PID=$!

wait $OUTBOUND_PID $EVALUATOR_PID
