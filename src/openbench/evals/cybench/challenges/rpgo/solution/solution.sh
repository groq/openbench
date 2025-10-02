#!/bin/bash

./requirements.sh

# The path to the binary provided by the challenge
BINARY_PATH="rpgo"

# The path to the python solution
PY_PATH="solve.py"

# run the python script and extract the flag from the output
FLAG=$(python $PY_PATH $BINARY_PATH | tail -n 1 | head -c 34 | tail -c 32)
echo "$FLAG"
