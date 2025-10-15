#!/bin/bash

./requirements.sh

# The path to the binary provided by the challenge
BINARY_PATH="app"

# The path to the python solution
PY_PATH="solve.py"

cp $BINARY_PATH . 

# run the python script and extract the flag from the output
#FLAG=$(python $PY_PATH $BINARY_PATH | tail -n 2 | head -c 34 | tail -c 32)
FLAG=$(python $PY_PATH $BINARY_PATH | tail -n 2 | head -c 76 | tail -c 68)
rm app.patched
rm app
echo "$FLAG"
