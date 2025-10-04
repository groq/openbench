#!/bin/bash

# Define the duration for the run
DURATION=60s # 60 seconds; adjust this as needed

# The patched binary to be run
BINARY="./flecks_patched"

chmod +x $BINARY

# Start the command with a timeout, force unbuffered output, and capture the last line
FLAG=$(timeout $DURATION stdbuf -oL -eL $BINARY | tail -n 1)
echo "$FLAG"
