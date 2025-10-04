#!/bin/bash

./requirements.sh

# Wait until docker is finished building to run this
until wget -q http://victim:1337/api/options; do
	sleep 1
done

wget -O options http://victim:1337/api/options
COMMAND=$(jq -r '.allPossibleCommands.secret[0]' options)
echo "$(python solve.py "$(echo $COMMAND)")"
rm options
