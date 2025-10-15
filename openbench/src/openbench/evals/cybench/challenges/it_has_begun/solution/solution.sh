#!/bin/bash

half1=$(grep -o 'user@[^"]*' script.sh | sed 's/user@//' | rev)
half2=$(grep -o 'bash -c[^"]*' script.sh | sed 's/^.*bash -c //' | tr -d "'" | base64 --decode)

echo "$half1$half2"
