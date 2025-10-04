#!/bin/bash
./requirements.sh >/dev/null

PREFIX="SEKAI"

for ATTEMPT in {1..5}; do
  OUTPUT=$(python3 solve.py)
  echo "$OUTPUT"
  if FLAG=$(echo "$OUTPUT" | grep -i -q "${PREFIX}{[^}]*}"); then
    exit 0
  fi
done
exit 1
