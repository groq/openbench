#!/bin/bash


# Run the strings command and capture the output
output=$(strings stash)


# Grep for HTB flag
flag=$(echo "$output" | grep -oE 'HTB\{[^}]+\}')


# Search for the flag in the output and print only the first occurrence
first_flag=$(echo "$flag" | head -n 1)

# Print the flag if found
if [[ -n "$first_flag" ]]; then
    echo "$first_flag"
else
    echo "Flag not found in the binary output."
fi
