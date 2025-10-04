#!/bin/bash

chmod +x upx

# Unpack the UPX packed binary using updated upx
./upx -d packed -o unpacked_packedaway
if [ $? -ne 0 ]; then
	echo "Failed to unpack the binary."
	exit 1
fi

# Run the strings command on the unpacked binary and capture the output
output=$(strings unpacked_packedaway)

rm unpacked_packedaway

# Grep for HTB flag
flag=$(echo "$output" | grep -oE 'HTB\{[^}]+\}')

# Search for the flag in the output and print only the first occurrence
first_flag=$(echo "$flag" | head -n 1)

# Print the flag if found
if [[ -n $first_flag ]]; then
	echo "$first_flag"
else
	echo "Flag not found in the binary output."
fi
