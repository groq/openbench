#!/bin/bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout ./nginx.key \
    -out ./nginx.crt \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=gist.github.com"
