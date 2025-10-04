#!/bin/bash

apt-get update -qq && apt-get -y install -qq cmake protobuf-compiler

pip3 install --break-system-packages -q pwntools==4.11.0 jwcrypto==1.5.6
