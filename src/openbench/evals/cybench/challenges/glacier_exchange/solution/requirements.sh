#!/bin/bash
apt update && apt install -y unzip

pip install --break-system-packages -r requirements.txt
