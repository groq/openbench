#!/bin/sh

socat tcp-listen:1337,fork,reuseaddr,su=nobody exec:/challenge/nettools 2>/dev/null
