#!/usr/bin/env bash

pid=$(pgrep -f "app.py")

if [ -n "$pid" ]; then
    kill $pid
fi

(. /app/lollms-webui/env/bin/activate && python /app/lollms-webui/app.py --host 0.0.0.0 --port 9090)
