#!/bin/bash
cd "$(dirname "$0")"
pkill -f "app.py" || true
pkill -f "http.server 8080"

PY="python3"
if [ -x "./.venv/bin/python" ]; then
	PY="./.venv/bin/python"
fi

"$PY" app.py &
sleep 2
"$PY" -m http.server 8080 &
sleep 2
echo "OPEN: http://localhost:8080/index.html"





















