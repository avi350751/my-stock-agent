#!/bin/bash
# Double-click to start the stockdesk single-stock check page at http://localhost:8765
cd "$(dirname "$0")"
LOG=".stockdesk-serve.log"
mkdir -p .stockdesk
echo "== stockdesk launcher $(date) ==" | tee "$LOG"

# Find Python 3.10+
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PY="$(command -v "$c")"; break
  fi
done
if [ -z "$PY" ]; then
  echo "ERROR: Python 3.10 or newer not found. Install it from https://www.python.org/downloads/ or 'brew install python'." | tee -a "$LOG"
  read -n 1 -s -r -p "Press any key to close"; exit 1
fi
echo "Using $PY ($("$PY" --version))" | tee -a "$LOG"

if [ ! -x .venv/bin/stockdesk ]; then
  echo "First run: creating .venv and installing required libraries (takes a minute or two)..." | tee -a "$LOG"
  "$PY" -m venv .venv 2>&1 | tee -a "$LOG"
  .venv/bin/python -m pip install -q --upgrade pip 2>&1 | tee -a "$LOG"
  .venv/bin/python -m pip install -q -r requirements.txt 2>&1 | tee -a "$LOG"
  .venv/bin/python -m pip install -q --no-deps . 2>&1 | tee -a "$LOG"
fi
if [ ! -x .venv/bin/stockdesk ]; then
  echo "ERROR: install failed - see $LOG" | tee -a "$LOG"
  read -n 1 -s -r -p "Press any key to close"; exit 1
fi

# Free the port if an old copy is still running
lsof -ti tcp:8765 2>/dev/null | xargs kill 2>/dev/null

echo "READY: starting server" | tee -a "$LOG"
PYTHONPATH="$PWD/src" .venv/bin/stockdesk serve --port 8765 2>&1 | tee -a "$LOG"
