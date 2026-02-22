#!/bin/bash

# Install dependencies
pip install -r requirements.txt || exit 1

# Run the application
# Usage:
#   ./run.sh           -> default port 5006
#   ./run.sh 5007      -> custom port
#   PORT=5007 ./run.sh -> custom port via env
PORT="${1:-${PORT:-5006}}"
python grid_trading_backtest.py --port "$PORT"
