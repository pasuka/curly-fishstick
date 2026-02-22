#!/bin/bash

# Install dependencies
pip install -r requirements.txt || exit 1

# Run the application
panel serve grid_trading_backtest.py --show --autoreload
