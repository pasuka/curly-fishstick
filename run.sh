#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run the application
panel serve grid_trading_backtest.py --show --autoreload