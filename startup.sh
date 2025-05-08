#!/bin/bash
set -e

# Check for data files
echo "Checking data directory..."
ls -la /app/data

# Start the application
echo "Starting Zwende API server..."
python main.py