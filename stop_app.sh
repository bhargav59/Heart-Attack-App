#!/bin/bash

# Heart Attack Risk Predictor - Stop Script
# This script stops the FastAPI backend

echo "⏹️  Stopping Heart Attack Risk Predictor..."

# Kill process on port 8000
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "🔍 Found process on port 8000, stopping..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    sleep 1
    echo "✅ Backend stopped successfully!"
else
    echo "ℹ️  No process found on port 8000"
fi

# Also kill any uvicorn processes
if pgrep -f "uvicorn backend.main:app" > /dev/null 2>&1; then
    echo "🔍 Found uvicorn processes, stopping..."
    pkill -f "uvicorn backend.main:app"
    sleep 1
    echo "✅ All uvicorn processes stopped!"
fi

echo "✅ Application stopped!"
