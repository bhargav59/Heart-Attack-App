#!/bin/bash

# Heart Attack Risk Predictor - Quick Start Script
# This script starts the FastAPI backend and opens the HTML frontend

echo "🚀 Starting Heart Attack Risk Predictor..."
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 8000 is already in use. Killing existing process..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    sleep 1
fi

# Activate virtual environment and start backend
echo "🔧 Starting FastAPI backend on http://localhost:8000..."
source .venv/bin/activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 3

# Check if backend started successfully
if ! ps -p $BACKEND_PID > /dev/null 2>&1; then
    echo "❌ Failed to start backend!"
    exit 1
fi

# Test backend health
echo "🏥 Checking backend health..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy!"
else
    echo "⚠️  Backend started but health check failed. Continuing anyway..."
fi

# Open HTML frontend
echo "🌐 Opening HTML frontend..."
sleep 1
open http://localhost:8000/app

echo ""
echo "✅ Application is running!"
echo ""
echo "📍 Access points:"
echo "   - HTML Frontend: http://localhost:8000/app"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "🎯 Quick Test:"
echo "   1. Click 'Fill High Risk Patient' button"
echo "   2. Click 'Predict Risk'"
echo "   3. Should see: 99.91% HIGH RISK"
echo ""
echo "⏹️  To stop the server: Press Ctrl+C or run ./stop_app.sh"
echo ""

# Keep the script running and show backend logs
echo "📋 Backend logs (Ctrl+C to stop):"
echo "─────────────────────────────────────────────────────────"
wait $BACKEND_PID
