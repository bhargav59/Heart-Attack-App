#!/bin/bash
# Stop both backend and frontend services

echo "========================================================================"
echo "Stopping Heart Attack Risk Predictor Services"
echo "========================================================================"

# Stop backend
BACKEND_PID=$(pgrep -f "uvicorn backend.main:app" || echo "")
if [ ! -z "$BACKEND_PID" ]; then
    echo "🛑 Stopping Backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID
    echo "   Backend stopped"
else
    echo "ℹ️  Backend not running"
fi

# Stop frontend
FRONTEND_PID=$(pgrep -f "streamlit run app.py" || echo "")
if [ ! -z "$FRONTEND_PID" ]; then
    echo "🛑 Stopping Frontend (PID: $FRONTEND_PID)..."
    kill $FRONTEND_PID
    echo "   Frontend stopped"
else
    echo "ℹ️  Frontend not running"
fi

echo ""
echo "✅ All services stopped"
echo ""
