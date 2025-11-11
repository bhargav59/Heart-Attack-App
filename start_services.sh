#!/bin/bash
# Start both backend and frontend services

set -e

echo "========================================================================"
echo "Heart Attack Risk Predictor - Full Stack Startup"
echo "========================================================================"
echo ""

# Check if services are already running
BACKEND_PID=$(pgrep -f "uvicorn backend.main:app" || echo "")
FRONTEND_PID=$(pgrep -f "streamlit run app.py" || echo "")

if [ ! -z "$BACKEND_PID" ]; then
    echo "⚠️  Backend already running (PID: $BACKEND_PID)"
else
    echo "🚀 Starting FastAPI Backend on port 8000..."
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
    BACKEND_PID=$!
    echo "   Backend started (PID: $BACKEND_PID)"
fi

sleep 3

if [ ! -z "$FRONTEND_PID" ]; then
    echo "⚠️  Frontend already running (PID: $FRONTEND_PID)"
else
    echo "🚀 Starting Streamlit Frontend on port 8501..."
    BACKEND_URL=http://localhost:8000 python -m streamlit run app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true > frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo "   Frontend started (PID: $FRONTEND_PID)"
fi

sleep 3

echo ""
echo "========================================================================"
echo "Services Status"
echo "========================================================================"

# Check backend
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API:  http://localhost:8000"
    echo "   Health:       http://localhost:8000/health"
    echo "   API Docs:     http://localhost:8000/docs"
else
    echo "❌ Backend API:  Failed to start"
fi

# Check frontend
if curl -s http://localhost:8501 > /dev/null 2>&1; then
    echo "✅ Frontend App: http://localhost:8501"
else
    echo "⏳ Frontend App: Starting... (may take a few more seconds)"
    echo "   URL:          http://localhost:8501"
fi

echo ""
echo "========================================================================"
echo "Quick Actions"
echo "========================================================================"
echo "View logs:"
echo "  Backend:  tail -f backend.log"
echo "  Frontend: tail -f frontend.log"
echo ""
echo "Stop services:"
echo "  ./stop_services.sh"
echo ""
echo "Test backend:"
echo "  curl http://localhost:8000/health"
echo "========================================================================"
echo ""
echo "🎉 Application is ready! Open http://localhost:8501 in your browser"
echo ""
