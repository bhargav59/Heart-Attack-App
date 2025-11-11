#!/bin/bash
# API Endpoint Testing Script

BASE_URL="http://localhost:8000"

echo "========================================================================"
echo "Heart Attack Risk API - Endpoint Testing"
echo "========================================================================"
echo ""

# Test 1: Health endpoint
echo "1. Testing GET /health"
echo "   URL: $BASE_URL/health"
curl -s "$BASE_URL/health" | jq .
echo ""

# Test 2: Root endpoint (should give 404)
echo "2. Testing GET / (root - expected 404)"
echo "   URL: $BASE_URL/"
curl -s "$BASE_URL/" 
echo ""
echo ""

# Test 3: Predict endpoint
echo "3. Testing POST /predict"
echo "   URL: $BASE_URL/predict"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{
      "age": 45,
      "sex": 1,
      "cp": 2,
      "trtbps": 130,
      "chol": 250,
      "fbs": 1,
      "restecg": 1,
      "thalachh": 150,
      "exng": 0,
      "oldpeak": 1.5,
      "slp": 1,
      "caa": 1,
      "thall": 2
    }],
    "client": "test_script"
  }' | jq .
echo ""

echo "========================================================================"
echo "Available Endpoints:"
echo "========================================================================"
echo "✅ GET  /health          - Health check"
echo "✅ POST /predict         - Predict heart attack risk"
echo "✅ POST /train           - Retrain model with new data"
echo "📚 GET  /docs            - Interactive API documentation (Swagger UI)"
echo "📚 GET  /redoc           - Alternative API documentation (ReDoc)"
echo "========================================================================"
echo ""
echo "Common Issues:"
echo "❌ GET  /                - Returns 404 (root endpoint not defined)"
echo "❌ Wrong method          - Use POST for /predict and /train"
echo "❌ Missing /             - Must include leading slash in path"
echo "========================================================================"
