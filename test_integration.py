#!/usr/bin/env python3
"""
Integration Test - Verify Frontend and Backend Communication
"""
import requests
import time

print("=" * 70)
print("Frontend-Backend Integration Test")
print("=" * 70)

# Test 1: Backend Health
print("\n1. Testing Backend Health...")
try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    if response.status_code == 200:
        print("   ✅ Backend is healthy:", response.json())
    else:
        print(f"   ❌ Backend returned status {response.status_code}")
except Exception as e:
    print(f"   ❌ Backend not accessible: {e}")

# Test 2: Backend Root Endpoint
print("\n2. Testing Backend API Info...")
try:
    response = requests.get("http://localhost:8000/", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ API: {data['message']}")
        print(f"   ✅ Version: {data['version']}")
        print(f"   ✅ Model: {data['model_version']}")
    else:
        print(f"   ❌ API returned status {response.status_code}")
except Exception as e:
    print(f"   ❌ API not accessible: {e}")

# Test 3: Backend Prediction
print("\n3. Testing Backend Prediction Endpoint...")
payload = {
    "data": [{
        "age": 55, "sex": 1, "cp": 2, "trtbps": 145, "chol": 280,
        "fbs": 1, "restecg": 1, "thalachh": 140, "exng": 1,
        "oldpeak": 2.0, "slp": 1, "caa": 1, "thall": 2
    }],
    "client": "integration_test"
}
try:
    response = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
    if response.status_code == 200:
        result = response.json()["results"][0]
        print(f"   ✅ Prediction successful")
        print(f"   ✅ Risk Level: {result['risk_level']}")
        print(f"   ✅ Risk Percent: {result['risk_percent']:.2f}%")
    else:
        print(f"   ❌ Prediction failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ Prediction not accessible: {e}")

# Test 4: Frontend Accessibility
print("\n4. Testing Frontend Accessibility...")
try:
    response = requests.get("http://localhost:8501", timeout=10)
    if response.status_code == 200 and "streamlit" in response.text.lower():
        print("   ✅ Frontend is accessible")
        print("   ✅ Streamlit app is running")
    else:
        print(f"   ❌ Frontend returned unexpected response")
except Exception as e:
    print(f"   ❌ Frontend not accessible: {e}")

print("\n" + "=" * 70)
print("Integration Test Complete")
print("=" * 70)
print("\n🌐 Access Points:")
print("   Frontend: http://localhost:8501")
print("   Backend:  http://localhost:8000")
print("   API Docs: http://localhost:8000/docs")
print("\n💡 Next Steps:")
print("   1. Open http://localhost:8501 in your browser")
print("   2. Fill in patient information")
print("   3. Click 'Predict Risk' button")
print("   4. View risk assessment results")
print("=" * 70)
