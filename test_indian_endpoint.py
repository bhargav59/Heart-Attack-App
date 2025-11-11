"""
Test the new /predict_indian endpoint with sample data
"""
import requests
import json

# Sample patient data (all 23 required features)
test_patient = {
    "patient_data": {
        "age": 55,
        "gender": "Male",
        "diabetes": 1,
        "hypertension": 1,
        "obesity": 0,
        "smoking": 1,
        "alcohol_consumption": 0,
        "physical_activity": 0,
        "diet_score": 4,
        "cholesterol_level": 250,
        "triglyceride_level": 180,
        "ldl_level": 160,
        "hdl_level": 35,
        "systolic_bp": 145,
        "diastolic_bp": 95,
        "air_pollution_exposure": 1,
        "family_history": 1,
        "stress_level": 8,
        "healthcare_access": 1,
        "heart_attack_history": 0,
        "emergency_response_time": 25,
        "annual_income": 500000,
        "health_insurance": 1
    }
}

def test_endpoint():
    print("=" * 80)
    print("Testing Indian Heart Attack Prediction API")
    print("=" * 80)
    
    # Start backend first if not running
    print("\n📋 Test Patient Profile:")
    print(f"   Age: {test_patient['patient_data']['age']}")
    print(f"   Gender: {test_patient['patient_data']['gender']}")
    print(f"   Diabetes: {'Yes' if test_patient['patient_data']['diabetes'] else 'No'}")
    print(f"   Hypertension: {'Yes' if test_patient['patient_data']['hypertension'] else 'No'}")
    print(f"   Smoking: {'Yes' if test_patient['patient_data']['smoking'] else 'No'}")
    print(f"   Cholesterol: {test_patient['patient_data']['cholesterol_level']} mg/dL")
    print(f"   Systolic BP: {test_patient['patient_data']['systolic_bp']} mmHg")
    
    print("\n🔍 Sending request to /predict_indian...")
    
    try:
        response = requests.post(
            "http://localhost:8000/predict_indian",
            json=test_patient,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS!")
            print(f"\n📊 Prediction Results:")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Risk Percentage: {result['risk_percent']:.2f}%")
            print(f"   Model Version: {result['model_version']}")
            print(f"\n   Probabilities:")
            print(f"      Low Risk:  {result['probabilities']['low']*100:.2f}%")
            print(f"      High Risk: {result['probabilities']['high']*100:.2f}%")
            
            print("\n⚠️  NOTE: This dataset appears to be synthetic/simulated.")
            print("   Real medical data would show stronger predictive patterns.")
            
        else:
            print(f"\n❌ ERROR: Status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to backend.")
        print("   Please start the backend first:")
        print("   uvicorn backend.main:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_endpoint()
