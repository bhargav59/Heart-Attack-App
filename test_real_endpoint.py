"""
Test Z-Alizadeh Sani Real Medical Data Endpoint
================================================

Test the /predict_real endpoint with real hospital data format.
"""

import requests
import json

# Test data - example patient with CAD risk factors
test_patient_cad = {
    # Demographics
    "Age": 67,
    "Sex": "Male",
    "Weight": 80,
    "Length": 175,
    "BMI": 26.12,
    
    # Risk Factors
    "DM": 1,  # Has diabetes
    "HTN": 1,  # Has hypertension
    "Current Smoker": 1,  # Current smoker
    "EX-Smoker": 0,
    "FH": "Y",  # Family history
    "Obesity": "Y",
    
    # Medical History
    "CRF": "N",
    "CVA": "N",
    "Airway disease": "N",
    "Thyroid Disease": "N",
    "CHF": "N",
    "DLP": "Y",  # Dyslipidemia
    
    # Vital Signs
    "BP": 150,  # Elevated BP
    "PR": 85,
    "Edema": 0,
    "Weak Peripheral Pulse": "N",
    "Lung rales": "N",
    "Systolic Murmur": "N",
    "Diastolic Murmur": "N",
    
    # Symptoms - HIGH RISK PATTERN
    "Typical Chest Pain": 1,  # Classic angina
    "Dyspnea": "Y",  # Shortness of breath
    "Function Class": 3,  # Significant limitation
    "Atypical": "N",
    "Nonanginal": "N",
    "Exertional CP": "N",
    "LowTH Ang": "N",
    
    # ECG - ABNORMAL FINDINGS
    "Q Wave": 1,  # Pathological Q wave (prior MI)
    "St Elevation": 0,
    "St Depression": 1,  # Ischemia
    "Tinversion": 1,  # T wave inversion
    "LVH": "Y",  # Left ventricular hypertrophy
    "Poor R Progression": "N",
    "BBB": "N",
    
    # Labs - ABNORMAL LIPIDS
    "FBS": 180,  # Elevated glucose
    "CR": 1.2,
    "TG": 220,  # High triglycerides
    "LDL": 160,  # High LDL
    "HDL": 32,  # Low HDL (bad)
    "BUN": 22,
    "ESR": 35,  # Elevated (inflammation)
    "HB": 14.0,
    "K": 4.3,
    "Na": 140,
    "WBC": 9500,
    "Lymph": 28,
    "Neut": 68,
    "PLT": 280,
    
    # Echo - CARDIAC DYSFUNCTION
    "EF-TTE": 38,  # Low ejection fraction (systolic dysfunction)
    "Region RWMA": 3,  # Multiple wall motion abnormalities
    "VHD": "mild"
}

# Test data - healthy patient
test_patient_normal = {
    # Demographics
    "Age": 45,
    "Sex": "Fmale",
    "Weight": 62,
    "Length": 160,
    "BMI": 24.22,
    
    # Risk Factors
    "DM": 0,
    "HTN": 0,
    "Current Smoker": 0,
    "EX-Smoker": 0,
    "FH": "N",
    "Obesity": "N",
    
    # Medical History
    "CRF": "N",
    "CVA": "N",
    "Airway disease": "N",
    "Thyroid Disease": "N",
    "CHF": "N",
    "DLP": "N",
    
    # Vital Signs
    "BP": 115,
    "PR": 70,
    "Edema": 0,
    "Weak Peripheral Pulse": "N",
    "Lung rales": "N",
    "Systolic Murmur": "N",
    "Diastolic Murmur": "N",
    
    # Symptoms - NO SYMPTOMS
    "Typical Chest Pain": 0,
    "Dyspnea": "N",
    "Function Class": 0,
    "Atypical": "N",
    "Nonanginal": "N",
    "Exertional CP": "N",
    "LowTH Ang": "N",
    
    # ECG - NORMAL
    "Q Wave": 0,
    "St Elevation": 0,
    "St Depression": 0,
    "Tinversion": 0,
    "LVH": "N",
    "Poor R Progression": "N",
    "BBB": "N",
    
    # Labs - NORMAL
    "FBS": 92,
    "CR": 0.9,
    "TG": 110,
    "LDL": 100,
    "HDL": 55,
    "BUN": 15,
    "ESR": 8,
    "HB": 13.5,
    "K": 4.1,
    "Na": 139,
    "WBC": 6500,
    "Lymph": 35,
    "Neut": 60,
    "PLT": 240,
    
    # Echo - NORMAL
    "EF-TTE": 60,
    "Region RWMA": 0,
    "VHD": "N"
}

def test_endpoint(base_url="http://localhost:8000"):
    """Test the /predict_real endpoint"""
    
    print("=" * 80)
    print("🏥 TESTING Z-ALIZADEH SANI REAL MEDICAL DATA ENDPOINT")
    print("=" * 80)
    
    # Test 1: HIGH RISK patient
    print("\n📋 TEST 1: HIGH RISK PATIENT (Multiple CAD risk factors)")
    print("-" * 80)
    print("Profile: 67yo male, DM+HTN+Smoker, Typical chest pain, Abnormal ECG, Low EF")
    
    try:
        response = requests.post(
            f"{base_url}/predict_real",
            json=test_patient_cad,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Status: {response.status_code}")
            print(f"🎯 Prediction: {'CAD (High Risk)' if result['prediction'] == 1 else 'Normal (Low Risk)'}")
            print(f"📊 Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
            print(f"⚠️  Risk Level: {result['risk_level']}")
            print(f"📈 Risk Percentage: {result['risk_percentage']:.2f}%")
            print(f"\n📦 Model Info:")
            print(f"   - Model: {result['model_info']['model_name']}")
            print(f"   - Dataset: {result['model_info']['dataset']}")
            print(f"   - Features: {result['model_info']['n_features']}")
            print(f"   - Accuracy: {result['model_info']['accuracy']*100:.2f}%")
            print(f"   - ROC AUC: {result['model_info']['roc_auc']*100:.2f}%")
            print(f"   - F1 Score: {result['model_info']['f1_score']*100:.2f}%")
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    
    # Test 2: LOW RISK patient
    print("\n" + "=" * 80)
    print("📋 TEST 2: LOW RISK PATIENT (Healthy profile)")
    print("-" * 80)
    print("Profile: 45yo female, No risk factors, No symptoms, Normal ECG, Normal EF")
    
    try:
        response = requests.post(
            f"{base_url}/predict_real",
            json=test_patient_normal,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Status: {response.status_code}")
            print(f"🎯 Prediction: {'CAD (High Risk)' if result['prediction'] == 1 else 'Normal (Low Risk)'}")
            print(f"📊 Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
            print(f"⚠️  Risk Level: {result['risk_level']}")
            print(f"📈 Risk Percentage: {result['risk_percentage']:.2f}%")
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    
    # Test 3: API root
    print("\n" + "=" * 80)
    print("📋 TEST 3: API ROOT ENDPOINT")
    print("-" * 80)
    
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Status: {response.status_code}")
            print(f"📝 Message: {result['message']}")
            print(f"🔖 Version: {result['version']}")
            print(f"\n📍 Available Endpoints:")
            for endpoint, description in result['endpoints'].items():
                print(f"   - {endpoint}: {description}")
            print(f"\n🏆 Recommended: {result.get('recommended_endpoint', 'N/A')}")
        else:
            print(f"\n❌ Error: {response.status_code}")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    
    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE")
    print("=" * 80)
    print("""
Next Steps:
1. Review the predictions - HIGH RISK patient should show >70% probability
2. Check model_info for performance metrics (86.89% accuracy, 92.38% ROC AUC)
3. Compare with /predict_indian endpoint (synthetic data, 69% accuracy)
4. Integrate /predict_real into your frontend for production use
    """)

if __name__ == "__main__":
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_endpoint(base_url)
