"""
Streamlit Frontend for Indian Heart Attack Risk Prediction
With all 26 features from Indian dataset
"""
import streamlit as st
import requests
import json

# Configuration
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Attack Risk Prediction System")
st.markdown("### Based on Indian Population Health Data (26 Comprehensive Features)")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🩺 Risk Assessment", "📊 Model Info", "🔄 Retrain Model"])

with tab1:
    st.header("Patient Information")
    
    # Demographics
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=20, max_value=100, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    # Medical Conditions
    st.subheader("🏥 Medical Conditions")
    col1, col2, col3 = st.columns(3)
    with col1:
        diabetes = st.checkbox("Diabetes")
        hypertension = st.checkbox("Hypertension")
    with col2:
        obesity = st.checkbox("Obesity")
        heart_attack_history = st.checkbox("Previous Heart Attack")
    with col3:
        family_history = st.checkbox("Family History of Heart Disease")
    
    # Lifestyle Factors
    st.subheader("🏃 Lifestyle Factors")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        smoking = st.checkbox("Smoker")
    with col2:
        alcohol = st.checkbox("Alcohol Consumption")
    with col3:
        physical_activity = st.checkbox("Physically Active")
    with col4:
        diet_score = st.slider("Diet Quality (0-10)", 0, 10, 5)
    
    # Blood Work
    st.subheader("🩸 Blood Test Results")
    col1, col2 = st.columns(2)
    with col1:
        cholesterol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200)
        triglyceride = st.number_input("Triglycerides (mg/dL)", 30, 400, 150)
    with col2:
        ldl = st.number_input("LDL Cholesterol (mg/dL)", 30, 300, 100)
        hdl = st.number_input("HDL Cholesterol (mg/dL)", 15, 100, 50)
    
    # Blood Pressure
    st.subheader("💉 Blood Pressure")
    col1, col2 = st.columns(2)
    with col1:
        systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    with col2:
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", 50, 130, 80)
    
    # Environmental & Social
    st.subheader("🌍 Environmental & Social Factors")
    col1, col2 = st.columns(2)
    with col1:
        air_pollution = st.checkbox("Exposed to Air Pollution")
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    with col2:
        healthcare_access = st.checkbox("Has Healthcare Access")
        health_insurance = st.checkbox("Has Health Insurance")
    
    # Socioeconomic
    st.subheader("💰 Socioeconomic Factors")
    col1, col2 = st.columns(2)
    with col1:
        emergency_time = st.number_input("Emergency Response Time (minutes)", 5, 500, 15)
    with col2:
        annual_income = st.number_input("Annual Income (INR)", 10000, 3000000, 500000, step=10000)
    
    # Predict button
    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        # Prepare data
        patient_data = {
            "age": age,
            "gender": gender,
            "diabetes": int(diabetes),
            "hypertension": int(hypertension),
            "obesity": int(obesity),
            "smoking": int(smoking),
            "alcohol_consumption": int(alcohol),
            "physical_activity": int(physical_activity),
            "diet_score": diet_score,
            "cholesterol_level": cholesterol,
            "triglyceride_level": triglyceride,
            "ldl_level": ldl,
            "hdl_level": hdl,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "air_pollution_exposure": int(air_pollution),
            "family_history": int(family_history),
            "stress_level": stress_level,
            "healthcare_access": int(healthcare_access),
            "heart_attack_history": int(heart_attack_history),
            "emergency_response_time": emergency_time,
            "annual_income": annual_income,
            "health_insurance": int(health_insurance)
        }
        
        try:
            # Make API call
            with st.spinner("Analyzing patient data..."):
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    json={"patient_data": patient_data},
                    timeout=10
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Risk level with color coding
                risk_level = result['risk_level']
                risk_percent = result['risk_percent']
                
                if risk_level == "LOW RISK":
                    st.success(f"### 🟢 {risk_level}")
                    st.metric("Risk Score", f"{risk_percent:.2f}%")
                elif risk_level == "MODERATE RISK":
                    st.warning(f"### 🟡 {risk_level}")
                    st.metric("Risk Score", f"{risk_percent:.2f}%")
                else:  # HIGH RISK
                    st.error(f"### 🔴 {risk_level}")
                    st.metric("Risk Score", f"{risk_percent:.2f}%")
                
                # Probabilities
                st.subheader("Probability Breakdown")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("High Risk", f"{result['probabilities']['high']:.1%}")
                with col2:
                    st.metric("Low Risk", f"{result['probabilities']['low']:.1%}")
                
                # Recommendations
                st.subheader("📋 Recommendations")
                if risk_level == "HIGH RISK":
                    st.error("""
                    **Immediate Actions Required:**
                    - Consult a cardiologist immediately
                    - Consider cardiac screening tests
                    - Implement lifestyle changes
                    - Review medication with your doctor
                    """)
                elif risk_level == "MODERATE RISK":
                    st.warning("""
                    **Preventive Measures:**
                    - Regular health check-ups
                    - Monitor blood pressure and cholesterol
                    - Improve diet and exercise
                    - Reduce stress levels
                    """)
                else:
                    st.success("""
                    **Maintain Healthy Lifestyle:**
                    - Continue healthy habits
                    - Annual health check-ups
                    - Stay physically active
                    - Maintain balanced diet
                    """)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend server. Please ensure the backend is running.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

with tab2:
    st.header("📊 Model Information")
    
    try:
        health_response = requests.get(f"{BACKEND_URL}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", health_data.get('status', 'Unknown'))
            with col2:
                st.metric("Model Loaded", "✅" if health_data.get('model_loaded') else "❌")
            with col3:
                st.metric("Features", health_data.get('features_count', 'N/A'))
            
            st.subheader("Model Details")
            st.info(f"**Model Type:** {health_data.get('model_type', 'Unknown')}")
            
        api_response = requests.get(f"{BACKEND_URL}/")
        if api_response.status_code == 200:
            api_data = api_response.json()
            st.json(api_data)
            
    except Exception as e:
        st.error(f"Cannot fetch model info: {str(e)}")
    
    st.subheader("📚 Features Used (26 Total)")
    st.markdown("""
    **Demographics (2):** Age, Gender
    
    **Medical Conditions (5):** Diabetes, Hypertension, Obesity, Family History, Previous Heart Attack
    
    **Lifestyle (4):** Smoking, Alcohol, Physical Activity, Diet Quality
    
    **Blood Work (4):** Total Cholesterol, Triglycerides, LDL, HDL
    
    **Blood Pressure (2):** Systolic BP, Diastolic BP
    
    **Environmental (2):** Air Pollution, Stress Level
    
    **Healthcare (2):** Healthcare Access, Health Insurance
    
    **Socioeconomic (2):** Emergency Response Time, Annual Income
    
    **Plus engineered features:** Age interactions, lifestyle score, lipid risk, BP risk, etc.
    """)

with tab3:
    st.header("🔄 Retrain Model")
    st.warning("⚠️ Model retraining may take 30-60 minutes")
    
    use_smote = st.checkbox("Use SMOTE for class balancing", value=True)
    target_accuracy = st.slider("Target Accuracy (%)", 50, 100, 85) / 100
    
    if st.button("Start Training", type="primary"):
        with st.spinner("Training model... This may take a while..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/train",
                    json={
                        "dataset_path": "data/_kaggle_tmp/heart_attack_prediction_india.csv",
                        "use_smote": use_smote,
                        "target_accuracy": target_accuracy
                    },
                    timeout=3600  # 1 hour timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result['success']:
                        st.success(f"✅ Model trained successfully!")
                        st.metric("Accuracy", f"{result['accuracy']:.2%}")
                        st.metric("ROC AUC", f"{result['roc_auc']:.4f}")
                        st.info(result['message'])
                    else:
                        st.error(f"❌ Training failed: {result['message']}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Training error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Heart Attack Risk Prediction System | Indian Population Data | 26 Comprehensive Features</p>
    <p><small>For medical advice, always consult with a healthcare professional</small></p>
</div>
""", unsafe_allow_html=True)
