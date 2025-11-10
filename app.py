import os
import json
import streamlit as st
import pandas as pd
import requests

# Backend API configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")
st.title("🫀 Heart Attack Risk Predictor — Full-stack")
st.write("Enter patient details to assess heart attack risk. This app now uses a backend API for predictions and supports retraining on Indian datasets.")

# Input fields
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trtbps = st.slider("Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("High Blood Sugar (> 120 mg/dl)", ["No", "Yes"])
    restecg = st.selectbox("Heart Rhythm Test", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])

with col2:
    thalachh = st.slider("Max Heart Rate During Exercise", 70, 200, 150)
    exng = st.selectbox("Chest Pain During Exercise", ["No", "Yes"])
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 0.0, step=0.1)
    slp = st.selectbox("ST Segment Slope", ["Upsloping", "Flat", "Downsloping"])
    caa = st.slider("Blocked Vessels (0-3)", 0, 3, 0)
    thall = st.selectbox("Thallium Scan", ["Normal", "Fixed Defect", "Reversible Defect"])

# Map inputs to numerical values (matching training dataset encoding)
sex_map = {"Male": 1, "Female": 0}
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
fbs_map = {"No": 0, "Yes": 1}
restecg_map = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
exng_map = {"No": 0, "Yes": 1}
slp_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thall_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

# Create input data with correct feature order matching the trained model
# Feature order: ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']
input_data = pd.DataFrame([[age, sex_map[sex], cp_map[cp], trtbps, chol, fbs_map[fbs], 
                           restecg_map[restecg], thalachh, exng_map[exng], oldpeak, 
                           slp_map[slp], caa, thall_map[thall]]],
                         columns=["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", 
                                 "thalachh", "exng", "oldpeak", "slp", "caa", "thall"])

if st.button("Predict Risk", type="primary"):
    # Prepare payload for backend API
    payload = {
        "data": [{
            "age": int(age),
            "sex": int(sex_map[sex]),
            "cp": int(cp_map[cp]),
            "trtbps": int(trtbps),
            "chol": int(chol),
            "fbs": int(fbs_map[fbs]),
            "restecg": int(restecg_map[restecg]),
            "thalachh": int(thalachh),
            "exng": int(exng_map[exng]),
            "oldpeak": float(oldpeak),
            "slp": int(slp_map[slp]),
            "caa": int(caa),
            "thall": int(thall_map[thall])
        }]
    }

    try:
        resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=20)
        if resp.status_code != 200:
            st.error(f"Prediction failed: {resp.status_code} {resp.text}")
            st.stop()
        data = resp.json()
        result = data["results"][0]
        risk_percent = result["risk_percent"]
        probability = [result["probabilities"]["high"], result["probabilities"]["low"]]
    except requests.RequestException as e:
        st.error(f"Could not reach backend at {BACKEND_URL}. Start the API server and try again. Error: {e}")
        st.stop()
    
    # Identify risk factors for explanation
    risk_factors = []

    # Major clinical risk factors
    if fbs == "Yes":
        risk_factors.append("Diabetes (High Blood Sugar)")

    if caa >= 2:
        risk_factors.append(f"Multiple Blocked Vessels ({caa})")
    elif caa == 1:
        risk_factors.append(f"One Blocked Vessel")

    if exng == "Yes":
        risk_factors.append("Exercise-Induced Chest Pain")

    if cp == "Typical Angina":
        risk_factors.append("Typical Angina")
    elif cp == "Atypical Angina":
        risk_factors.append("Atypical Angina")

    if oldpeak > 2.0:
        risk_factors.append(f"Significant ST Depression ({oldpeak})")
    elif oldpeak > 1.0:
        risk_factors.append(f"Mild ST Depression ({oldpeak})")

    if thall == "Reversible Defect":
        risk_factors.append("Reversible Thallium Defect")
    elif thall == "Fixed Defect":
        risk_factors.append("Fixed Thallium Defect")

    # Age and other factors
    if age > 65:
        risk_factors.append(f"Age Over 65 ({age} years)")

    if trtbps > 140:
        risk_factors.append(f"High Blood Pressure ({trtbps})")

    if chol > 240:
        risk_factors.append(f"High Cholesterol ({chol})")
    
    st.header("Results")
    
    # Determine risk level based on corrected model prediction
    if risk_percent >= 70:
        final_risk = "HIGH RISK"
        risk_emoji = "🔴"
        recommendation = "Consult a healthcare professional immediately for evaluation and possible intervention."
    elif risk_percent >= 40:
        final_risk = "MODERATE RISK"
        risk_emoji = "🟡"
        recommendation = "Monitor closely and discuss with your healthcare provider."
    else:
        final_risk = "LOW RISK"
        risk_emoji = "🟢"
        recommendation = "Continue maintaining a healthy lifestyle."
    
    # Display result
    if final_risk == "HIGH RISK":
        st.error(f"{risk_emoji} {final_risk}: {risk_percent:.1f}%")
    elif final_risk == "MODERATE RISK":
        st.warning(f"{risk_emoji} {final_risk}: {risk_percent:.1f}%")
    else:
        st.success(f"{risk_emoji} {final_risk}: {risk_percent:.1f}%")
    
    st.write(recommendation)
    
    # Show risk factors
    if risk_factors:
        st.subheader("Identified Risk Factors:")
        for factor in risk_factors:
            st.write(f"• {factor}")
    
    # Show model details (with corrected interpretation)
    st.subheader("Risk Assessment Details:")
    st.write(f"**Model Prediction:** {risk_percent:.1f}% chance of heart disease")
    st.write(f"**Low Risk Probability:** {probability[1]*100:.1f}%")  # Class 1 = Low Risk
    st.write(f"**High Risk Probability:** {probability[0]*100:.1f}%")  # Class 0 = High Risk
    
    # Medical disclaimer
    st.info("**Note:** This is a screening tool only. Always consult healthcare professionals for medical advice.")

# Simple sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This tool predicts heart attack risk using a machine learning model trained on clinical data.

    **How it works:**
    - Uses 13 clinical features including age, gender, chest pain type, blood pressure, cholesterol, and cardiac test results
    - Trained on heart disease dataset with logistic regression
    - Provides probability-based risk assessment

    **Risk Levels:**
    - Low Risk (< 40%): Continue healthy lifestyle
    - Moderate Risk (40-69%): Monitor and discuss with provider
    - High Risk (≥ 70%): Seek medical evaluation

    **Key Risk Factors:**
    - Age (especially >65)
    - Diabetes (High Blood Sugar)
    - Multiple Blocked Vessels  
    - Exercise-Induced Chest Pain
    - ST Depression on ECG
    - Abnormal Thallium Scan Results
    - High Blood Pressure (>140)
    - High Cholesterol (>240)

    **Model Performance:**
    - Properly identifies low risk in young healthy individuals
    - Accurately detects high risk in elderly patients with multiple conditions
    - Accounts for gender differences in heart disease presentation

    **Disclaimer:** This is for educational purposes only. Always consult healthcare professionals for medical advice.
    """)

    st.divider()
    st.subheader("Train on Indian Dataset")
    st.write("Upload a CSV with Indian cohort data matching the 13-feature schema to retrain the model.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Must contain columns: age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall,target")
    if uploaded is not None:
        # Save to data/ and trigger backend training
        try:
            os.makedirs("data", exist_ok=True)
            save_path = os.path.abspath(os.path.join("data", uploaded.name))
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.info(f"Saved dataset to {save_path}")
            # Call backend /train
            train_payload = {"dataset_path": save_path, "target_column": "target"}
            resp = requests.post(f"{BACKEND_URL}/train", json=train_payload, timeout=120)
            if resp.status_code == 200:
                tr = resp.json()
                st.success(f"Training complete. Model {tr['model_version']} | Accuracy: {tr['metrics'].get('accuracy', 0):.3f}")
            else:
                st.error(f"Training failed: {resp.status_code} {resp.text}")
        except Exception as e:
            st.error(f"Error handling uploaded file: {e}")

