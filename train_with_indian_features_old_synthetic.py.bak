"""
Advanced Model Training with Indian Dataset Features
====================================================
Uses all 26 original features from Indian heart attack dataset including:
- Lifestyle: Smoking, Alcohol, Physical Activity, Diet
- Medical: Diabetes, Hypertension, Obesity, Cholesterol, BP
- Environmental: Air Pollution, Stress Level
- Socioeconomic: Income, Healthcare Access, Insurance
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# SMOTE for class balancing
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADVANCED MODEL TRAINING WITH INDIAN DATASET FEATURES")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    DATA_PATH = 'data/_kaggle_tmp/heart_attack_prediction_india.csv'
    MODEL_DIR = Path('models')
    
    # Features to use (drop ID, State, and target)
    DROP_FEATURES = ['Patient_ID', 'State_Name', 'Heart_Attack_Risk']
    
    # Features to keep as-is (already encoded or numeric)
    KEEP_FEATURES = [
        # Demographics
        'Age', 'Gender',
        
        # Medical Conditions (Binary)
        'Diabetes', 'Hypertension', 'Obesity',
        
        # Lifestyle (Binary/Numeric)
        'Smoking', 'Alcohol_Consumption', 'Physical_Activity', 'Diet_Score',
        
        # Blood Work (Continuous)
        'Cholesterol_Level', 'Triglyceride_Level', 'LDL_Level', 'HDL_Level',
        
        # Blood Pressure (Continuous)
        'Systolic_BP', 'Diastolic_BP',
        
        # Environmental & Social
        'Air_Pollution_Exposure', 'Family_History', 'Stress_Level',
        
        # Healthcare
        'Healthcare_Access', 'Heart_Attack_History',
        
        # Socioeconomic
        'Emergency_Response_Time', 'Annual_Income', 'Health_Insurance'
    ]
    
    # Feature groups for interaction features
    LIFESTYLE_FEATURES = ['Smoking', 'Alcohol_Consumption', 'Physical_Activity', 'Diet_Score']
    BLOOD_FEATURES = ['Cholesterol_Level', 'Triglyceride_Level', 'LDL_Level', 'HDL_Level']
    BP_FEATURES = ['Systolic_BP', 'Diastolic_BP']
    RISK_FACTORS = ['Diabetes', 'Hypertension', 'Obesity', 'Family_History', 'Heart_Attack_History']
    
    # Training
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # SMOTE
    USE_SMOTE = True
    SMOTE_RATIO = 0.8  # Target 80% minority class
    
    # Target accuracy
    TARGET_ACCURACY = 0.85

config = Config()
config.MODEL_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("\n[1] Loading Indian Heart Attack Dataset...")
df = pd.read_csv(config.DATA_PATH)
print(f"   ✅ Loaded {len(df):,} records with {len(df.columns)} features")
print(f"   Target distribution: {df['Heart_Attack_Risk'].value_counts().to_dict()}")

# Separate features and target
X = df.drop(columns=config.DROP_FEATURES)
y = df['Heart_Attack_Risk']

print(f"\n   Features: {list(X.columns)}")
print(f"   Feature count: {len(X.columns)}")

# Encode Gender (only categorical feature)
print("\n[2] Encoding categorical features...")
label_encoder = LabelEncoder()
X['Gender'] = label_encoder.fit_transform(X['Gender'])
print(f"   ✅ Gender encoded: Female=0, Male=1")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n[3] Creating interaction features...")

# Age-based interactions
X['age_x_cholesterol'] = X['Age'] * X['Cholesterol_Level']
X['age_x_bp'] = X['Age'] * X['Systolic_BP']
X['age_x_stress'] = X['Age'] * X['Stress_Level']

# Lifestyle score (composite)
X['lifestyle_score'] = (
    X['Smoking'] * 0.3 + 
    X['Alcohol_Consumption'] * 0.2 + 
    (1 - X['Physical_Activity']) * 0.3 +  # Inverse: low activity = high risk
    (10 - X['Diet_Score']) * 0.2  # Inverse: low diet score = high risk
)

# Blood lipid risk score
X['lipid_risk'] = (
    (X['Cholesterol_Level'] / 300) * 0.25 +
    (X['Triglyceride_Level'] / 300) * 0.25 +
    (X['LDL_Level'] / 200) * 0.3 +
    (1 - X['HDL_Level'] / 80) * 0.2  # Inverse: low HDL = high risk
)

# Blood pressure risk
X['bp_risk'] = (
    (X['Systolic_BP'] - 90) / 90 * 0.6 +
    (X['Diastolic_BP'] - 60) / 60 * 0.4
)

# Cumulative risk factors
X['total_risk_factors'] = (
    X['Diabetes'] + X['Hypertension'] + X['Obesity'] + 
    X['Smoking'] + X['Family_History'] + X['Heart_Attack_History']
)

# Healthcare quality score
X['healthcare_quality'] = (
    X['Healthcare_Access'] * 0.4 +
    X['Health_Insurance'] * 0.3 +
    (1 - X['Emergency_Response_Time'] / 400) * 0.3  # Lower time = better
)

# Socioeconomic-health interaction
X['income_x_healthcare'] = (X['Annual_Income'] / 2000000) * X['Healthcare_Access']

print(f"   ✅ Created {len(X.columns) - len(config.KEEP_FEATURES)} new interaction features")
print(f"   Total features now: {len(X.columns)}")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================

print("\n[4] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
)
print(f"   Training set: {len(X_train):,} samples")
print(f"   Test set: {len(X_test):,} samples")

# ============================================================================
# APPLY SMOTE
# ============================================================================

if config.USE_SMOTE:
    print(f"\n[5] Applying SMOTE (target ratio: {config.SMOTE_RATIO})...")
    print(f"   Before SMOTE: {y_train.value_counts().to_dict()}")
    
    smote = SMOTE(random_state=config.RANDOM_STATE, sampling_strategy=config.SMOTE_RATIO)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"   After SMOTE: {y_train_resampled.value_counts().to_dict()}")
    print(f"   ✅ Dataset balanced: {len(X_train_resampled):,} samples")
    
    X_train = X_train_resampled
    y_train = y_train_resampled

# ============================================================================
# SCALE FEATURES
# ============================================================================

print("\n[6] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Features scaled using StandardScaler")

# ============================================================================
# MODEL TRAINING & COMPARISON
# ============================================================================

print("\n[7] Training and comparing models...")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=config.RANDOM_STATE),
    'XGBoost': XGBClassifier(n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=200, random_state=config.RANDOM_STATE, verbose=0)
}

results = []

for name, model in models.items():
    print(f"\n⚙️  Training {name}...")
    
    # Train
    start_time = datetime.now()
    model.fit(X_train_scaled, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=config.CV_FOLDS, scoring='accuracy', n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results.append({
        'model': name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'training_time': training_time,
        'cv_accuracy': cv_mean,
        'cv_std': cv_std,
        'model_object': model
    })
    
    print(f"   Accuracy: {accuracy:.4f} | ROC AUC: {roc_auc:.4f} | F1: {f1:.4f}")
    print(f"   CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

# ============================================================================
# HYPERPARAMETER TUNING FOR BEST MODEL
# ============================================================================

print("\n" + "=" * 80)
print("[8] Hyperparameter tuning for top models...")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('roc_auc', ascending=False)

# Tune top 3 models
top_models = results_df.head(3)

tuned_results = []

for idx, row in top_models.iterrows():
    model_name = row['model']
    print(f"\n⚙️  Tuning {model_name}...")
    
    if 'Gradient Boosting' in model_name:
        param_grid = {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [5, 7, 10],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 1.0]
        }
        base_model = GradientBoostingClassifier(random_state=config.RANDOM_STATE)
        
    elif 'XGBoost' in model_name:
        param_grid = {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [5, 7, 10],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        base_model = XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='logloss', n_jobs=-1)
        
    elif 'LightGBM' in model_name:
        param_grid = {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [5, 7, 10],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 1.0]
        }
        base_model = LGBMClassifier(random_state=config.RANDOM_STATE, verbose=-1, n_jobs=-1)
    else:
        continue
    
    # GridSearchCV
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    tuned_accuracy = accuracy_score(y_test, y_pred)
    tuned_roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"   ✅ Best params: {grid_search.best_params_}")
    print(f"   Tuned Accuracy: {tuned_accuracy:.4f} | Tuned ROC AUC: {tuned_roc_auc:.4f}")
    
    tuned_results.append({
        'model': f"{model_name} (Tuned)",
        'accuracy': tuned_accuracy,
        'roc_auc': tuned_roc_auc,
        'best_params': grid_search.best_params_,
        'model_object': best_model
    })

# ============================================================================
# BUILD ENSEMBLE MODEL
# ============================================================================

print("\n" + "=" * 80)
print("[9] Building Ensemble Model...")
print("=" * 80)

# Get top 3 tuned models
top_3_models = sorted(tuned_results, key=lambda x: x['roc_auc'], reverse=True)[:3]

ensemble_estimators = [
    (result['model'], result['model_object']) 
    for result in top_3_models
]

ensemble = VotingClassifier(
    estimators=ensemble_estimators,
    voting='soft',
    n_jobs=-1
)

print(f"\n⚙️  Training Ensemble with {len(ensemble_estimators)} models...")
ensemble.fit(X_train_scaled, y_train)

y_pred_ensemble = ensemble.predict(X_test_scaled)
y_pred_proba_ensemble = ensemble.predict_proba(X_test_scaled)[:, 1]

ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
ensemble_roc_auc = roc_auc_score(y_test, y_pred_proba_ensemble)
ensemble_f1 = f1_score(y_test, y_pred_ensemble)
ensemble_precision = precision_score(y_test, y_pred_ensemble)
ensemble_recall = recall_score(y_test, y_pred_ensemble)

print(f"\n   ✅ Ensemble Performance:")
print(f"      Accuracy: {ensemble_accuracy:.4f}")
print(f"      ROC AUC: {ensemble_roc_auc:.4f}")
print(f"      F1 Score: {ensemble_f1:.4f}")
print(f"      Precision: {ensemble_precision:.4f}")
print(f"      Recall: {ensemble_recall:.4f}")

# ============================================================================
# SELECT BEST MODEL
# ============================================================================

print("\n" + "=" * 80)
print("[10] Selecting Best Model...")
print("=" * 80)

all_final_results = tuned_results + [{
    'model': 'Ensemble',
    'accuracy': ensemble_accuracy,
    'roc_auc': ensemble_roc_auc,
    'model_object': ensemble
}]

best_result = max(all_final_results, key=lambda x: x['accuracy'])

print(f"\n🏆 BEST MODEL: {best_result['model']}")
print(f"   Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
print(f"   ROC AUC: {best_result['roc_auc']:.4f}")

# ============================================================================
# SAVE MODEL
# ============================================================================

if best_result['accuracy'] >= config.TARGET_ACCURACY:
    print(f"\n✅ TARGET ACHIEVED! Accuracy {best_result['accuracy']*100:.2f}% >= {config.TARGET_ACCURACY*100}%")
    save_decision = True
else:
    print(f"\n⚠️  Target not achieved: {best_result['accuracy']*100:.2f}% < {config.TARGET_ACCURACY*100}%")
    print("   Saving anyway for evaluation...")
    save_decision = True

if save_decision:
    print("\n[11] Saving model and scaler...")
    
    model_path = config.MODEL_DIR / 'heart_attack_model.pkl'
    scaler_path = config.MODEL_DIR / 'scaler.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_result['model_object'], f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"   ✅ Model saved: {model_path}")
    print(f"   ✅ Scaler saved: {scaler_path}")
    
    # Save feature names
    feature_names_path = config.MODEL_DIR / 'feature_names.pkl'
    with open(feature_names_path, 'wb') as f:
        pickle.dump(list(X.columns), f)
    print(f"   ✅ Feature names saved: {feature_names_path}")

# ============================================================================
# FINAL REPORT
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE - FINAL REPORT")
print("=" * 80)

print(f"\n📊 Dataset:")
print(f"   Total samples: {len(df):,}")
print(f"   Features used: {len(X.columns)}")
print(f"   Original features: {len(config.KEEP_FEATURES)}")
print(f"   Engineered features: {len(X.columns) - len(config.KEEP_FEATURES)}")

print(f"\n🏆 Best Model: {best_result['model']}")
print(f"   Accuracy: {best_result['accuracy']*100:.2f}%")
print(f"   ROC AUC: {best_result['roc_auc']:.4f}")

print(f"\n📈 All Model Results:")
results_df_display = results_df[['model', 'accuracy', 'roc_auc', 'f1_score', 'cv_accuracy']]
print(results_df_display.to_string(index=False))

print("\n" + "=" * 80)
print("✅ Training completed successfully!")
print("=" * 80)
