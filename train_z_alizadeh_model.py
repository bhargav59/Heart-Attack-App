"""
Train Heart Attack Prediction Model on Z-Alizadeh Sani Real Medical Dataset
============================================================================

Dataset: Z-Alizadeh Sani et al. (2013)
Source: UCI Machine Learning Repository
Patients: 303 Asian patients (216 CAD, 87 Normal)
Features: 56 clinical features from coronary angiography
Quality: Max correlation 0.5430 (REAL medical data)

This script trains an advanced ensemble model with:
- Feature engineering and selection
- SMOTE-Tomek for class balancing
- RobustScaler for outlier resistance
- Multiple models (XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees)
- Stacking ensemble with logistic regression meta-learner
- Probability calibration
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)

# Advanced models
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM not available")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️  CatBoost not available")

# Imbalanced learning
try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import TomekLinks
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("⚠️  imbalanced-learn not available")

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load Z-Alizadeh Sani dataset and prepare for training"""
    print("=" * 80)
    print("📂 LOADING Z-ALIZADEH SANI DATASET")
    print("=" * 80)
    
    # Load dataset
    data_file = Path("data/real_datasets/z_alizadeh_sani/z_alizadeh_sani.csv")
    df = pd.read_csv(data_file)
    
    print(f"\n✅ Loaded {len(df)} patient records with {len(df.columns)} columns")
    
    # Convert target variable: Cad = 1 (high risk), Normal = 0 (low risk)
    print("\n🎯 Converting target variable:")
    print(f"   Cath column: {df['Cath'].value_counts().to_dict()}")
    
    df['Target'] = (df['Cath'] == 'Cad').astype(int)
    print(f"   Target: CAD={df['Target'].sum()}, Normal={(df['Target']==0).sum()}")
    print(f"   Class distribution: {df['Target'].value_counts(normalize=True).to_dict()}")
    
    # Drop original target column
    df = df.drop('Cath', axis=1)
    
    # Handle categorical variables
    print("\n🔤 Processing categorical variables:")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    label_encoders = {}
    for col in categorical_cols:
        if col != 'Target':
            print(f"   - {col}: {df[col].nunique()} unique values")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Handle missing values
    print("\n🔍 Checking for missing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"   Found {missing.sum()} missing values")
        for col in missing[missing > 0].index:
            print(f"   - {col}: {missing[col]} missing")
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        print("   ✅ No missing values")
    
    return df, label_encoders


def create_advanced_features(df):
    """Create advanced engineered features from Z-Alizadeh Sani data"""
    print("\n" + "=" * 80)
    print("⚙️  FEATURE ENGINEERING")
    print("=" * 80)
    
    df_engineered = df.copy()
    
    # Cardiovascular risk score composite
    cv_risk_features = []
    if 'Age' in df.columns:
        df_engineered['Age_Risk'] = (df['Age'] > 60).astype(int)
        cv_risk_features.append('Age_Risk')
    
    if 'HTN' in df.columns:
        cv_risk_features.append('HTN')
    if 'DM' in df.columns:
        cv_risk_features.append('DM')
    if 'Current Smoker' in df.columns:
        cv_risk_features.append('Current Smoker')
    if 'DLP' in df.columns:
        cv_risk_features.append('DLP')
    
    if cv_risk_features:
        df_engineered['CV_Risk_Score'] = df_engineered[cv_risk_features].sum(axis=1)
        print(f"✅ Created CV_Risk_Score from {len(cv_risk_features)} factors")
    
    # Metabolic syndrome indicator
    metabolic_features = []
    if 'BMI' in df.columns:
        df_engineered['High_BMI'] = (df['BMI'] > 30).astype(int)
        metabolic_features.append('High_BMI')
    if 'FBS' in df.columns:
        df_engineered['High_FBS'] = (df['FBS'] > 126).astype(int)
        metabolic_features.append('High_FBS')
    if 'TG' in df.columns:
        df_engineered['High_TG'] = (df['TG'] > 150).astype(int)
        metabolic_features.append('High_TG')
    if 'HDL' in df.columns:
        df_engineered['Low_HDL'] = (df['HDL'] < 40).astype(int)
        metabolic_features.append('Low_HDL')
    
    if metabolic_features:
        df_engineered['Metabolic_Syndrome'] = df_engineered[metabolic_features].sum(axis=1)
        print(f"✅ Created Metabolic_Syndrome from {len(metabolic_features)} indicators")
    
    # ECG abnormality score
    ecg_features = []
    for col in ['Q Wave', 'St Elevation', 'St Depression', 'Tinversion', 'LVH', 'Poor R Progression']:
        if col in df.columns:
            ecg_features.append(col)
    
    if ecg_features:
        df_engineered['ECG_Abnormality_Score'] = df_engineered[ecg_features].sum(axis=1)
        print(f"✅ Created ECG_Abnormality_Score from {len(ecg_features)} findings")
    
    # Lipid ratios (strong predictors)
    if 'LDL' in df.columns and 'HDL' in df.columns:
        df_engineered['LDL_HDL_Ratio'] = df['LDL'] / (df['HDL'] + 1)  # +1 to avoid division by zero
        print("✅ Created LDL_HDL_Ratio")
    
    if 'TG' in df.columns and 'HDL' in df.columns:
        df_engineered['TG_HDL_Ratio'] = df['TG'] / (df['HDL'] + 1)
        print("✅ Created TG_HDL_Ratio")
    
    # Age-related risk factors
    if 'Age' in df.columns:
        if 'Sex' in df.columns:
            # Higher risk for men at younger age, women at older age
            df_engineered['Age_Sex_Risk'] = df['Age'] * (df['Sex'] == 1)  # Assuming Male=1
            print("✅ Created Age_Sex_Risk interaction")
        
        if 'DM' in df.columns:
            df_engineered['Age_DM_Risk'] = df['Age'] * df['DM']
            print("✅ Created Age_DM_Risk interaction")
    
    # Cardiac function indicators
    if 'EF-TTE' in df.columns:
        df_engineered['Low_EF'] = (df['EF-TTE'] < 40).astype(int)
        print("✅ Created Low_EF (ejection fraction < 40%)")
    
    # Symptom severity
    symptom_features = []
    for col in ['Typical Chest Pain', 'Dyspnea', 'Atypical', 'Nonanginal', 'Exertional CP']:
        if col in df.columns:
            symptom_features.append(col)
    
    if symptom_features:
        df_engineered['Symptom_Severity'] = df_engineered[symptom_features].sum(axis=1)
        print(f"✅ Created Symptom_Severity from {len(symptom_features)} symptoms")
    
    # Lab abnormality score
    lab_features = []
    if 'ESR' in df.columns:
        df_engineered['High_ESR'] = (df['ESR'] > 20).astype(int)
        lab_features.append('High_ESR')
    if 'WBC' in df.columns:
        df_engineered['High_WBC'] = (df['WBC'] > 11000).astype(int)
        lab_features.append('High_WBC')
    if 'CR' in df.columns:
        df_engineered['High_CR'] = (df['CR'] > 1.3).astype(int)
        lab_features.append('High_CR')
    
    if lab_features:
        df_engineered['Lab_Abnormality'] = df_engineered[lab_features].sum(axis=1)
        print(f"✅ Created Lab_Abnormality from {len(lab_features)} tests")
    
    new_features = len(df_engineered.columns) - len(df.columns)
    print(f"\n📊 Total features: {len(df.columns)} → {len(df_engineered.columns)} (+{new_features})")
    
    return df_engineered


def select_best_features(X, y, n_features=40):
    """Select best features using mutual information"""
    print("\n" + "=" * 80)
    print("🎯 FEATURE SELECTION")
    print("=" * 80)
    
    # Use mutual information for feature selection
    selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
    selector.fit(X, y)
    
    # Get selected feature names
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    print(f"\n📊 Top 20 features by mutual information:")
    for i, row in feature_scores.head(20).iterrows():
        print(f"   {row.name + 1:2d}. {row['feature']:30s}: {row['score']:.4f}")
    
    selected_features = feature_scores.head(n_features)['feature'].tolist()
    print(f"\n✅ Selected {len(selected_features)} features")
    
    return selected_features, feature_scores


def train_model(X_train, y_train, X_test, y_test):
    """Train advanced ensemble model"""
    print("\n" + "=" * 80)
    print("🤖 MODEL TRAINING")
    print("=" * 80)
    
    # Balance classes with SMOTE-Tomek
    if HAS_IMBLEARN:
        print("\n⚖️  Balancing classes with SMOTE-Tomek...")
        smote_tomek = SMOTETomek(
            smote=SMOTE(sampling_strategy=0.85, random_state=42),
            tomek=TomekLinks(sampling_strategy='majority'),
            random_state=42
        )
        X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
        print(f"   Before: {y_train.value_counts().to_dict()}")
        print(f"   After: {pd.Series(y_train_balanced).value_counts().to_dict()}")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Scale features
    print("\n📏 Scaling features with RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Define base models
    print("\n🏗️  Building ensemble models...")
    models = {}
    
    # Random Forest
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Extra Trees
    models['ExtraTrees'] = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting
    models['GradientBoosting'] = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42
    )
    
    # XGBoost
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    
    # LightGBM
    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    
    # CatBoost
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostClassifier(
            iterations=150,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=3,
            subsample=0.8,
            random_seed=42,
            verbose=0
        )
    
    # Train and evaluate each model
    results = []
    trained_models = {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train_balanced, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        # Train on full training set
        model.fit(X_train_scaled, y_train_balanced)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'CV_AUC_Mean': cv_scores.mean(),
            'CV_AUC_Std': cv_scores.std(),
            'Test_Accuracy': accuracy,
            'Test_ROC_AUC': roc_auc,
            'Test_F1': f1
        })
        
        trained_models[name] = model
        
        print(f"      CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"      Test Accuracy: {accuracy:.4f}")
        print(f"      Test ROC AUC: {roc_auc:.4f}")
        print(f"      Test F1: {f1:.4f}")
    
    # Create stacking ensemble
    print("\n🏆 Building Stacking Ensemble...")
    
    base_estimators = [
        ('rf', models['RandomForest']),
        ('et', models['ExtraTrees']),
        ('gb', models['GradientBoosting'])
    ]
    
    if HAS_XGBOOST:
        base_estimators.append(('xgb', models['XGBoost']))
    if HAS_LIGHTGBM:
        base_estimators.append(('lgbm', models['LightGBM']))
    if HAS_CATBOOST:
        base_estimators.append(('cat', models['CatBoost']))
    
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42
        ),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    print("   Training stacking ensemble...")
    stacking_model.fit(X_train_scaled, y_train_balanced)
    
    y_pred_stack = stacking_model.predict(X_test_scaled)
    y_pred_proba_stack = stacking_model.predict_proba(X_test_scaled)[:, 1]
    
    stack_accuracy = accuracy_score(y_test, y_pred_stack)
    stack_roc_auc = roc_auc_score(y_test, y_pred_proba_stack)
    stack_f1 = f1_score(y_test, y_pred_stack)
    
    results.append({
        'Model': 'Stacking',
        'CV_AUC_Mean': None,
        'CV_AUC_Std': None,
        'Test_Accuracy': stack_accuracy,
        'Test_ROC_AUC': stack_roc_auc,
        'Test_F1': stack_f1
    })
    
    print(f"\n   Stacking Results:")
    print(f"      Test Accuracy: {stack_accuracy:.4f}")
    print(f"      Test ROC AUC: {stack_roc_auc:.4f}")
    print(f"      Test F1: {stack_f1:.4f}")
    
    # Calibrate the best model
    print("\n🎯 Calibrating best model...")
    
    # Find best model by ROC AUC
    results_df = pd.DataFrame(results)
    best_model_name = results_df.loc[results_df['Test_ROC_AUC'].idxmax(), 'Model']
    
    if best_model_name == 'Stacking':
        best_model = stacking_model
    else:
        best_model = trained_models[best_model_name]
    
    print(f"   Best model: {best_model_name}")
    
    calibrated_model = CalibratedClassifierCV(
        best_model,
        method='sigmoid',
        cv=5
    )
    calibrated_model.fit(X_train_scaled, y_train_balanced)
    
    y_pred_cal = calibrated_model.predict(X_test_scaled)
    y_pred_proba_cal = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    cal_accuracy = accuracy_score(y_test, y_pred_cal)
    cal_roc_auc = roc_auc_score(y_test, y_pred_proba_cal)
    cal_f1 = f1_score(y_test, y_pred_cal)
    
    results.append({
        'Model': f'Calibrated_{best_model_name}',
        'CV_AUC_Mean': None,
        'CV_AUC_Std': None,
        'Test_Accuracy': cal_accuracy,
        'Test_ROC_AUC': cal_roc_auc,
        'Test_F1': cal_f1
    })
    
    print(f"\n   Calibrated Results:")
    print(f"      Test Accuracy: {cal_accuracy:.4f}")
    print(f"      Test ROC AUC: {cal_roc_auc:.4f}")
    print(f"      Test F1: {cal_f1:.4f}")
    
    # Display all results
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Final best model
    final_best_idx = results_df['Test_ROC_AUC'].idxmax()
    final_best_name = results_df.loc[final_best_idx, 'Model']
    
    if final_best_name.startswith('Calibrated'):
        final_model = calibrated_model
    elif final_best_name == 'Stacking':
        final_model = stacking_model
    else:
        final_model = trained_models[final_best_name]
    
    print(f"\n🏆 FINAL BEST MODEL: {final_best_name}")
    print(f"   Accuracy: {results_df.loc[final_best_idx, 'Test_Accuracy']:.4f}")
    print(f"   ROC AUC: {results_df.loc[final_best_idx, 'Test_ROC_AUC']:.4f}")
    print(f"   F1 Score: {results_df.loc[final_best_idx, 'Test_F1']:.4f}")
    
    return final_model, scaler, results_df, final_best_name


def save_model(model, scaler, feature_names, model_name):
    """Save trained model and artifacts"""
    print("\n" + "=" * 80)
    print("💾 SAVING MODEL")
    print("=" * 80)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "heart_attack_model_real.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = models_dir / "scaler_real.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved: {scaler_path}")
    
    # Save feature names
    features_path = models_dir / "feature_names_real.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"✅ Feature names saved: {features_path}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'n_features': len(feature_names),
        'training_date': datetime.now().isoformat(),
        'dataset': 'Z-Alizadeh Sani (UCI)',
        'dataset_size': 303,
        'feature_names': feature_names
    }
    
    metadata_path = models_dir / "model_metadata_real.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✅ Metadata saved: {metadata_path}")
    
    print(f"\n📦 Model artifacts saved to {models_dir}/")


def main():
    """Main training pipeline"""
    print("\n" + "=" * 80)
    print("🏥 HEART ATTACK PREDICTION MODEL TRAINING")
    print("    Dataset: Z-Alizadeh Sani (Real Medical Data)")
    print("=" * 80)
    
    # Load data
    df, label_encoders = load_and_prepare_data()
    
    # Feature engineering
    df_engineered = create_advanced_features(df)
    
    # Prepare features and target
    X = df_engineered.drop('Target', axis=1)
    y = df_engineered['Target']
    
    # Feature selection
    selected_features, feature_scores = select_best_features(X, y, n_features=40)
    X_selected = X[selected_features]
    
    # Split data
    print("\n" + "=" * 80)
    print("📊 DATA SPLIT")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    print(f"Features: {len(selected_features)}")
    print(f"\nClass distribution (train): {y_train.value_counts().to_dict()}")
    print(f"Class distribution (test): {y_test.value_counts().to_dict()}")
    
    # Train model
    model, scaler, results_df, model_name = train_model(X_train, y_train, X_test, y_test)
    
    # Save model
    save_model(model, scaler, selected_features, model_name)
    
    # Save results
    results_file = Path("z_alizadeh_model_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ Results saved: {results_file}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    
    print(f"""
Next Steps:
1. Review model performance metrics
2. Update backend to use new model (heart_attack_model_real.pkl)
3. Test API with real-data-trained model
4. Compare with synthetic model results
5. Deploy to production

Model artifacts:
- models/heart_attack_model_real.pkl
- models/scaler_real.pkl
- models/feature_names_real.pkl
- models/model_metadata_real.pkl
    """)


if __name__ == "__main__":
    main()
