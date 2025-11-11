"""
Advanced feature engineering for heart attack prediction.
Created features based on domain knowledge and interaction patterns.
"""

import pandas as pd


def create_advanced_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction and domain-specific features.
    
    Args:
        X_df: DataFrame with 13 standard features
        
    Returns:
        DataFrame with original + engineered features (22 total)
    """
    X_new = X_df.copy()
    
    # Interaction features (use actual column names: trtbps, thalachh)
    X_new['age_chol'] = X_new['age'] * X_new['chol']
    X_new['age_trtbps'] = X_new['age'] * X_new['trtbps']
    X_new['chol_trtbps'] = X_new['chol'] * X_new['trtbps']
    X_new['age_thalachh'] = X_new['age'] * X_new['thalachh']
    
    # Risk scores
    X_new['cardiovascular_risk'] = (
        (X_new['age'] > 60).astype(int) +
        (X_new['chol'] > 240).astype(int) +
        (X_new['trtbps'] > 140).astype(int) +
        (X_new['thalachh'] < 100).astype(int)
    )
    
    # Age groups
    X_new['age_group'] = pd.cut(X_new['age'], bins=[0, 40, 55, 70, 100], labels=[0, 1, 2, 3])
    X_new['age_group'] = X_new['age_group'].astype(int)
    
    # Cholesterol categories
    X_new['chol_category'] = pd.cut(X_new['chol'], bins=[0, 200, 240, 999], labels=[0, 1, 2])
    X_new['chol_category'] = X_new['chol_category'].astype(int)
    
    # Blood pressure categories
    X_new['bp_category'] = pd.cut(X_new['trtbps'], bins=[0, 120, 140, 999], labels=[0, 1, 2])
    X_new['bp_category'] = X_new['bp_category'].astype(int)
    
    # Heart rate categories
    X_new['hr_category'] = pd.cut(X_new['thalachh'], bins=[0, 100, 150, 999], labels=[0, 1, 2])
    X_new['hr_category'] = X_new['hr_category'].astype(int)
    
    return X_new
