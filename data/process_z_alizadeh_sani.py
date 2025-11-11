"""
Process Z-Alizadeh Sani Dataset
Real medical data: 303 Asian patients with coronary artery disease
"""
import pandas as pd
import numpy as np
from pathlib import Path

def process_z_alizadeh_sani():
    """Load and analyze the Z-Alizadeh Sani dataset"""
    print("=" * 80)
    print("PROCESSING Z-ALIZADEH SANI DATASET (REAL MEDICAL DATA)")
    print("=" * 80)
    
    # Load the Excel file
    file_path = Path("data/real_datasets/z_alizadeh_sani/Z-Alizadeh sani dataset.xlsx")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"\n📂 Loading: {file_path.name}")
    
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        
        print(f"✅ Loaded successfully!")
        print(f"\n📊 Dataset Overview:")
        print(f"   Records: {len(df):,}")
        print(f"   Features: {len(df.columns)}")
        print(f"   Shape: {df.shape}")
        
        # Display column names
        print(f"\n📋 All Features ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        # Basic statistics
        print(f"\n📈 Data Quality:")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        print(f"   Duplicates: {df.duplicated().sum()}")
        
        # Check for target variable
        print(f"\n🎯 Potential Target Variables:")
        target_candidates = ['Cath', 'CAD', 'Diagnosis', 'Class', 'Target', 'Severity']
        for col in df.columns:
            col_lower = col.lower()
            if any(candidate.lower() in col_lower for candidate in target_candidates):
                print(f"   - {col}: {df[col].nunique()} unique values")
                print(f"     Distribution: {df[col].value_counts().to_dict()}")
        
        # Show first few rows
        print(f"\n📄 First 5 Rows:")
        print(df.head())
        
        # Data types
        print(f"\n🔢 Data Types:")
        print(df.dtypes.value_counts())
        
        # Correlation analysis (only for numeric columns)
        print(f"\n🔗 Correlation Analysis:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"   Numeric features: {len(numeric_cols)}")
        
        # Try to find target column
        target_col = None
        for col in ['Cath', 'CAD', 'Diagnosis', 'Class']:
            if col in df.columns:
                target_col = col
                break
        
        if target_col and target_col in numeric_cols:
            correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
            print(f"\n   Top 10 correlations with '{target_col}':")
            for feature, corr in correlations.head(11).items():
                if feature != target_col:
                    print(f"      {feature}: {corr:.4f}")
        
        # Save as CSV for easier processing
        csv_path = file_path.parent / "z_alizadeh_sani.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Saved as CSV: {csv_path}")
        
        print("\n" + "=" * 80)
        print("✅ PROCESSING COMPLETE")
        print("=" * 80)
        print("""
This is REAL MEDICAL DATA from hospital coronary angiography!

Key differences from synthetic data:
✅ Real correlations between features and outcomes
✅ Clinical patterns that make medical sense
✅ Published and peer-reviewed
✅ Used in actual cardiovascular research

Next Steps:
1. Identify the target variable (likely 'Cath' or 'CAD')
2. Map features to your schema
3. Handle missing values appropriately
4. Split train/test properly
5. Retrain model and compare with synthetic dataset

Expected improvements:
- Stronger correlations (> 0.3 for key features)
- Better ROC AUC (> 0.75)
- Higher accuracy (75-85%)
- Results will match medical literature
        """)
        
        return df
        
    except Exception as e:
        print(f"\n❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    df = process_z_alizadeh_sani()
    
    if df is not None:
        print("\n📊 Dataset loaded successfully!")
        print("Access it with: df = pd.read_csv('data/real_datasets/z_alizadeh_sani/z_alizadeh_sani.csv')")
