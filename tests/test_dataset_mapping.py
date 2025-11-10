import pandas as pd

from ml.train import FEATURES, TARGET, map_indian_columns


def test_map_indian_columns_produces_expected_schema():
    sample = pd.DataFrame([
        {
            "Age": 55,
            "Gender": "Male",
            "Diabetes": 1,
            "Hypertension": 0,
            "Obesity": 1,
            "Smoking": 0,
            "Alcohol_Consumption": 1,
            "Physical_Activity": 3,
            "Stress_Level": 7,
            "Cholesterol_Level": 210,
            "HDL_Level": 35,
            "Systolic_BP": 140,
            "Family_History": 1,
            "Heart_Attack_Risk": 1,
        }
    ])

    mapped = map_indian_columns(sample)

    assert list(mapped.columns) == FEATURES + [TARGET]
    assert mapped.shape[0] == 1
    assert mapped[TARGET].iloc[0] == 1
    assert mapped.notna().all().all()
