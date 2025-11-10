import json
from backend.ml_service import MLService, ModelNotFound


def test_model_load_or_error():
    try:
        ml = MLService()
        assert ml.model is not None
        assert ml.scaler is not None
    except ModelNotFound:
        # Acceptable in CI without artifacts present
        assert True
