import os
import pytest

# Basic smoke test to ensure app module imports

def test_imports():
    import backend.main as main
    assert hasattr(main, 'app')
