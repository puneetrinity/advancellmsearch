# tests/test_config.py
"""
Test configuration management
"""

def test_settings_loading():
    """Test settings are loaded correctly"""
    from app.core.config import get_settings
    
    settings = get_settings()
    assert settings.app_name == "AI Search System"
    assert settings.api_port == 8000


def test_model_assignments():
    """Test model assignments are defined"""
    from app.core.config import MODEL_ASSIGNMENTS
    
    assert "simple_classification" in MODEL_ASSIGNMENTS
    assert "qa_and_summary" in MODEL_ASSIGNMENTS
    assert MODEL_ASSIGNMENTS["simple_classification"] == "phi:mini"


def test_priority_tiers():
    """Test model priority tiers"""
    from app.core.config import PRIORITY_TIERS
    
    assert "T0" in PRIORITY_TIERS
    assert "phi:mini" in PRIORITY_TIERS["T0"]


# Run with: pytest -v
# Run integration tests: pytest -v -m integration
# Run with coverage: pytest --cov=app tests/
