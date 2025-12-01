"""
Unit tests for model_loader.py
Tests singleton pattern, lazy loading, and caching behavior.
"""

import sys
from pathlib import Path

# Add project root to path
def find_project_root(root_folder_name: str = "CICD_ML_Infr_Lambda") -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if parent.name == root_folder_name:
            return parent
    raise RuntimeError(f"Could not find project root '{root_folder_name}'")

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now we can import
import pytest
from src.model_loader import get_model, get_metadata, ModelLoader


class TestModelLoader:
    """Test suite for ModelLoader singleton."""
    
    def test_get_model_returns_model(self):
        """Test that get_model() returns a valid model object."""
        model = get_model()
        
        # Check model has predict method
        assert hasattr(model, 'predict'), "Model should have predict method"
        assert hasattr(model, 'predict_proba'), "Model should have predict_proba method"
        
        # Check it's a sklearn model
        assert hasattr(model, 'fit'), "Model should have fit method (sklearn API)"
    
    def test_singleton_returns_same_instance(self):
        """Test that ModelLoader implements singleton pattern correctly."""
        loader1 = ModelLoader()
        loader2 = ModelLoader()
        
        # Should be exact same object in memory
        assert loader1 is loader2, "ModelLoader should return same instance (singleton)"
    
    def test_model_caching(self):
        """Test that model is cached and not reloaded on subsequent calls."""
        model1 = get_model()
        model2 = get_model()
        
        # Should be exact same object (cached)
        assert model1 is model2, "get_model() should return cached instance"
    
    def test_get_metadata_returns_dict(self):
        """Test that get_metadata() returns valid metadata dictionary."""
        metadata = get_metadata()
        
        assert isinstance(metadata, dict), "Metadata should be a dictionary"
        
        # Check required keys
        assert 'version' in metadata, "Metadata should contain 'version'"
        assert 'model_type' in metadata, "Metadata should contain 'model_type'"
    
    def test_metadata_caching(self):
        """Test that metadata is cached."""
        metadata1 = get_metadata()
        metadata2 = get_metadata()
        
        # Should be exact same object (cached)
        assert metadata1 is metadata2, "get_metadata() should return cached instance"
    
    def test_model_has_expected_attributes(self):
        """Test that loaded model has expected sklearn attributes."""
        model = get_model()
        
        # Check RandomForest specific attributes
        assert hasattr(model, 'n_estimators'), "Model should have n_estimators"
        assert hasattr(model, 'max_depth'), "Model should have max_depth"
        
        # Verify it's trained (has classes_)
        assert hasattr(model, 'classes_'), "Model should be trained (has classes_ attribute)"
        assert len(model.classes_) == 3, "Iris model should have 3 classes"


class TestModelPaths:
    """Test that model paths are correctly resolved."""
    
    def test_model_file_exists(self):
        """Test that model.pkl file exists at expected location."""
        from src.path_config import MODELS_DIR
        
        model_path = MODELS_DIR / "model.pkl"
        assert model_path.exists(), f"Model file should exist at {model_path}"
    
    def test_metadata_file_exists(self):
        """Test that model_metadata.json exists."""
        from src.path_config import MODELS_DIR
        
        metadata_path = MODELS_DIR / "model_metadata.json"
        assert metadata_path.exists(), f"Metadata file should exist at {metadata_path}"