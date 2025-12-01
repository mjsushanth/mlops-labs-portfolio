"""
Model loader with singleton pattern for Lambda optimization.
Loads model once on cold start, caches for warm requests.
"""

import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Import project root from centralized config
from src.path_config import PROJECT_ROOT, MODELS_DIR


class ModelLoader:
    """Singleton model loader for scikit-learn inference."""
    
    _instance: Optional['ModelLoader'] = None
    _model: Optional[Any] = None
    _metadata: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize paths but don't load model yet (lazy loading)."""
        # Don't reload if already initialized
        if self._model is not None:
            return
        
        # Use absolute paths from root
        self.model_dir = MODELS_DIR
        self.model_path = self.model_dir / "model.pkl"
        self.metadata_path = self.model_dir / "model_metadata.json"
    
    def load_model(self) -> Any:
        """
        Load model from pickle file (lazy loading).
        Returns cached model on subsequent calls.
        """
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    f"Project root: {PROJECT_ROOT}\n"
                    f"Models directory: {self.model_dir}"
                )
            
            with open(self.model_path, 'rb') as f:
                self._model = pickle.load(f)
            
            print(f"Model loaded from: {self.model_path}")
        
        return self._model
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load model metadata from JSON file.
        Returns cached metadata on subsequent calls.
        """
        if self._metadata is None:
            if not self.metadata_path.exists():
                # Metadata is optional, return defaults
                self._metadata = {
                    'version': 'unknown',
                    'model_type': 'RandomForestClassifier'
                }
            else:
                with open(self.metadata_path, 'r') as f:
                    self._metadata = json.load(f)
            
            print(f"Metadata loaded: version {self._metadata.get('version')}")
        
        return self._metadata
    
    def get_model(self) -> Any:
        """Public API: Get loaded model (loads if needed)."""
        return self.load_model()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Public API: Get model metadata."""
        return self.load_metadata()


# Module-level singleton instance
_loader = ModelLoader()


def get_model():
    """
    Convenience function to get model instance.
    This is the main public API for other modules.
    """
    return _loader.get_model()


def get_metadata():
    """
    Convenience function to get model metadata.
    """
    return _loader.get_metadata()