"""
Inference service for Iris classification.
Pure Python business logic, no AWS dependencies.
"""

import numpy as np
from typing import Dict, List, Any
from src.model_loader import get_model, get_metadata


class InferenceService:
    """Service for ML inference operations."""
    
    # Expected feature schema (from Iris dataset)
    EXPECTED_FEATURES = [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width'
    ]
    
    def __init__(self):
        """Initialize service (model loaded lazily on first use)."""
        self.model = None
        self.metadata = None
    
    def _ensure_model_loaded(self):
        """Lazy load model and metadata."""
        if self.model is None:
            self.model = get_model()
            self.metadata = get_metadata()
    
    def validate_input(self, features: Dict[str, float]) -> None:
        """
        Validate input features against expected schema.
        
        Args:
            features: Dictionary of feature names to values
            
        Raises:
            ValueError: If input is invalid
        """
        # Check all required features present
        missing = set(self.EXPECTED_FEATURES) - set(features.keys())
        if missing:
            raise ValueError(
                f"Missing required features: {missing}"
            )
        
        # Check no extra features
        extra = set(features.keys()) - set(self.EXPECTED_FEATURES)
        if extra:
            raise ValueError(
                f"Unexpected features: {extra}"
            )
        
        # Check all values are numeric
        for key, value in features.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Feature '{key}' must be numeric, got {type(value).__name__}"
                )
            
            # Check for reasonable ranges (basic sanity check)
            if value < 0 or value > 100:
                raise ValueError(
                    f"Feature '{key}' value {value} outside reasonable range [0, 100]"
                )
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Run inference on input features.
        
        Args:
            features: Dictionary of feature names to values
                Example: {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
        
        Returns:
            Dictionary containing:
                - prediction: Class name (str)
                - confidence: Probability of predicted class (float)
                - probabilities: All class probabilities (list)
                - model_version: Model version string
        
        Raises:
            ValueError: If input validation fails
        """
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Validate input
        self.validate_input(features)
        
        # Convert features dict to array in correct order
        feature_array = np.array([[
            features[name] for name in self.EXPECTED_FEATURES
        ]])
        
        # Get prediction
        prediction_idx = self.model.predict(feature_array)[0]
        
        # Get probabilities
        probabilities = self.model.predict_proba(feature_array)[0]
        
        # Get class name from metadata
        class_names = self.metadata.get('target_names', ['class_0', 'class_1', 'class_2'])
        predicted_class = class_names[prediction_idx]
        
        # Get confidence (probability of predicted class)
        confidence = float(probabilities[prediction_idx])
        
        # Format response
        result = {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.tolist(),
            'model_version': self.metadata.get('version', 'unknown'),
            'input_features': features
        }
        
        return result


# Module-level service instance (singleton-like behavior)
_service = InferenceService()


def predict(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Convenience function for inference.
    Main public API for this module.
    
    Args:
        features: Dictionary of feature names to values
        
    Returns:
        Prediction results dictionary
    """
    return _service.predict(features)