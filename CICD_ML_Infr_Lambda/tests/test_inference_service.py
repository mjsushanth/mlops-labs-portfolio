"""
Unit tests for inference_service.py
Tests input validation, prediction logic, and error handling.
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
from src.inference_service import predict, InferenceService


class TestInferenceService:
    """Test suite for InferenceService."""
    
    def test_valid_setosa_prediction(self):
        """Test prediction for typical setosa sample."""
        # Known setosa sample (small measurements)
        input_features = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        result = predict(input_features)
        
        # Check prediction
        assert result['prediction'] == 'setosa', "Should predict setosa for small flowers"
        
        # Check confidence is high
        assert result['confidence'] > 0.9, "Confidence should be high for clear cases"
        
        # Check response structure
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'model_version' in result
        assert 'input_features' in result
    
    def test_valid_virginica_prediction(self):
        """Test prediction for typical virginica sample."""
        # Known virginica sample (large measurements)
        input_features = {
            "sepal_length": 6.5,
            "sepal_width": 3.0,
            "petal_length": 5.5,
            "petal_width": 1.8
        }
        
        result = predict(input_features)
        
        # Should predict virginica or versicolor (both have large petals)
        assert result['prediction'] in ['virginica', 'versicolor']
        assert result['confidence'] > 0.5
    
    def test_probabilities_sum_to_one(self):
        """Test that prediction probabilities sum to 1.0."""
        input_features = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        result = predict(input_features)
        prob_sum = sum(result['probabilities'])
        
        # Allow small floating point error
        assert abs(prob_sum - 1.0) < 0.001, "Probabilities should sum to 1.0"
    
    def test_probabilities_length(self):
        """Test that probabilities array has correct length."""
        input_features = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        result = predict(input_features)
        
        assert len(result['probabilities']) == 3, "Should have 3 class probabilities"


class TestInputValidation:
    """Test input validation logic."""
    
    def test_missing_feature_raises_error(self):
        """Test that missing required features raise ValueError."""
        incomplete_input = {
            "sepal_length": 5.1,
            "sepal_width": 3.5
            # Missing petal_length and petal_width
        }
        
        with pytest.raises(ValueError, match="Missing required features"):
            predict(incomplete_input)
    
    def test_extra_feature_raises_error(self):
        """Test that extra unexpected features raise ValueError."""
        input_with_extra = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
            "extra_field": 999  # This shouldn't be here
        }
        
        with pytest.raises(ValueError, match="Unexpected features"):
            predict(input_with_extra)
    
    def test_non_numeric_feature_raises_error(self):
        """Test that non-numeric values raise ValueError."""
        invalid_input = {
            "sepal_length": "not a number",  # String instead of number
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        with pytest.raises(ValueError, match="must be numeric"):
            predict(invalid_input)
    
    def test_negative_value_raises_error(self):
        """Test that negative values raise ValueError."""
        invalid_input = {
            "sepal_length": -5.1,  # Negative value
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        with pytest.raises(ValueError, match="outside reasonable range"):
            predict(invalid_input)
    
    def test_extremely_large_value_raises_error(self):
        """Test that unreasonably large values raise ValueError."""
        invalid_input = {
            "sepal_length": 500.0,  # Way too large
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        with pytest.raises(ValueError, match="outside reasonable range"):
            predict(invalid_input)


class TestResponseFormat:
    """Test that response format is correct."""
    
    def test_response_has_all_required_fields(self):
        """Test that response contains all expected fields."""
        input_features = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        result = predict(input_features)
        
        required_fields = ['prediction', 'confidence', 'probabilities', 
                          'model_version', 'input_features']
        
        for field in required_fields:
            assert field in result, f"Response should contain '{field}'"
    
    def test_prediction_is_string(self):
        """Test that prediction is a string (class name)."""
        input_features = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        result = predict(input_features)
        
        assert isinstance(result['prediction'], str), "Prediction should be string"
    
    def test_confidence_is_float(self):
        """Test that confidence is a float between 0 and 1."""
        input_features = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        result = predict(input_features)
        
        assert isinstance(result['confidence'], float), "Confidence should be float"
        assert 0.0 <= result['confidence'] <= 1.0, "Confidence should be between 0 and 1"
    
    def test_probabilities_is_list(self):
        """Test that probabilities is a list."""
        input_features = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        result = predict(input_features)
        
        assert isinstance(result['probabilities'], list), "Probabilities should be list"