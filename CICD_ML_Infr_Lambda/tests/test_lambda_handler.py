"""
Unit tests for Lambda handler.
Tests the handler without deploying to AWS.
"""

import sys
from pathlib import Path
import json

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

import pytest
from src.lambda_function import lambda_handler


class TestLambdaHandler:
    """Test Lambda handler locally without AWS."""
    
    def test_direct_invoke_success(self):
        """Test Lambda handler with direct invocation (no API Gateway)."""
        # Simulate direct Lambda invoke
        event = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        context = None  # Context not needed for our handler
        
        # Call handler directly
        response = lambda_handler(event, context)
        
        # Verify response structure
        assert response['statusCode'] == 200
        assert 'body' in response
        
        # Parse body
        body = json.loads(response['body'])
        
        # Verify prediction
        assert body['prediction'] == 'setosa'
        assert body['confidence'] > 0.9
        assert 'probabilities' in body
        assert 'model_version' in body
    
    def test_api_gateway_invoke_success(self):
        """Test Lambda handler with API Gateway event format."""
        # Simulate API Gateway proxy event
        event = {
            "body": json.dumps({
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }),
            "headers": {
                "Content-Type": "application/json"
            }
        }
        context = None
        
        response = lambda_handler(event, context)
        
        assert response['statusCode'] == 200
        body = json.loads(response['body'])
        assert body['prediction'] == 'setosa'
    
    def test_invalid_input_returns_400(self):
        """Test that invalid input returns 400 Bad Request."""
        event = {
            "sepal_length": 5.1
            # Missing required features
        }
        context = None
        
        response = lambda_handler(event, context)
        
        assert response['statusCode'] == 400
        body = json.loads(response['body'])
        assert 'error' in body
        assert 'Missing required features' in body['message']
    
    def test_non_numeric_input_returns_400(self):
        """Test that non-numeric input returns 400."""
        event = {
            "sepal_length": "not a number",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        context = None
        
        response = lambda_handler(event, context)
        
        assert response['statusCode'] == 400


"""
pytest tests/test_lambda_handler.py -v

"""