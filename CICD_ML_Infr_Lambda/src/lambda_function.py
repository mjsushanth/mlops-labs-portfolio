"""
AWS Lambda handler for ML inference.
Thin adapter layer - delegates to inference service.
"""

import json
from typing import Dict, Any
from src.inference_service import predict


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda entry point.
    
    Supports two invocation types:
    1. API Gateway: event has 'body' key with JSON string
    2. Direct invoke: event is the input dictionary
    
    Args:
        event: Lambda event (dict or API Gateway proxy event)
        context: Lambda context (unused)
        
    Returns:
        API Gateway formatted response with statusCode and body
    """
    try:
        # Parse input based on invocation type
        if 'body' in event:
            # API Gateway invocation
            features = json.loads(event['body'])
        else:
            # Direct Lambda invocation
            features = event
        
        # Call business logic
        result = predict(features)
        
        # Return success response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps(result)
        }
    
    except ValueError as e:
        # Input validation errors
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': 'Invalid input',
                'message': str(e)
            })
        }
    
    except Exception as e:
        # Unexpected errors
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }