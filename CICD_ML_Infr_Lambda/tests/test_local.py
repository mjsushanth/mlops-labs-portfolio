"""Quick local test of the inference pipeline."""

import sys
from pathlib import Path


def find_project_root(root_folder_name: str = "CICD_ML_Infr_Lambda") -> Path:
    """Find project root by searching upward for specific folder name."""
    current = Path(__file__).resolve()
    
    for parent in [current] + list(current.parents):
        if parent.name == root_folder_name:
            return parent
    
    raise RuntimeError(
        f"Could not find project root folder '{root_folder_name}'"
    )


# Find and add project root to Python path
PROJECT_ROOT = find_project_root("CICD_ML_Infr_Lambda")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}\n")

# Now imports work
from src.inference_service import predict

# Test input (should predict "setosa")
test_input = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

print("Testing inference pipeline...")
result = predict(test_input)

print("\nResult:")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Probabilities: {result['probabilities']}")
print(f"Model version: {result['model_version']}")