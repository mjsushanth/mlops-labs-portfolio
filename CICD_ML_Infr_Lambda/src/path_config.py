"""
Project path configuration.
Single source of truth for all path resolution.
Works in both local dev and Lambda environments.
"""

import os
from pathlib import Path


def find_project_root(root_folder_name: str = "CICD_ML_Infr_Lambda") -> Path:
    """
    Find project root by searching upward for specific folder name.
    In Lambda, uses /var/task directly.
    
    Args:
        root_folder_name: Exact name of project root folder (ignored in Lambda)
        
    Returns:
        Absolute path to project root
        
    Raises:
        RuntimeError: If root folder not found
    """
    # Lambda detection - highest priority
    if os.environ.get('LAMBDA_TASK_ROOT'):
        # We're in Lambda, use the Lambda root
        return Path(os.environ['LAMBDA_TASK_ROOT'])
    
    if os.environ.get('AWS_EXECUTION_ENV'):
        # Alternative Lambda detection
        return Path('/var/task')
    
    # Local development - search by folder name
    current = Path(__file__).resolve()
    
    for parent in [current] + list(current.parents):
        if parent.name == root_folder_name:
            return parent
    
    # If not found locally, raise error
    raise RuntimeError(
        f"Could not find project root folder '{root_folder_name}'. "
        f"Searched from: {current}"
    )


# Module-level constant - computed once on import
PROJECT_ROOT = find_project_root("CICD_ML_Infr_Lambda")
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"