import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_wine_data(data_path: str = None):
    """Load and prepare wine quality dataset from local file"""
    
    if data_path is None:
        script_dir = Path(__file__).parent  # src/ directory
        project_root = script_dir.parent    # project root
        data_path = project_root / "data" / "WineQT.csv"
    
    data_file = Path(data_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}. Please ensure WineQT.csv is in the data/ directory.")
    
    df = pd.read_csv(data_file)
    
    print(f"Dataset loaded from: {data_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # if target column exists 
    if 'quality' not in df.columns:
        print("Available columns:", df.columns.tolist())
        print("Please check if the target column is named 'quality' or adjust accordingly.")
    else:
        print(f"Quality distribution:\n{df['quality'].value_counts().sort_index()}")
    
    return df

def prepare_data(df, test_size=0.2, random_state=42):
    """Split and scale the wine data"""
    
    # dropping both 'quality' and 'Id' columns as they are not features.
    X = df.drop(['quality', 'Id'], axis=1)
    y = df['quality']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)

def get_feature_info():
    """Feature descriptions for API documentation"""
    return {
        'fixed_acidity': 'Fixed acidity (g/L)',
        'volatile_acidity': 'Volatile acidity (g/L)', 
        'citric_acid': 'Citric acid (g/L)',
        'residual_sugar': 'Residual sugar (g/L)',
        'chlorides': 'Chlorides (g/L)',
        'free_sulfur_dioxide': 'Free SO2 (mg/L)',
        'total_sulfur_dioxide': 'Total SO2 (mg/L)',
        'density': 'Density (g/mL)',
        'pH': 'pH level',
        'sulphates': 'Sulphates (g/L)',
        'alcohol': 'Alcohol (%)'
    }