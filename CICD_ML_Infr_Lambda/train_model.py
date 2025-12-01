"""
Train Iris classification model for Lambda deployment.
Run once to generate models/model.pkl artifact.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
from pathlib import Path
from datetime import datetime


# Configuration
CONFIG = {
    'n_estimators': 100,
    'max_depth': 5,
    'random_state': 42,
    'test_size': 0.2,
    'model_version': 'v1.0'
}


def load_data():
    """Load and prepare Iris dataset."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    return X, y, feature_names, target_names


def train_model(X_train, y_train):
    """Train RandomForest classifier."""
    model = RandomForestClassifier(
        n_estimators=CONFIG['n_estimators'],
        max_depth=CONFIG['max_depth'],
        random_state=CONFIG['random_state']
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))
    
    return accuracy


def save_artifacts(model, feature_names, target_names, accuracy):
    """Save model and metadata to models/ directory."""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save model pickle
    model_path = models_dir / 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForestClassifier',
        'version': CONFIG['model_version'],
        'training_date': datetime.now().isoformat(),
        'test_accuracy': float(accuracy),
        'hyperparameters': {
            'n_estimators': CONFIG['n_estimators'],
            'max_depth': CONFIG['max_depth'],
            'random_state': CONFIG['random_state']
        },
        'feature_names': feature_names,
        'target_names': list(target_names),
        'sklearn_version': '1.5.2'
    }
    
    metadata_path = models_dir / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Iris Model Training Pipeline")
    print("=" * 60)
    
    # Load data
    X, y, feature_names, target_names = load_data()
    print(f"\nDataset loaded: {len(X)} samples, {len(feature_names)} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train
    model = train_model(X_train, y_train)
    
    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test, target_names)
    
    # Save
    save_artifacts(model, feature_names, target_names, accuracy)
    
    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()