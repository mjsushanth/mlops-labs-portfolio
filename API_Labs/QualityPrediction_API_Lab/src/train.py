import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from data import load_wine_data, prepare_data

def train_wine_model():
    """Train Random Forest model on wine quality data"""
    
    print("Loading wine quality dataset...")
    df = load_wine_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model with all components
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    
    model_path = Path("model/wine_quality_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {model_path}")
    
    # Show feature importance
    feature_importance = list(zip(feature_names, model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 Important Features:")
    for feature, importance in feature_importance[:5]:
        print(f"  {feature}: {importance:.4f}")

if __name__ == "__main__":
    train_wine_model()