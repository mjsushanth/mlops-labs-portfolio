"""
Wine Quality Prediction Service
"""
import pickle
import numpy as np
from pathlib import Path
from models import WineData, WineResponse

class WinePredictionService:
    def __init__(self, model_path: str = "model/wine_quality_model.pkl"):
        self.model_path = Path(model_path)
        self.model_data = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and components"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        print(f"Model loaded from {self.model_path}")
    
    def predict_single(self, wine_data: WineData) -> WineResponse:
        """Predict wine quality for a single sample"""
        if self.model_data is None:
            raise ValueError("Model not loaded")
        
        # Convert input to array
        features = np.array([[
            wine_data.fixed_acidity,
            wine_data.volatile_acidity,
            wine_data.citric_acid,
            wine_data.residual_sugar,
            wine_data.chlorides,
            wine_data.free_sulfur_dioxide,
            wine_data.total_sulfur_dioxide,
            wine_data.density,
            wine_data.pH,
            wine_data.sulphates,
            wine_data.alcohol
        ]])
        
        # Scale features
        features_scaled = self.model_data['scaler'].transform(features)
        
        # Make prediction
        prediction = self.model_data['model'].predict(features_scaled)[0]
        probabilities = self.model_data['model'].predict_proba(features_scaled)[0]
        
        # Get confidence (max probability)
        confidence = float(np.max(probabilities))
        
        return WineResponse(
            quality=int(prediction),
            confidence=confidence
        )
    
    def predict_batch(self, wine_samples: list) -> list:
        """Predict wine quality for multiple samples"""
        return [self.predict_single(wine) for wine in wine_samples]
    
    def get_feature_names(self):
        """Get feature names from the model"""
        if self.model_data is None:
            raise ValueError("Model not loaded")
        return self.model_data['feature_names']

# Global prediction service instance
predictor = WinePredictionService()