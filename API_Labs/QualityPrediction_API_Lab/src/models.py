from pydantic import BaseModel, Field
from typing import List

class WineData(BaseModel):
    """Input model for wine quality prediction"""
    fixed_acidity: float = Field(..., ge=0, le=20, description="Fixed acidity (g/L)")
    volatile_acidity: float = Field(..., ge=0, le=2, description="Volatile acidity (g/L)")
    citric_acid: float = Field(..., ge=0, le=2, description="Citric acid (g/L)")
    residual_sugar: float = Field(..., ge=0, le=50, description="Residual sugar (g/L)")
    chlorides: float = Field(..., ge=0, le=1, description="Chlorides (g/L)")
    free_sulfur_dioxide: float = Field(..., ge=0, le=100, description="Free SO2 (mg/L)")
    total_sulfur_dioxide: float = Field(..., ge=0, le=300, description="Total SO2 (mg/L)")
    density: float = Field(..., ge=0.9, le=1.1, description="Density (g/mL)")
    pH: float = Field(..., ge=2.5, le=4.5, description="pH level")
    sulphates: float = Field(..., ge=0, le=3, description="Sulphates (g/L)")
    alcohol: float = Field(..., ge=8, le=16, description="Alcohol (%)")

class WineResponse(BaseModel):
    """Response model for wine quality prediction"""
    quality: int = Field(..., description="Predicted wine quality (3-8)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")

class BatchWineData(BaseModel):
    """Batch prediction input"""
    wines: List[WineData] = Field(..., description="List of wine samples")

class BatchWineResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[WineResponse] = Field(..., description="List of predictions")
    
class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str = Field(..., description="Type of model")
    features: List[str] = Field(..., description="List of feature names")
    version: str = Field(..., description="Model version")