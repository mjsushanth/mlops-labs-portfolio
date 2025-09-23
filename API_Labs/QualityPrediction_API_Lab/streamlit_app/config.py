"""
Configuration file for Streamlit Wine Quality Prediction App
"""

import os
from pathlib import Path

# FastAPI Backend Configuration
FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"

# Model and Data Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTAPI_WINE_MODEL_LOCATION = PROJECT_ROOT / 'model' / 'wine_quality_model.pkl'
WINE_DATA_LOCATION = PROJECT_ROOT / 'data' / 'winequality-red.csv'


"""
- this acts like user input configuration help texts, min, max, default values. and step sizes for sliders.
- WINE_FEATURES will be used in streamlit_app.py to create sliders dynamically.
"""
# Wine Quality Features Configuration
WINE_FEATURES = {
    "fixed_acidity": {
        "min": 4.6,
        "max": 15.9,
        "default": 8.32,
        "step": 0.1,
        "help": "Non-volatile acids that do not evaporate (g/dm³)"
    },
    "volatile_acidity": {
        "min": 0.12,
        "max": 1.58,
        "default": 0.53,
        "step": 0.01,
        "help": "Acetic acid amount, too high leads to vinegar taste (g/dm³)"
    },
    "citric_acid": {
        "min": 0.0,
        "max": 1.0,
        "default": 0.27,
        "step": 0.01,
        "help": "Adds freshness and flavor to wines (g/dm³)"
    },
    "residual_sugar": {
        "min": 0.9,
        "max": 15.5,
        "default": 2.54,
        "step": 0.1,
        "help": "Sugar remaining after fermentation (g/dm³)"
    },
    "chlorides": {
        "min": 0.012,
        "max": 0.611,
        "default": 0.087,
        "step": 0.001,
        "help": "Amount of salt in wine (g/dm³)"
    },
    "free_sulfur_dioxide": {
        "min": 1.0,
        "max": 72.0,
        "default": 15.87,
        "step": 1.0,
        "help": "SO2 that prevents microbial growth (mg/dm³)"
    },
    "total_sulfur_dioxide": {
        "min": 6.0,
        "max": 289.0,
        "default": 46.47,
        "step": 1.0,
        "help": "Total amount of SO2, preservative (mg/dm³)"
    },
    "density": {
        "min": 0.99007,
        "max": 1.00369,
        "default": 0.99674,
        "step": 0.00001,
        "help": "Wine density, related to alcohol and sugar (g/cm³)"
    },
    "pH": {
        "min": 2.74,
        "max": 4.01,
        "default": 3.31,
        "step": 0.01,
        "help": "Acidity level, 0 very acidic to 14 very basic"
    },
    "sulphates": {
        "min": 0.33,
        "max": 2.0,
        "default": 0.66,
        "step": 0.01,
        "help": "Wine additive, antimicrobial and antioxidant (g/dm³)"
    },
    "alcohol": {
        "min": 8.4,
        "max": 14.9,
        "default": 10.42,
        "step": 0.1,
        "help": "Alcohol percentage by volume (%)"
    }
}

# Quality mapping for display
QUALITY_LABELS = {
    0: "Very Poor (3)",
    1: "Poor (4)", 
    2: "Below Average (5)",
    3: "Average (6)",
    4: "Above Average (7)",
    5: "Good (8)"
}

# App Configuration
APP_CONFIG = {
    "page_title": "Wine Quality Prediction",
    "page_icon": "🍷",
    "layout": "wide"
}