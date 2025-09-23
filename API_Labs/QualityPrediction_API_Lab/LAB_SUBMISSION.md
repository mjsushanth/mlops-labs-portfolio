# Wine Quality Prediction API - Lab Submission

**Student:** Joel Markapudi  
**Course:** MLOps IE7374  
**Lab:** FastAPI MLOps Implementation  + Streamlit Implementation

## Overview
Implemented a complete MLOps pipeline using FastAPI to serve wine quality predictions via REST API.
Implemented a Streamlit front-end for user interaction. Dashboard has sliders for each feature and buttons to trigger single or batch predictions, code passes the slider values to the FastAPI backend and displays the prediction results.

## Key Modifications from Base Lab

- **Dataset:** Wine Quality Dataset (1,143 samples, 11 features) instead of Iris
- **Model:** Random Forest Classifier
- **Features:** Advanced wine chemistry parameters (alcohol, pH, sulphates, etc.)
- **Environment:** Conda/Mamba approach. YML file is present.

### Architecture
```
wine-quality-api/
├── src/
│   ├── data.py          # Wine data loading
│   ├── train.py         # Random Forest model training
│   ├── models.py        # Pydantic validation models
│   ├── predict.py       # Prediction service logic
│   └── main.py          # FastAPI application
├── model/
│   └── wine_quality_model.pkl
├── data/
│   └── WineQT.csv
├── wine_quality_environment.yml
├── requirements.txt
└── screenshots/         # API testing evidence
```

Update: Structure is updated now to further include Streamlit app. Has streamlit_app/ folder, and results - streamlit/ folder.

### RESULTS:
- All results are recorded in screenshots/ folder, as a way to provide evidence of testing and successful running.

### Model Performance
- **Training Accuracy:** 91.47%
- **Test Accuracy:** 67.69%
- **Top Features:** Alcohol (17.6%), Sulphates (12.8%), Volatile Acidity (10.6%)

### API Endpoints
1. `POST /predict` - Single wine quality prediction
2. `POST /predict_batch` - Batch wine quality prediction
3. `GET /health` - Service health check
4. `GET /model_info` - Model metadata
5. `GET /features` - Feature descriptions

**Single Prediction Example:**
```json
Input: High-quality wine characteristics
Output: {"quality": 6, "confidence": 0.44}
```

**Batch Prediction:** Successfully processed multiple wine samples simultaneously

### SETUP Commands:

Use conda or mamba based installation, YML file is provided.

**To remove, create or update environment:**
mamba env remove -n wine-quality-api
mamba env create -f wine_quality_env.yml
mamba env update -f wine_quality_env.yml

**To activate environment:**
conda activate wine-quality-api

**To run FastAPI server:**
cd src
uvicorn main:app --reload --host 127.0.0.1 --port 8000

**To run Streamlit app:**
cd streamlit_app
conda activate wine-quality-api
streamlit run streamlit_app.py


