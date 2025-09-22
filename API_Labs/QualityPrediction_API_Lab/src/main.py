
from fastapi import FastAPI, HTTPException
from models import WineData, WineResponse, BatchWineData, BatchWineResponse, ModelInfo
from predict import predictor
from data import get_feature_info

import uvicorn


# FastAPI app
# FastAPI - passes instance to ASGI server uvicorn, server uses app isntance to handle web requests and responses.
app = FastAPI(
    title="Wine Quality Prediction API",
    description="Predict wine quality using Random Forest model",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Wine Quality Prediction API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test if model is loaded
        feature_names = predictor.get_feature_names()
        return {"status": "healthy", "model_loaded": True, "features_count": len(feature_names)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/predict", response_model=WineResponse)
async def predict_wine_quality(wine_data: WineData):
    """Predict wine quality for a single wine sample"""
    try:
        prediction = predictor.predict_single(wine_data)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch", response_model=BatchWineResponse) 
async def predict_batch_wine_quality(batch_data: BatchWineData):
    """Predict wine quality for multiple wine samples"""
    try:
        predictions = predictor.predict_batch(batch_data.wines)
        return BatchWineResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the model"""
    try:
        feature_names = predictor.get_feature_names()
        return ModelInfo(
            model_type="RandomForestClassifier",
            features=feature_names,
            version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/features")
async def get_features():
    """Get feature descriptions"""
    return get_feature_info()


"""
Run instructions:
uvicorn app:main --reload
uvicorn main:app --reload --host 127.0.0.1 --port 8000
"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



# --------------------------------------------------------------------------------------------------------------------

""" -- quick tutorial from github readme:
@app.get("/") or @app.post("/predict") - decorator to tell FastAPI. route handlers, responsible for responding to client requests.
decorators - in fastapi, assosciate functions with specific HTTP methods and paths.
app instance - of FastAPI class. core of app, has routes, handlers, middleware, settings.
http methods - GET, POST, etc.
path/endpoints - URL paths that clients use where API is accessible. 

Pydantic model - define structure of request and response data. validate and serialize data.
                 you can do response_model=WineResponse in the decorator.
ASGI server - like uvicorn, handles incoming HTTP requests, passes to FastAPI app, sends back responses.

@app.get("/")                   --> when client sends GET request to root path "/", run root() function.
async def root():
    return {"message": "Hello World"}

@app.post("/predict")           --> when client sends POST request to "/predict", run predict_wine_quality() function.
async def predict_wine_quality(wine_data: WineData):
    return prediction

wine_data: WineData = expects JSON data matching Pydantic model.

async def fast_function():
    await some_io_operation()  # Server can handle other requests while waiting
    return "Done"

FastAPI automatically converts JSON → Python object, validates it, and passes to function. Returns 422 error if validation fails.
Validation: @app.post("/predict", response_model=WineResponse) 
Status codes: 200 (success), 400 (bad request), 500 (server error)


Client → Server Flow:

Client sends HTTP request (browser, app, curl)
    FastAPI matches URL to decorator (@app.post("/predict"))
        Pydantic validates input (converts JSON to Python objects)
            Func runs (ML prediction)
                Response sent back as JSON

- FastAPI sees POST request to /predict
- Converts JSON to WineData object
- Calls predict_wine_quality(wine_data)
- Function runs ML model
- Returns WineResponse as JSON
"""