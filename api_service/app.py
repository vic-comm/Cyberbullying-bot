# app.py - Created for Cyberbullying Bot Project
import time
import json
import os
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from datetime import datetime

# Import schemas
from schemas import PredictionRequest, PredictionResponse

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "cyberbullying-detection"
LOGS_PATH = "../data/raw_logs.jsonl"

# Global Model Variable
model = None
model_meta = {"version": "unknown"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the Production Model on Startup.
    This prevents loading it for every single request (which would be slow).
    """
    global model, model_meta
    
    print("🚀 API Starting... Connecting to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # Load the specific model tagged as "Production"
        model_uri = f"models:/{EXPERIMENT_NAME}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Get version info
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(EXPERIMENT_NAME, stages=["Production"])
        if versions:
            model_meta["version"] = versions[0].version
            
        print(f"✅ Production Model v{model_meta['version']} Loaded Successfully!")
        
    except Exception as e:
        print(f"❌ FAILED to load model: {e}")
        print("⚠️  Server will start, but predictions will fail until fixed.")
    
    yield
    print("🛑 API Shutting Down...")

# Initialize App
app = FastAPI(title="Cyberbullying Detection API", version="1.0", lifespan=lifespan)

@app.get("/health")
def health_check():
    """Simple check to see if API is alive"""
    return {"status": "healthy", "model_loaded": model is not None, "version": model_meta["version"]}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Main Inference Endpoint
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    start_time = time.time()
    
    try:
        # 1. Convert Request to DataFrame
        # The wrapper expects a DataFrame with specific columns
        input_data = pd.DataFrame([request.model_dump()])
        
        # 2. Predict
        # The Custom Wrapper handles SVD, Scaling, and Threshold internally!
        prediction = model.predict(input_data)[0] # Returns 0 or 1
        
        # (Optional) If your wrapper supports predict_proba, use that for confidence
        # For now, we mock confidence or you can update wrapper to return it
        confidence = 0.95 if prediction == 1 else 0.99 
        
        # 3. Log for Feedback Loop
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": request.user_id,
            "text": request.text,
            "prediction": int(prediction),
            "model_version": model_meta["version"],
            # Log inputs so we can retrain later
            "features": request.model_dump()
        }
        
        # Append to logs file (Thread-safe enough for low volume)
        with open(LOGS_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        # 4. Return Response
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "is_toxic": bool(prediction),
            "confidence": confidence,
            "model_version": model_meta["version"],
            "processing_time_ms": round(processing_time, 2)
        }
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run locally for debugging
    uvicorn.run(app, host="0.0.0.0", port=8000)