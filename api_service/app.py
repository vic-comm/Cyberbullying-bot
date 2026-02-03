import time
import json
import os
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.responses import Response
from mlflow.tracking import MlflowClient

from mlops.feature_store import FeatureStore
from mlops.utils import calculate_text_features
from api_service.schemas import PredictionRequest, PredictionResponse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOG_FILE = LOGS_DIR / "api.log"

LOGS_DIR.mkdir(exist_ok=True)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "cyberbullying-detection")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOGS_PATH = os.getenv("LOGS_PATH", "../data/raw_logs.jsonl")
STAGE = os.getenv("MODEL_STAGE", "Production")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# PROMETHEUS METRICS
# Track request counts by endpoint and status
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

# Track prediction latency
PREDICTION_LATENCY = Histogram(
    'prediction_duration_seconds',
    'Time spent processing prediction',
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5]
)

# Track toxicity predictions
TOXICITY_PREDICTIONS = Counter(
    'toxicity_predictions_total',
    'Total toxicity predictions',
    ['result']  
)

MODEL_VERSION = Gauge(
    'model_version_info',
    'Current model version in production',
    ['version']
)

# Track feature store latency
FEATURE_STORE_LATENCY = Histogram(
    'feature_store_duration_seconds',
    'Time spent fetching features from Redis'
)


# GLOBAL STATE
model: Optional[mlflow.pyfunc.PyFuncModel] = None
model_meta: Dict[str, Any] = {
    "version": "unknown",
    "loaded_at": None,
    "stage": STAGE
}
fs: Optional[FeatureStore] = None

MODEL_TABULAR_FEATURES = [
    'msg_len', 
    'caps_ratio', 
    'personal_pronoun_count', 
    'slur_count',
    'user_bad_ratio_7d', 
    'user_toxicity_trend',
    'channel_toxicity_ratio', 
    'hours_since_last_msg', 
    'is_new_to_channel'
]

MODEL_INT_FEATURES = [
    'msg_len', 
    'personal_pronoun_count', 
    'slur_count', 
    'is_new_to_channel'
]

# LIFESPAN MANAGEMENT
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Manage application startup and shutdown.
#     - Loads model from MLflow on startup
#     - Initializes feature store connection
#     - Handles graceful shutdown
#     """
#     global model, model_meta, fs
    
#     logger.info(" Starting Cyberbullying Detection API...")
#     logger.info(f"MLflow URI: {MLFLOW_TRACKING_URI}")
#     logger.info(f"Model: {EXPERIMENT_NAME} (Stage: {STAGE})")
    
#     # Initialize Feature Store
#     try:
#         fs = FeatureStore(redis_host=REDIS_HOST, redis_port=REDIS_PORT)
#         logger.info(f" Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
#     except Exception as e:
#         logger.error(f" Failed to connect to Redis: {e}")
#         logger.warning("  API will start but feature enrichment will fail")
    
#     # Load Model from MLflow
#     try:
#         mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#         model_uri = f"models:/{EXPERIMENT_NAME}/{STAGE}"
        
#         logger.info(f"Loading model from: {model_uri}")
#         model = mlflow.pyfunc.load_model(model_uri)
        
#         # Fetch model metadata
#         client = MlflowClient()
#         versions = client.get_latest_versions(EXPERIMENT_NAME, stages=[STAGE])
        
#         if versions:
#             model_version = versions[0].version
#             model_meta.update({
#                 "version": model_version,
#                 "loaded_at": datetime.now().isoformat(),
#                 "run_id": versions[0].run_id
#             })
            
#             # Update Prometheus metric
#             MODEL_VERSION.labels(version=model_version).set(1)
            
#             logger.info(f" Model v{model_version} loaded successfully!")
#         else:
#             logger.warning(f"  No model found in '{STAGE}' stage")
            
#     except Exception as e:
#         logger.error(f" Failed to load model: {e}", exc_info=True)
#         logger.warning("  Server starting without model - predictions will fail")
    
#     # Ensure logs directory exists
#     os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
    
#     yield
    
#     # Shutdown
#     logger.info("🛑 Shutting down API...")
#     if fs and fs.redis:
#         fs.redis.close()
#         logger.info("Closed Redis connection")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.
    - PRIORITIZES local 'baked' model (Production safe)
    - Fallback to MLflow Registry (Dev convenience)
    - Initializes feature store
    """
    global model, model_meta, fs
    
    logger.info("🚀 Starting Cyberbullying Detection API...")
    
    # 1. Initialize Feature Store
    try:
        fs = FeatureStore(redis_host=REDIS_HOST, redis_port=REDIS_PORT)
        logger.info(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        logger.warning("⚠️ API will start but feature enrichment will fail")
    
    # 2. Load Model (The Critical Fix)
    try:
        # Check for baked-in model first (Production Path)
        local_path = os.getenv("MODEL_LOCAL_PATH", "/app/baked_model")
        
        if os.path.exists(local_path):
            logger.info(f"📂 Found BAKED model at: {local_path}")
            model = mlflow.pyfunc.load_model(local_path)
            
            model_meta.update({
                "version": "baked-in-prod",
                "loaded_at": datetime.now().isoformat(),
                "source": "local_container"
            })
            logger.info("✅ Loaded model from local container (Offline Mode)")
            
        else:
            # Fallback to Internet Download (Dev Mode)
            logger.warning(f"⚠️ No local model found at {local_path}. Attempting remote download...")
            
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{EXPERIMENT_NAME}/{STAGE}"
            
            logger.info(f"☁️ Loading from MLflow: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Try to fetch version info (Might fail if offline)
            try:
                client = MlflowClient()
                versions = client.get_latest_versions(EXPERIMENT_NAME, stages=[STAGE])
                if versions:
                    model_meta["version"] = versions[0].version
            except Exception:
                model_meta["version"] = "remote-unknown"

            logger.info(f"✅ Loaded model from MLflow Registry")

    except Exception as e:
        logger.error(f"❌ CRITICAL: Failed to load model: {e}", exc_info=True)
        # We generally want the app to crash if the model is missing, 
        # otherwise Kubernetes/Docker thinks it's healthy when it's useless.
        # But for debugging, we let it run.
    
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down API...")
    if fs and fs.redis:
        fs.redis.close()
        logger.info("👋 Closed Redis connection")

# FASTAPI APP INITIALIZATION
app = FastAPI(
    title="Cyberbullying Detection API",
    description="Real-time toxicity detection with MLflow model serving",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MIDDLEWARE
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and track metrics"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    logger.info(
        f"{request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Duration: {process_time:.3f}s"
    )
    
    return response

# EXCEPTION HANDLERS
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else "An unexpected error occurred"
        }
    )

# API ENDPOINTS
@app.get("/")
def root():
    return {
        "service": "Cyberbullying Detection API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "predict": "/predict"
        }
    }

@app.get("/health")
def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "loaded": model is not None,
            "version": model_meta.get("version", "unknown"),
            "stage": model_meta.get("stage", "unknown"),
            "loaded_at": model_meta.get("loaded_at")
        },
        "dependencies": {
            "redis": "unknown",
            "mlflow": "unknown"
        }
    }
    
    if fs:
        try:
            fs.redis.ping()
            health_status["dependencies"]["redis"] = "healthy"
        except Exception as e:
            health_status["dependencies"]["redis"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
    
    try:
        client = MlflowClient()
        client.get_experiment_by_name(EXPERIMENT_NAME)
        health_status["dependencies"]["mlflow"] = "healthy"
    except Exception as e:
        health_status["dependencies"]["mlflow"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Overall health
    if model is None:
        health_status["status"] = "unhealthy"
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )
    
    return health_status

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service unavailable."
        )
    
    request_start = time.time()
    
    try:
        input_dict = request.model_dump()
        static_features = calculate_text_features(request.text)
        input_dict.update(static_features)
        
        # 3. Enrich from Redis (Returns ~5 features)
        # We initialize defaults first in case Redis is down or user is new
        redis_defaults = {
            'user_bad_ratio_7d': 0.0,
            'user_toxicity_trend': 0.0,
            'channel_toxicity_ratio': 0.0,
            'hours_since_last_msg': 24.0, # Default to "been a while"
            'is_new_to_channel': 1        # Default to "new user"
        }
        input_dict.update(redis_defaults)
        # 2. Enrich with user features from Feature Store (if available)
        if fs and request.user_id:
            feature_start = time.time()
            try:
                user_features = fs.get_online_features(
                    feature_group_name="user_toxicity",
                    entity_id=request.user_id,
                    version="prod"
                )
                if user_features:
                    input_dict.update(user_features)
                    logger.debug(f"Enriched with features for user {request.user_id}")
                
                FEATURE_STORE_LATENCY.observe(time.time() - feature_start)
            except Exception as e:
                logger.warning(f"Feature enrichment failed: {e}")
        final_input = { 'text': [request.text] }
        for feature in MODEL_TABULAR_FEATURES:
            raw_val = input_dict.get(feature, 0)
            final_input[feature] = [raw_val] # Just put the value in, we cast later
            
        input_df = pd.DataFrame(final_input)
        
        # --- THE FIX: FORCE PANDAS DTYPES ---
        # This prevents Pandas from "optimizing" 24.0 into 24 (int)
        
        for feature in MODEL_TABULAR_FEATURES:
            if feature in MODEL_INT_FEATURES:
                input_df[feature] = input_df[feature].astype('int64')
            else:
                input_df[feature] = input_df[feature].astype('float64')
            
        input_df = pd.DataFrame(final_input)
        
        prediction_start = time.time()
        prediction, confidence = model.predict(input_df)
        prediction_time = time.time() - prediction_start
        
        # Convert to Python native types
        is_toxic = bool(prediction[0]) if hasattr(prediction, '__iter__') else bool(prediction)
        confidence_score = float(confidence[0]) if hasattr(confidence, '__iter__') else float(confidence)
        # 5. Track metrics
        if confidence_score < 0.5:
            confidence_score = 1.0 - confidence_score
        PREDICTION_LATENCY.observe(prediction_time)
        TOXICITY_PREDICTIONS.labels(
            result="toxic" if is_toxic else "non_toxic"
        ).inc()
        
        # 6. Log prediction for monitoring/retraining
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": request.user_id,
            "text": request.text[:250] + "..." if len(request.text) > 250 else request.text,  # Truncate for privacy
            "prediction": int(is_toxic),
            "confidence": confidence_score,
            "model_version": model_meta["version"],
            "processing_time_ms": round((time.time() - request_start) * 1000, 2),
            "features_enriched": bool(fs and request.user_id)
        }
        
        # Async logging (could be improved with async queue)
        try:
            with open(LOGS_PATH, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")
        
        # 7. Build response
        total_time = (time.time() - request_start) * 1000
        
        return PredictionResponse(
            is_toxic=is_toxic,
            confidence=confidence_score,
            model_version=model_meta["version"],
            processing_time_ms=round(total_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/reload-model")
def reload_model():
    """
    Hot-reload the production model without restarting the server.
    Useful for zero-downtime deployments.
    
    Note: In production, this should be protected with authentication.
    """
    global model, model_meta
    
    logger.info("Model reload requested...")
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{EXPERIMENT_NAME}/{STAGE}"
        
        # Load new model
        new_model = mlflow.pyfunc.load_model(model_uri)
        
        # Get metadata
        client = MlflowClient()
        versions = client.get_latest_versions(EXPERIMENT_NAME, stages=[STAGE])
        
        if versions:
            new_version = versions[0].version
            
            old_version = model_meta.get("version", "unknown")
            model = new_model
            model_meta.update({
                "version": new_version,
                "loaded_at": datetime.now().isoformat(),
                "run_id": versions[0].run_id
            })
            
            MODEL_VERSION.labels(version=new_version).set(1)
            
            logger.info(f" Model reloaded: v{old_version} → v{new_version}")
            
            return {
                "status": "success",
                "old_version": old_version,
                "new_version": new_version,
                "reloaded_at": model_meta["loaded_at"]
            }
        else:
            raise ValueError(f"No model found in '{STAGE}' stage")
            
    except Exception as e:
        logger.error(f"Model reload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level=LOG_LEVEL.lower(),
        access_log=True
    )