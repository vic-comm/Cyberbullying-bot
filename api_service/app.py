import time
import json
import os
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from datetime import datetime
import boto3
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.tasks import repeat_every
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.responses import Response
from mlflow.tracking import MlflowClient
from mlops.feature_store import FeatureStore
from mlops.utils import calculate_text_features
from api_service.schemas import PredictionRequest, PredictionResponse
from api_service.explainer import ToxicityExplainer
from pathlib import Path
from dotenv import load_dotenv
import contextlib
import asyncio
from scripts import sync_bot_to_logs
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOG_FILE = LOGS_DIR / "api.log"

LOGS_DIR.mkdir(exist_ok=True)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "cyberbullying-detection")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOGS_PATH = os.getenv("LOGS_PATH", "data/raw_logs.jsonl")
STAGE = os.getenv("MODEL_STAGE", "Production")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
BUCKET_NAME = os.getenv("BUCKET_NAME") 
S3_LOGS_KEY = os.getenv("S3_LOGS_KEY", "logs/raw_logs.jsonl") 
REDIS_URL = os.getenv("REDIS_URL")

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
explainer_service = None

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_meta, fs, explainer_service
    
    logger.info("🚀 Starting Cyberbullying Detection API...")
    
    # -------------------------------------------------
    # 1. Initialize Feature Store
    # -------------------------------------------------
    try:
        fs = FeatureStore(redis_url=REDIS_URL)
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        logger.warning("⚠️ API will start but feature enrichment will fail")
    
    
    try:
        logger.info("☁️ Loading model directly from DagsHub/MLflow...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{EXPERIMENT_NAME}/{STAGE}"
        
        logger.info(f"Downloading from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        try:
            client = MlflowClient()
            versions = client.get_latest_versions(EXPERIMENT_NAME, stages=[STAGE])
            if versions:
                model_meta["version"] = versions[0].version
        except Exception:
            model_meta["version"] = "remote-unknown"

        logger.info(f"✅ Loaded model v{model_meta['version']} from MLflow Registry")

    except Exception as e:
        logger.error(f"❌ CRITICAL: Failed to load model from DagsHub: {e}", exc_info=True)

    if model is not None:
        try:
            explainer_service = ToxicityExplainer(
                model_pipeline=model, 
                feature_calculator=calculate_text_features,
                model_tabular_features=MODEL_TABULAR_FEATURES,
                model_int_features=MODEL_INT_FEATURES
            )
            logger.info("✅ Explainer initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize explainer: {e}", exc_info=True)
    else:
        logger.warning("⚠️ Skipping explainer initialization (model not loaded)")
    
    scheduler_task = asyncio.create_task(run_periodic_sync())
    logger.info(" Periodic sync scheduler started")

    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)

    try:
        yield
    finally:
        logger.info(" Shutting down API...")

        # Stop scheduler cleanly
        scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scheduler_task
        logger.info(" Scheduler stopped")

        # Close Redis
        if fs and fs.redis:
            fs.redis.close()
            logger.info(" Closed Redis connection")


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
            final_input[feature] = [raw_val] 
            
        input_df = pd.DataFrame(final_input)
        
        for feature in MODEL_TABULAR_FEATURES:
            if feature in MODEL_INT_FEATURES:
                input_df[feature] = input_df[feature].astype('int64')
            else:
                input_df[feature] = input_df[feature].astype('float64')
            
        
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

def upload_logs_to_s3():
    if not BUCKET_NAME:
        return

    s3 = boto3.client('s3')
    
    
    try:
        # 1. Download existing S3 logs (if they exist)
        try:
            s3.download_file(BUCKET_NAME, S3_LOGS_KEY, LOGS_PATH)
            # Read existing hashes to avoid duplicates
            existing_hashes = set()
            with open(LOGS_PATH, 'r') as f:
                for line in f:
                    try: 
                        log = json.loads(line)
                        if 'text_hash' in log: existing_hashes.add(log['text_hash'])
                    except: continue
        except Exception:
            # File doesn't exist in S3 yet (first run)
            existing_hashes = set()
            open(LOGS_PATH, 'w').close() # Create empty file

        # 2. Read new local logs
        if not os.path.exists(LOGS_PATH):
            return

        new_records = []
        with open(LOGS_PATH, 'r') as f:
            for line in f:
                try:
                    log = json.loads(line)
                    # Only add if not already in S3
                    if log.get('text_hash') not in existing_hashes:
                        new_records.append(line.strip())
                except: continue
        
        if not new_records:
            print("✅ S3 is already up to date.")
            return

        # 3. Append new records to the temp S3 file
        with open(LOGS_PATH, 'a') as f:
            for record in new_records:
                f.write(record + '\n')

        # 4. Upload the combined file back to S3
        s3.upload_file(LOGS_PATH, BUCKET_NAME, S3_LOGS_KEY)
        print(f"✅ Appended {len(new_records)} new records to S3.")
        
        # Cleanup
        if os.path.exists(LOGS_PATH):
            os.remove(LOGS_PATH)

    except Exception as e:
        print(f"❌ S3 Sync failed: {e}")

async def run_periodic_sync():
    while True:
        try:
            logger.info("⏳ Waiting 12 hours for next sync...")
            await asyncio.sleep(43200)  
            
            logger.info("🔄 Running Sync & Upload...")
            
            records = await asyncio.to_thread(sync_bot_to_logs, lookback_hours=24)
            if records > 0:
                logger.info(f"   Synced {records} new records.")
            
            await asyncio.to_thread(upload_logs_to_s3)
            
        except asyncio.CancelledError:
            logger.info("Sync task cancelled.")
            break
        except Exception as e:
            logger.error(f"Background sync failed: {e}", exc_info=True)
            await asyncio.sleep(60) 

@app.post("/reload-model")
def reload_model():
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

class ExplanationRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    num_features: int = 6  
    
    class Config:
        extra = "allow"
        
@app.post("/explain")
async def explain_text(payload: ExplanationRequest):
    """
    Returns LIME explanation for a given text using the SAME features as /predict
    Body: {
        "text": "You are stupid",
        "user_id": "user_123",
        "channel_id": "channel_456",
        "msg_len": 15,
        ... (any other features you have)
    }
    """
    if explainer_service is None:
        logger.error("❌ /explain called but explainer_service is None")
        raise HTTPException(
            status_code=503,
            detail="Explainer not initialized. Model may have failed to load."
        )
    
    text = payload.text
    logger.info(f"DEBUG /explain called with text: '{text[:50]}'")
    logger.info(f"DEBUG explainer_service type: {type(explainer_service)}")
    logger.info(f"DEBUG explainer_service.pipeline: {explainer_service.pipeline}")
    if not text:
        return {"error": "No text provided"}
    
    request_dict = payload.model_dump()    
    static_features = calculate_text_features(text)
    request_dict.update(static_features)
    
    redis_defaults = {
        'user_bad_ratio_7d': 0.0,
        'user_toxicity_trend': 0.0,
        'channel_toxicity_ratio': 0.0,
        'hours_since_last_msg': 24.0,
        'is_new_to_channel': 1
    }
    request_dict.update(redis_defaults)
    
    user_id = payload.user_id
    if fs and user_id:
        try:
            user_features = fs.get_online_features(
                feature_group_name="user_toxicity",
                entity_id=user_id,
                version="prod"
            )
            if user_features:
                request_dict.update(user_features)
                logger.debug(f"Enriched explanation with features for user {user_id}")
        except Exception as e:
            logger.warning(f"Feature enrichment failed for explanation: {e}")
    
    explanation = explainer_service.explain(text, request_dict)
    return explanation

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level=LOG_LEVEL.lower(),
        access_log=True
    )