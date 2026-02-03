import mlflow
import shutil
from load_dotenv import load_dotenv
import os
from transformers import DistilBertModel, DistilBertTokenizer

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "cyberbullying-detection"
STAGE = "Production"

# Output Paths (Where we save files locally)
LOCAL_MODEL_DIR = "baked_model"
LOCAL_BERT_DIR = "baked_bert"
load_dotenv()
def bake_artifacts():
    print(" Starting Bake-In Process...")

    print(f"⬇  Downloading MLflow Artifact: {MODEL_NAME} ({STAGE})...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    if os.path.exists(LOCAL_MODEL_DIR):
        shutil.rmtree(LOCAL_MODEL_DIR)
        
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}/{STAGE}",
        dst_path=LOCAL_MODEL_DIR
    )
    print("✅ MLflow Artifact downloaded.")

    print(f"⬇️  Downloading DistilBERT Weights (~260MB)...")
    if os.path.exists(LOCAL_BERT_DIR):
        shutil.rmtree(LOCAL_BERT_DIR)
        
    DistilBertModel.from_pretrained('distilbert-base-uncased').save_pretrained(LOCAL_BERT_DIR)
    DistilBertTokenizer.from_pretrained('distilbert-base-uncased').save_pretrained(LOCAL_BERT_DIR)
    print("✅ DistilBERT downloaded.")

if __name__ == "__main__":
    bake_artifacts()