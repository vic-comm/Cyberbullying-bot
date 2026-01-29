import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("https://dagshub.com/obiezuevictor/Cyberbullying-bot.mlflow")

client = MlflowClient()

# Try to list artifacts
run_id = "920e9c864ba34594a0a7f654dae893de"  # Your winning LightGBM run

try:
    artifacts = client.list_artifacts(run_id)
    print("✅ DagsHub can access your S3 bucket!")
    print(f"   Artifacts: {[a.path for a in artifacts]}")
except Exception as e:
    print(f"❌ DagsHub cannot access S3: {e}")
    print("   → Check AWS credentials in DagsHub settings")