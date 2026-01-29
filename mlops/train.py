import os
import subprocess
import mlflow
from prefect import flow
from mlflow.tracking import MlflowClient
from model_training import load_and_prepare_data, run_all_experiments, promote_best_model
import dagshub

dagshub.init(repo_owner='obiezuevictor', repo_name='Cyberbullying-bot', mlflow=True)
DATA_PATH = "../data/training_data_with_history.parquet"
S3_BUCKET='s3://cyberbullying-artifacts-victor-obi/mlflow'
ARTIFACTS_DIR = "../api_service/artifacts"
SVD_COMPONENTS = 128
RANDOM_STATE = 42
EXPERIMENT_NAME = "cyberbullying-detection"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def pull_dvc_data():
    print("Starting DVC Pull from S3...")
    result = subprocess.run(["dvc", "pull", "-v"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Data pulled successfully!")
        print(result.stdout)
    else:
        print("DVC Pull Failed!")
        print(result.stderr)
        raise Exception("DVC Pull failed")
    
def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    bucket = os.getenv("MLFLOW_ARTIFACT_LOCATION", "s3://cyberbullying-artifacts-victor-obi/mlruns")
    
    if not tracking_uri:
        print("⚠️  Warning: MLFLOW_TRACKING_URI not found. Saving locally.")
    else:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"📊 MLflow Tracking URI: {tracking_uri}")
    
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        mlflow.create_experiment(name=EXPERIMENT_NAME, artifact_location=bucket)
        print(f"Created experiment: {EXPERIMENT_NAME}")
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Experiment: {EXPERIMENT_NAME}")
    
    return client

@flow(name="Cyberbullying Detection Training Pipeline")
def main_flow(use_dvc=False, n_trials=50, force_recompute=False):
    print("\n" + "="*60)
    print("CYBERBULLYING DETECTION TRAINING PIPELINE")
    print("="*60 + "\n")
    if use_dvc:
        pull_dvc_data()
    
    client = setup_mlflow()
    
    X_train, X_val, X_test, y_train, y_val, y_test, scale_pos_weight, variance_retained = load_and_prepare_data(force_recompute=force_recompute)
    
    winner_id, winner_uuid, winner_f2, winner_name = run_all_experiments(
        X_train, X_val, X_test, y_train, y_val, y_test, scale_pos_weight, n_trials
    )
    
    promote_best_model(winner_id, winner_uuid, winner_f2, winner_name, client)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print(f"   Best Model: {winner_name}")
    print(f"   F2-Score: {winner_f2:.4f}")
    print(f"   View results: {os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cyberbullying Detection Training")
    parser.add_argument("--mode", choices=["serve", "run"], default="run",
                      help="Run mode: 'serve' for Prefect deployment, 'run' for direct execution")
    parser.add_argument("--dvc", action="store_true", help="Pull data from DVC")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--force-recompute", action='store_true', help="Ignore cache and re-run BERT")
    
    args = parser.parse_args()
    
    if args.mode == "serve":
        print("📡 Deploying Prefect flow - waiting for triggers...")
        main_flow.serve(
            name='cyberbullying-training',
            parameters={"use_dvc": args.dvc, "n_trials": args.trials, 'force_recompute':args.force_recompute}
        )
    else:
        print("🚀 Running training pipeline directly...")
        main_flow(use_dvc=args.dvc, n_trials=args.trials, force_recompute=args.force_recompute)
