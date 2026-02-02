import os
import sys
import subprocess
from pathlib import Path
from typing import Optional
import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
load_dotenv.load_dotenv()
from model_training import Config, ModelType, load_and_prepare_data, run_all_experiments, promote_best_model
import dagshub

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def pull_dvc_data() -> bool:
    print("\n" + "="*60)
    print("PULLING DATA FROM DVC")
    print("="*60)
    
    try:
        result = subprocess.run(["dvc", "pull", "-v"], capture_output=True, text=True, timeout=300 )
        
        if result.returncode == 0:
            print("  Data pulled successfully")
            if result.stdout:
                print(f"\n{result.stdout}")
            return True
        else:
            print("  DVC Pull Failed")
            if result.stderr:
                print(f"\nError output:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   DVC pull timed out after 5 minutes")
        return False
    except FileNotFoundError:
        print("   DVC not found - please install with: pip install dvc[s3]")
        return False
    except Exception as e:
        print(f"   Unexpected error: {e}")
        return False


def setup_mlflow(config: Config) -> MlflowClient:
    print("\n" + "="*60)
    print("MLFLOW SETUP")
    print("="*60)
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    artifact_location = os.getenv("MLFLOW_ARTIFACT_LOCATION", config.S3_BUCKET)
    
    try:
        dagshub_repo = os.getenv("DAGSHUB_REPO")
        dagshub_owner = os.getenv("DAGSHUB_OWNER")
        
        if dagshub_repo and dagshub_owner:
            dagshub.init(
                repo_owner=dagshub_owner,
                repo_name=dagshub_repo,
                mlflow=True
            )
            print(f"   Initialized DagsHub tracking")
            print(f"   Repository: {dagshub_owner}/{dagshub_repo}")
        else:
            print("     DagsHub credentials not found in environment")
    except Exception as e:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"    Tracking URI: {tracking_uri}")
    
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(config.EXPERIMENT_NAME)
        
        if not experiment:
            experiment_id = mlflow.create_experiment(
                name=config.EXPERIMENT_NAME,
                artifact_location=artifact_location
            )
            print(f"   Created experiment: {config.EXPERIMENT_NAME}")
            print(f"   Artifact location: {artifact_location}")
        else:
            experiment_id = experiment.experiment_id
            print(f"   Using existing experiment: {config.EXPERIMENT_NAME}")
        
        mlflow.set_experiment(config.EXPERIMENT_NAME)
        
    except Exception as e:
        print(f"   Experiment setup failed: {e}")
        raise
    
    print("="*60)
    
    return client


def validate_environment():
    print("\n" + "="*60)
    print("ENVIRONMENT VALIDATION")
    print("="*60)
    
    issues = []
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('sklearn', 'scikit-learn'),
        ('mlflow', 'MLflow'),
        ('optuna', 'Optuna')
    ]
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   {name} available")
        except ImportError:
            print(f"   {name} not found")
            issues.append(f"Install {name}")
    
    config = Config()
    if config.DATA_PATH.exists():
        print(f"   Training data found: {config.DATA_PATH}")
    else:
        print(f"   Training data not found: {config.DATA_PATH}")
        issues.append("Run DVC pull or provide training data")
    
    # Check for write permissions
    try:
        config.setup_directories()
        print(f"   Directories created/verified")
    except Exception as e:
        print(f"   Cannot create directories: {e}")
        issues.append("Check file system permissions")
    
    print("="*60)
    
    if issues:
        print("\n  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("\nEnvironment validation passed")
    return True


# MAIN PIPELINE
@flow(name="Cyberbullying Detection Training Pipeline", log_prints=True)
def main_flow(use_dvc: bool = False, n_trials: int = 50, force_recompute: bool = False, models: Optional[str] = None):
    print("\n" + "="*80)
    print(" " * 20 + "CYBERBULLYING DETECTION")
    print(" " * 22 + "TRAINING PIPELINE")
    print("="*80 + "\n")
    
    if not validate_environment():
        print("\nEnvironment validation failed - please fix issues above")
        sys.exit(1)
    
    if use_dvc:
        success = pull_dvc_data()
        if not success:
            print("\nDVC pull failed - cannot proceed")
            sys.exit(1)
    
    config = Config()
    config.setup_directories()
    
    client = setup_mlflow(config)
    
    models_to_train = None
    if models:
        model_map = {
            'xgboost': ModelType.XGBOOST,
            'lightgbm': ModelType.LIGHTGBM,
            'logistic': ModelType.LOGISTIC
        }
        
        requested_models = [m.strip().lower() for m in models.split(',')]
        models_to_train = []
        
        for model_name in requested_models:
            if model_name in model_map:
                models_to_train.append(model_map[model_name])
            else:
                print(f"    Unknown model: {model_name}")
        
        if not models_to_train:
            print("   No valid models specified - training all models")
            models_to_train = None
        else:
            print(f"\n   Training models: {[m.value for m in models_to_train]}")
    
    try:
        training_data = load_and_prepare_data(force_recompute=force_recompute)
        
        winner_id, winner_uuid, winner_f2, winner_name = run_all_experiments(
            data=training_data,
            n_trials=n_trials,
            models_to_train=models_to_train
        )
        
        # Step 3: Model Promotion
        version = promote_best_model(
            winner_id=winner_id,
            winner_uuid=winner_uuid,
            winner_f2=winner_f2,
            winner_name=winner_name,
            client=client,
            experiment_name=config.EXPERIMENT_NAME
        )
        
        # Pipeline Summary
        print("\n" + "="*80)
        print(" " * 30 + "PIPELINE COMPLETE")
        print("="*80)
        print(f"\n   🏆 Best Model: {winner_name}")
        print(f"   📊 F2-Score: {winner_f2:.4f}")
        print(f"   📦 Model Version: {version}")
        print(f"   🔗 Run ID: {winner_id}")
        
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        print(f"\n   View results: {tracking_uri}")
        print("\n" + "="*80 + "\n")
        
        return {
            'model_name': winner_name,
            'f2_score': winner_f2,
            'run_id': winner_id,
            'version': version
        }
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ==========================================
# CLI ENTRY POINT
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cyberbullying Detection Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default settings
  python train.py
  
  # Pull data from DVC and run with 100 trials
  python train.py --dvc --trials 100
  
  # Force recompute embeddings and train only XGBoost
  python train.py --force-recompute --models xgboost
  
  # Deploy as Prefect flow
  python train.py --mode serve --trials 50
        """
    )
    
    parser.add_argument("--mode", choices=["serve", "run"], default="run", help="Execution mode: 'serve' for Prefect deployment, 'run' for direct execution")
    parser.add_argument("--dvc", action="store_true", help="Pull data from DVC before training")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials for hyperparameter tuning (default: 50)")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore cache and regenerate BERT embeddings")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models to train: xgboost,lightgbm,logistic (default: all)")
    
    args = parser.parse_args()
    
    if args.mode == "serve":
        print("="*80)
        print(" " * 25 + "PREFECT DEPLOYMENT MODE")
        print("="*80)
        print("\n Deploying Prefect flow - waiting for triggers...")
        print("   Use Prefect UI or CLI to trigger runs\n")
        
        main_flow.serve(
            name='cyberbullying-training',
            parameters={"use_dvc": args.dvc, "n_trials": args.trials, "force_recompute": args.force_recompute, "models": args.models}
        )
    else:
        print(" Running training pipeline directly...\n")
        
        result = main_flow(use_dvc=args.dvc, n_trials=args.trials, force_recompute=args.force_recompute, models=args.models)
        
        if result:
            print(f"\n Training completed successfully!")
            print(f"   Model: {result['model_name']}")
            print(f"   F2 Score: {result['f2_score']:.4f}")