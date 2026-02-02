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

import pandas as pd
import numpy as np
import torch
import joblib
import os
import json
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    fbeta_score, classification_report, precision_score, accuracy_score, f1_score,
    recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
# MLflow imports
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import os
import time
# Optuna imports

# from prefect import task
try:
    from prefect import task
except ImportError:
    # If prefect is missing (inside API container), define a dummy decorator
    # so the code doesn't crash when reading the @task lines.
    def task(log_prints=True):
        def decorator(func):
            return func
        return decorator
# ==========================================
# CONFIGURATION
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(SCRIPT_DIR, "../data/training_data_with_history.parquet")
EMBEDDING_PATH = os.path.join(SCRIPT_DIR, "../cache")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "../api_service/artifacts")
S3_BUCKET='s3://cyberbullying-artifacts-victor-obi/mlflow'

SVD_COMPONENTS = 128
RANDOM_STATE = 42
EXPERIMENT_NAME = "cyberbullying-detection"

TABULAR_FEATURES = [
    'msg_len', 'caps_ratio', 'personal_pronoun_count', 'slur_count',
    'user_bad_ratio_7d', 'user_toxicity_trend',
    'channel_toxicity_ratio', 'hours_since_last_msg', 'is_new_to_channel'
]

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


class CyberBullyingModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        
        import joblib
        from transformers import DistilBertTokenizer, DistilBertModel

        # Load artifacts from the MLflow bundle
        self.model = joblib.load(context.artifacts["model_path"])
        self.svd = joblib.load(context.artifacts["svd_path"])
        self.scaler = joblib.load(context.artifacts["scaler_path"])
        self.threshold = joblib.load(context.artifacts["threshold_path"])
        
        # Load DistilBERT (Download once to cache)
        from transformers import DistilBertTokenizer, DistilBertModel
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.bert_model.eval()

    def _get_bert_embeddings(self, text_list):
        import torch
        from transformers import DistilBertTokenizer, DistilBertModel

        import numpy as np
        
        # (Same logic as your training script)
        inputs = self.tokenizer(
            text_list, padding=True, truncation=True, 
            max_length=128, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def predict(self, context, model_input):
        """
        The Unified Predict Function.
        Input: A pandas DataFrame with ['text', 'msg_len', 'caps_ratio', ...]
        Output: Binary classes [0, 1, 1, 0]
        """
        import numpy as np
        
        
        text_data = model_input['text'].astype(str).tolist()
        
        vectors_768 = self._get_bert_embeddings(text_data)
        vectors_128 = self.svd.transform(vectors_768)
        
        
        tabular_data = model_input.drop(columns=['text']).values
        tabular_scaled = self.scaler.transform(tabular_data)
        
        X_final = np.hstack([vectors_128, tabular_scaled])
        
        probs = self.model.predict_proba(X_final)[:, 1]
        predictions = (probs >= self.threshold).astype(int)
        
        return predictions, probs
    
# ==========================================
# TEXT EMBEDDINGS
# ==========================================
def get_bert_embeddings(text_list, batch_size=32):
    """
    Generate DistilBERT embeddings in batches.
    Returns numpy array of shape (N, 768).
    """
    print("   -> Loading DistilBERT Model...")
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  
    else:
        device = torch.device("cpu")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    model.eval()
    
    total = len(text_list) 
    
    # ✅ Pre-allocate array (memory efficient)
    all_embeddings = np.zeros((total, 768), dtype=np.float32)
    
    print(f"   -> Vectorizing {total} messages on {device}...")
    
    for i in range(0, total, batch_size):
        batch_text = text_list[i : i+batch_size]
        
        inputs = tokenizer(
            batch_text, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        current_batch_len = len(batch_vectors)
        all_embeddings[i : i + current_batch_len] = batch_vectors
        
        
        del inputs, outputs, batch_vectors
        
        if i % 1000 == 0 and i > 0:
            print(f"      Processed {i}/{total}...")
            import gc
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
    
    del model, tokenizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    
    return all_embeddings 

# ==========================================
# DATA PREPARATION
# ==========================================
@task(log_prints=True)
def load_and_prepare_data(force_recompute=False):
    from cache import EmbeddingCache
    """Load data and prepare features"""
    print("\n1. Loading Data...")
    df = pd.read_parquet(DATA_PATH)
    
    print(f"   Shape: {df.shape}")
    class_dist = df['label'].value_counts(normalize=True)
    print(f"   Class Balance:")
    print(f"      Safe (0): {class_dist[0]*100:.2f}%")
    print(f"      Toxic (1): {class_dist[1]*100:.2f}%")
    
    scale_pos_weight = (df['label'] == 0).sum() / (df['label'] == 1).sum()
    print(f"   Scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Text Processing
    print("\n2. Text Processing with DistilBERT...")
    
    # CACHE_DIR = "/Users/chidera/Projects/cyberbullying-bot/cache"
    CACHE_DIR = os.path.join(SCRIPT_DIR, "../cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    LOCAL_EMBEDDINGS = os.path.join(CACHE_DIR, 'bert_embeddings.pkl')
    LOCAL_METADATA = os.path.join(CACHE_DIR, 'cache_metadata.json')
    print(LOCAL_EMBEDDINGS)
    print(LOCAL_METADATA)
    has_local_cache = os.path.exists(LOCAL_EMBEDDINGS) and os.path.exists(LOCAL_METADATA)
    s3_bucket = S3_BUCKET  

    if has_local_cache:
        print("   📁 Local cache detected - using local-only mode")
        use_s3 = False
    else:
        print("   🪣 No local cache - enabling S3 sync")
        use_s3 = True
    if os.path.exists(os.path.join(EMBEDDING_PATH, 'bert_embeddings.pkl')):
        s3_bucket = None
    cache = EmbeddingCache(
        cache_dir=CACHE_DIR,
        s3_bucket=s3_bucket, 
        use_s3=use_s3,
        s3_prefix="embeddings-cache"
    )

    if not force_recompute:
        print("    Using cached embeddings (data unchanged)")
        vectors_768, cache_metadata = cache.load_embeddings()
    else:
        if force_recompute: 
            print("    Force recompute enabled - generating new embeddings")
        else:
            print("    Data changed - generating new embeddings")
        
        raw_text = df['text'].astype(str).tolist()
        vectors_768 = get_bert_embeddings(raw_text)
        
        # Save to cache (both local and S3)
        cache.save_embeddings(
            vectors_768, 
            DATA_PATH,
            additional_info={'num_texts': len(raw_text)}
        )

    # raw_text = df['text'].astype(str).tolist()
    # vectors_768 = get_bert_embeddings(raw_text)
    
    # Dimensionality Reduction
    print(f"   -> Applying TruncatedSVD (768 -> {SVD_COMPONENTS})...")
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    vectors_reduced = svd.fit_transform(vectors_768)
    
    variance_retained = svd.explained_variance_ratio_.sum()
    print(f"   -> Variance Retained: {variance_retained*100:.2f}%")
    
    # Save SVD transformer
    joblib.dump(svd, f"{ARTIFACTS_DIR}/svd_transformer.pkl")
    
    # Tabular Processing
    print("\n3. Processing Tabular Features...")
    tabular_data = df[TABULAR_FEATURES].values
    
    scaler = StandardScaler()
    tabular_scaled = scaler.fit_transform(tabular_data)
    
    # Save scaler
    joblib.dump(scaler, f"{ARTIFACTS_DIR}/scaler.pkl")
    
    # Feature Fusion
    print("\n4. Combining Features...")
    X = np.hstack([vectors_reduced, tabular_scaled])
    y = df['label'].values
    
    print(f"   Final Shape: {X.shape}")
    print(f"   - Text features: {vectors_reduced.shape[1]}")
    print(f"   - Behavioral features: {tabular_scaled.shape[1]}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"   Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scale_pos_weight, variance_retained

# ==========================================
# METRICS CALCULATION
# ==========================================
def calculate_metrics(y_true, y_pred, y_proba):
    return {
        'f2_score': fbeta_score(y_true, y_pred, beta=2),
        'f1_score': fbeta_score(y_true, y_pred, beta=1),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
    }

def optimize_threshold(model, X_val, y_val, beta=2):
    y_proba = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    f_scores = [fbeta_score(y_val, (y_proba >= t).astype(int), beta=beta) for t in thresholds]
    
    optimal_idx = np.argmax(f_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f_score = f_scores[optimal_idx]
    
    return optimal_threshold, optimal_f_score

# ==========================================
# VISUALIZATION
# ==========================================
def log_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe', 'Toxic'])
    disp.plot(cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    
    cm_path = f"{ARTIFACTS_DIR}/confusion_matrix_{model_name}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm_path

def log_feature_importance(model, model_name):
    if not hasattr(model, 'feature_importances_'):
        return None
    
    feature_names = [f'svd_{i}' for i in range(SVD_COMPONENTS)] + TABULAR_FEATURES
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Feature Importances - {model_name}')
    plt.tight_layout()
    
    fi_path = f"{ARTIFACTS_DIR}/feature_importance_{model_name}.png"
    plt.savefig(fi_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save CSV
    csv_path = f"{ARTIFACTS_DIR}/feature_importance_{model_name}.csv"
    importance_df.to_csv(csv_path, index=False)
    
    return fi_path, csv_path

# ==========================================
# OPTUNA OBJECTIVES
# ==========================================
def create_xgboost_objective(X_train, y_train, X_val, y_val, scale_pos_weight):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'aucpr',
            'random_state': RANDOM_STATE,
            'n_jobs': 1
        }
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)           # Extra Info
        recall = recall_score(y_val, preds)          
        f_score = f1_score(y_val, preds)

        mlflow.log_metric('f1_score', f_score)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", recall)
        return fbeta_score(y_val, preds, beta=2)
    
    return objective

def create_logistic_objective(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.001, 10.0, log=True),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'max_iter': 1000,
            'random_state': RANDOM_STATE
        }
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # 3. Evaluate
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)           
        recall = recall_score(y_val, preds)          
        f_score = f1_score(y_val, preds)
        
        mlflow.log_metric('f1_score', f_score)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", recall)

        return fbeta_score(y_val, preds, beta=2)
    
    return objective
def create_lightgbm_objective(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'class_weight': 'balanced',
            'random_state': RANDOM_STATE,
            'verbose': -1,
            'n_jobs': 1
        }
        
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)           
        recall = recall_score(y_val, preds)         
        f_score = f1_score(y_val, preds)

        mlflow.log_metric('f1_score', f_score)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", recall)
        return fbeta_score(y_val, preds, beta=2)
    
    return objective

# ==========================================
# MODEL TRAINING
# ==========================================
@task(log_prints=True)
def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test,scale_pos_weight, n_trials=50):
    import optuna
    from optuna.integration.mlflow import MLflowCallback

    print("\n Tuning XGBoost with Optuna...")
    
    mlflow_callback = MLflowCallback(metric_name='f2_score',
                                     create_experiment=False,
                                     mlflow_kwargs={"nested": True})
    
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_optimization',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    
    study.optimize(
        create_xgboost_objective(X_train, y_train, X_val, y_val, scale_pos_weight),
        n_trials=n_trials,
        callbacks=[mlflow_callback],
        show_progress_bar=True
    )
    
    print(f"   Best F2: {study.best_value:.4f}")
    
    if mlflow.active_run():
        mlflow.end_run()
    # Train final model with best params
    with mlflow.start_run(run_name="xgboost_best") as run:
        best_params = study.best_params.copy()
        best_params.update({
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'aucpr',
            'random_state': RANDOM_STATE,
            'n_jobs': 1
        })
        
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        optimal_threshold, _ = optimize_threshold(model, X_val, y_val, beta=2)
        
        # Test predictions
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_param('optimal_threshold', optimal_threshold)
        mlflow.log_metrics({f'test_{k}': v for k, v in test_metrics.items()})
        
        # Log artifacts
        cm_path = log_confusion_matrix(y_test, y_test_pred, "XGBoost")
        fi_paths = log_feature_importance(model, "XGBoost")
        
        mlflow.log_artifact(cm_path)
        if fi_paths:
            mlflow.log_artifact(fi_paths[0])
            mlflow.log_artifact(fi_paths[1])
        
        # Log model
        joblib.dump(model, f"{ARTIFACTS_DIR}/xgboost_model.pkl")
        joblib.dump(optimal_threshold, f"{ARTIFACTS_DIR}/xgboost_threshold.pkl")
        
        artifacts = {
            "model_path": f"{ARTIFACTS_DIR}/xgboost_model.pkl",
            "threshold_path": f"{ARTIFACTS_DIR}/xgboost_threshold.pkl",
            # We reuse the SVD/Scaler created during data loading
            "svd_path": f"{ARTIFACTS_DIR}/svd_transformer.pkl",
            "scaler_path": f"{ARTIFACTS_DIR}/scaler.pkl"
        }

        input_example = pd.DataFrame([{
            "text": "This is a sample message",
            "user_id": "user_123",
            "msg_len": 24,
            "caps_ratio": 0.0,
            "personal_pronoun_count": 0,
            "slur_count": 0,
            "user_bad_ratio_7d": 0.0,
            "user_toxicity_trend": 0.0,
            "channel_toxicity_ratio": 0.0,
            "hours_since_last_msg": 1.0,
            "is_new_to_channel": 0
        }])

        output_example = np.array([0]) 

        signature = infer_signature(input_example, output_example)
        dependencies = [os.path.abspath(__file__)]
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CyberBullyingModelWrapper(), 
            artifacts=artifacts,                 
            signature=signature,
            code_path=dependencies,
            pip_requirements=["torch","transformers==4.57.6", "scikit-learn", "xgboost", 'joblib']
        )
        run_id = run.info.run_id
        
        mlflow.set_tag("model_type", "xgboost")
        
        run_uuid = run.info.artifact_uri.split('/')[-2]
        
        print(f"   ✅ XGBoost F2: {test_metrics['f2_score']:.4f}")
        
        return run_id, run_uuid, test_metrics['f2_score']

@task(log_prints=True)
def train_logistic(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=50):
    import optuna
    from optuna.integration.mlflow import MLflowCallback

    print("Tuning Logictic Regression")

    mlflow_callback = MLflowCallback(metric_name='f2_score',
                                     create_experiment=False,
                                     mlflow_kwargs={"nested": True})
    
    study = optuna.create_study(direction='maximize',
                                study_name='logistic_optimization',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    
    study.optimize(create_logistic_objective(X_train, y_train, X_val, y_val),
                   n_trials=n_trials,
                   callbacks=[mlflow_callback],
                   show_progress_bar=True)
    
    print(f"   Best F2: {study.best_value:.4f}")

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name='logistic_reg_best') as run:
        best_params = study.best_params.copy()
        
        model = LogisticRegression(**best_params)
        model.fit(X_train, y_train)

        optimal_threshold, _ = optimize_threshold(model, X_val, y_val, beta=2)

        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        # Log params & metrics
        mlflow.log_params(best_params)
        mlflow.log_param('optimal_threshold', optimal_threshold)
        mlflow.log_metrics({f'test_{k}': v for k, v in test_metrics.items()})

        cm_path = log_confusion_matrix(y_test, y_test_pred, "Logistic_reg")
        # fi_paths = log_feature_importance(model, "Logistic_reg")
        mlflow.log_artifact(cm_path)
        # if fi_paths:
        #     mlflow.log_artifact(fi_paths[0])
        #     mlflow.log_artifact(fi_paths[1])

    
        joblib.dump(model, f"{ARTIFACTS_DIR}/logistic_reg_model.pkl")
        joblib.dump(optimal_threshold, f"{ARTIFACTS_DIR}/logistic_reg_threshold.pkl")
        
        # 2. Define the artifact map (The Wrapper expects these keys!)
        artifacts = {
            "model_path": f"{ARTIFACTS_DIR}/logistic_reg_model.pkl",
            "threshold_path": f"{ARTIFACTS_DIR}/logistic_reg_threshold.pkl",
            # We reuse the SVD/Scaler created during data loading
            "svd_path": f"{ARTIFACTS_DIR}/svd_transformer.pkl",
            "scaler_path": f"{ARTIFACTS_DIR}/scaler.pkl"
        }
        # artifacts = {
        #     "model_path": "xgboost_model.pkl",
        #     "threshold_path": "xgboost_threshold.pkl", 
        #     "svd_path": "svd_transformer.pkl",
        #     "scaler_path": "scaler.pkl"
        # }
        input_example = pd.DataFrame([{
            "text": "This is a sample message",
            "user_id": "user_123",
            "msg_len": 24,
            "caps_ratio": 0.0,
            "personal_pronoun_count": 0,
            "slur_count": 0,
            "user_bad_ratio_7d": 0.0,
            "user_toxicity_trend": 0.0,
            "channel_toxicity_ratio": 0.0,
            "hours_since_last_msg": 1.0,
            "is_new_to_channel": 0
        }])

       
        output_example = np.array([0]) 

        signature = infer_signature(input_example, output_example)

        dependencies = [os.path.abspath(__file__)]
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CyberBullyingModelWrapper(), 
            artifacts=artifacts,                 
            signature=signature,
            pip_requirements=["torch", "transformers==4.57.6", "scikit-learn", "joblib"],
            code_path=dependencies
        )

        run_id = run.info.run_id
        client = MlflowClient()
                
        mlflow.set_tag("model_type", "logistic_reg")
        
        run_uuid = run.info.artifact_uri.split('/')[-2]
        
        print(f"   Logistic_reg F2: {test_metrics['f2_score']:.4f}")
        
        return run_id, run_uuid, test_metrics['f2_score']
        

@task(log_prints=True)
def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=50):
    import optuna
    from optuna.integration.mlflow import MLflowCallback

    print("\n Tuning LightGBM with Optuna...")
    
    mlflow_callback = MLflowCallback(metric_name='f2_score',
                                     create_experiment=False,
                                     mlflow_kwargs={"nested": True})
    
    study = optuna.create_study(
        direction='maximize',
        study_name='lightgbm_optimization',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    
    study.optimize(
        create_lightgbm_objective(X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        callbacks=[mlflow_callback],
        show_progress_bar=True
    )
    
    print(f"   Best F2: {study.best_value:.4f}")
    
    # Train final model with best params
    if mlflow.active_run():
        mlflow.end_run()
        
    with mlflow.start_run(run_name="lightgbm_best") as run:
        best_params = study.best_params.copy()
        best_params.update({
            'class_weight': 'balanced',
            'random_state': RANDOM_STATE,
            'verbose': -1,
            'n_jobs': 1
        })
        
        model = LGBMClassifier(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        # Optimize threshold
        optimal_threshold, _ = optimize_threshold(model, X_val, y_val, beta=2)
        
        # Test predictions
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        # Log params & metrics
        mlflow.log_params(best_params)
        mlflow.log_param('optimal_threshold', optimal_threshold)
        mlflow.log_metrics({f'test_{k}': v for k, v in test_metrics.items()})
        
        # Log visualization artifacts
        cm_path = log_confusion_matrix(y_test, y_test_pred, "LightGBM")
        fi_paths = log_feature_importance(model, "LightGBM")
        mlflow.log_artifact(cm_path)
        if fi_paths:
            mlflow.log_artifact(fi_paths[0])
            mlflow.log_artifact(fi_paths[1])

    
        joblib.dump(model, f"{ARTIFACTS_DIR}/lightgbm_model.pkl")
        joblib.dump(optimal_threshold, f"{ARTIFACTS_DIR}/lightgbm_threshold.pkl")
        
        # 2. Define the artifact map (The Wrapper expects these keys!)
        artifacts = {
            "model_path": f"{ARTIFACTS_DIR}/lightgbm_model.pkl",
            "threshold_path": f"{ARTIFACTS_DIR}/lightgbm_threshold.pkl",
            # We reuse the SVD/Scaler created during data loading
            "svd_path": f"{ARTIFACTS_DIR}/svd_transformer.pkl",
            "scaler_path": f"{ARTIFACTS_DIR}/scaler.pkl"
        }

        # 3. Log the "Hybrid" Model using pyfunc
        input_example = pd.DataFrame([{
            "text": "This is a sample message",
            "user_id": "user_123",
            "msg_len": 24,
            "caps_ratio": 0.0,
            "personal_pronoun_count": 0,
            "slur_count": 0,
            "user_bad_ratio_7d": 0.0,
            "user_toxicity_trend": 0.0,
            "channel_toxicity_ratio": 0.0,
            "hours_since_last_msg": 1.0,
            "is_new_to_channel": 0
        }])

        output_example = np.array([0]) 

        signature = infer_signature(input_example, output_example)

        dependencies = [os.path.abspath(__file__)]
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CyberBullyingModelWrapper(), 
            artifacts=artifacts,                 
            signature=signature,
            code_path=dependencies,
            pip_requirements=["torch", "transformers==4.57.6", "scikit-learn", "lightgbm", "joblib"]
        )

        run_id = run.info.run_id
        client = MlflowClient()
                
        mlflow.set_tag("model_type", "lightgbm")
        
        run_uuid = run.info.artifact_uri.split('/')[-2]
        
        print(f"   LightGBM F2: {test_metrics['f2_score']:.4f}")
        
        return run_id, run_uuid, test_metrics['f2_score']
    
# ==========================================
# EXPERIMENT ORCHESTRATION
# ==========================================
@task(log_prints=True)
def run_all_experiments(X_train, X_val, X_test, y_train, y_val, y_test, scale_pos_weight, n_trials=50):
    """Run all model experiments and return best"""
    print("\n" + "="*60)
    print("STARTING MODEL TOURNAMENT")
    print("="*60)
    
    # Train models
    xgb_run_id, xgb_uuid, xgb_f2 = train_xgboost(
        X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight, n_trials
    )
    
    lgb_run_id, lgb_uuid, lgb_f2 = train_lightgbm(
        X_train, y_train, X_val, y_val, X_test, y_test, n_trials
    )
    
    log_run_id, log_uuid, log_f2 = train_logistic(
        X_train, y_train, X_val, y_val, X_test, y_test, n_trials
    )
    # Compare results
    results = [
        ("XGBoost", xgb_run_id, xgb_uuid, xgb_f2),
        ("LightGBM", lgb_run_id, lgb_uuid, lgb_f2),
        ("Logistic_reg", log_run_id, log_uuid, log_f2)
    ]
    
    results.sort(key=lambda x: x[3], reverse=True)
    winner_name, winner_id, winner_uuid, winner_f2 = results[0]
    
    print("\n" + "="*60)
    print("TOURNAMENT RESULTS")
    print("="*60)
    for name, run_id, uuid, f2 in results:
        print(f"   {name:12s} | F2: {f2:.4f} | Run: {run_id}")
    print("="*60)
    print(f"🏆 WINNER: {winner_name} with F2-Score: {winner_f2:.4f}")
    print("="*60)
    
    return winner_id, winner_uuid, winner_f2, winner_name

# ==========================================
# MODEL PROMOTION
# ==========================================
@task(log_prints=True)
def promote_best_model(winner_id, winner_uuid, winner_f2, winner_name, client):
    print("\nPromoting Best Model to Registry (Robust Method)...")
    
    # 1. Get the Staging Area Path (Run Artifacts)
    run = client.get_run(winner_id)
    artifact_uri = run.info.artifact_uri
    # model_url = f"{artifact_uri}/model"
    model_url=f"runs:/{winner_id}/model"
    # print(f"model url used: {model_url}")
    # print(f"model url that's supposed to be used: {f"runs:/{winner_id}/model"}")
    
    print(f"   Run ID: {winner_id}")
    print(f"   Looking for model in Staging Area: {model_url}")

    # 2. Get Current Production Metrics (Safely)
    prod_f2 = 0.0
    try:
        # Note: mlflow < 2.9.0 uses get_latest_versions
        prod_models = client.get_latest_versions(name=EXPERIMENT_NAME, stages=['Production'])
        if prod_models:
            current_prod = prod_models[0]
            if current_prod.run_id:
                prod_run = client.get_run(current_prod.run_id)
                prod_f2 = prod_run.data.metrics.get('test_f2_score', 0.0)
                print(f"   Current Production F2: {prod_f2:.4f}")
            else:
                print("   ⚠️ Production model has missing run_id. Ignoring.")
    except Exception as e:
        print(f"   ⚠️ Could not fetch production metrics: {e}")

    mv = mlflow.register_model(model_url, name=EXPERIMENT_NAME)

    # 4. Promote
    if winner_f2 > prod_f2:
        client.transition_model_version_stage(
            name=EXPERIMENT_NAME, version=mv.version, stage='Production', archive_existing_versions=True
        )
        
        # Add Tags
        client.set_model_version_tag(EXPERIMENT_NAME, mv.version, "model_type", winner_name)
        client.set_model_version_tag(EXPERIMENT_NAME, mv.version, "f2_score", str(winner_f2))
        client.set_model_version_tag(EXPERIMENT_NAME, mv.version, "run_id", winner_id)
        
        # Calculate Improvement
        if prod_f2 > 0:
            imp = ((winner_f2 - prod_f2)/prod_f2)*100
        else:
            imp = 100.0
            
        print(f"   ✅ Promoted v{mv.version} to Production (+{imp:.2f}%)")
        print(f"   Final Model: {winner_name} (F2: {winner_f2:.4f})")
    else:
        client.transition_model_version_stage(
            name=EXPERIMENT_NAME, version=mv.version, stage='Archived', archive_existing_versions=False
        )
        print(f"   ⚠️ Archived v{mv.version} (Did not beat production)")

    return mv.version