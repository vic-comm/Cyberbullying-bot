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
    fbeta_score, classification_report, precision_score, 
    recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from transformers import DistilBertTokenizer, DistilBertModel
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

# Optuna imports
import optuna
from optuna.integration.mlflow import MLflowCallback

# Prefect imports (optional - remove if not using)
from prefect import task, flow

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = "../data/training_data_with_history.parquet"
ARTIFACTS_DIR = "../api_service/artifacts"
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
        
        return predictions
    
# ==========================================
# TEXT EMBEDDINGS
# ==========================================
def get_bert_embeddings(text_list, batch_size=64):
    """
    Generate DistilBERT embeddings in batches.
    Returns numpy array of shape (N, 768).
    """
    print("   -> Loading DistilBERT Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    model.eval()
    
    all_embeddings = []
    total = len(text_list)
    
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
            
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
        
        if i % 1000 == 0 and i > 0:
            print(f"      Processed {i}/{total}...")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
            
    return np.vstack(all_embeddings)

# ==========================================
# DATA PREPARATION
# ==========================================
@task(log_prints=True)
def load_and_prepare_data():
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
    raw_text = df['text'].astype(str).tolist()
    vectors_768 = get_bert_embeddings(raw_text)
    
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
            'n_jobs': -1
        }
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
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
            'n_jobs': -1
        }
        
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        return fbeta_score(y_val, preds, beta=2)
    
    return objective

# ==========================================
# MODEL TRAINING
# ==========================================
@task(log_prints=True)
def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=50):
    print("\n Tuning XGBoost with Optuna...")
    
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_optimization',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    
    study.optimize(
        create_xgboost_objective(X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"   Best F2: {study.best_value:.4f}")
    
    # Train final model with best params
    with mlflow.start_run(run_name="xgboost_best") as run:
        best_params = study.best_params.copy()
        best_params.update({
            
            'eval_metric': 'aucpr',
            'random_state': RANDOM_STATE,
            'n_jobs': -1
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

        signature = infer_signature(X_train, model.predict_proba(X_train))
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CyberBullyingModelWrapper(), 
            artifacts=artifacts,                 
            signature=signature,
            pip_requirements=["torch", "transformers", "scikit-learn", "xgboost"]
        )
        
        mlflow.set_tag("model_type", "xgboost")
        
        run_id = run.info.run_id
        run_uuid = run.info.artifact_uri.split('/')[-2]
        
        print(f"   ✅ XGBoost F2: {test_metrics['f2_score']:.4f}")
        
        return run_id, run_uuid, test_metrics['f2_score']

@task(log_prints=True)
def train_logistic(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=50):
    print("Tuning Logictic Regression")

    study = optuna.create_study(direction='maximize',
                                study_name='logistic_optimization',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    
    study.optimize(create_logistic_objective(X_train, y_train, X_val, y_val),
                   n_trials=n_trials,
                   show_progress_bar=True)
    
    print(f"   Best F2: {study.best_value:.4f}")

    with mlflow.start_run(run_name='logistic_reg_best') as run:
        best_params = study.best_params.copy()
        
        model = LogisticRegression(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

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

        signature = infer_signature(X_train, model.predict_proba(X_train))
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CyberBullyingModelWrapper(), 
            artifacts=artifacts,                 
            signature=signature,
            pip_requirements=["torch", "transformers", "scikit-learn", "logistic_reg"]
        )
        
        mlflow.set_tag("model_type", "logistic_reg")
        
        run_id = run.info.run_id
        run_uuid = run.info.artifact_uri.split('/')[-2]
        
        print(f"   Logistic_reg F2: {test_metrics['f2_score']:.4f}")
        
        return run_id, run_uuid, test_metrics['f2_score']
        

@task(log_prints=True)
def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=50):
    print("\n Tuning LightGBM with Optuna...")
    
    study = optuna.create_study(
        direction='maximize',
        study_name='lightgbm_optimization',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    
    study.optimize(
        create_lightgbm_objective(X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"   Best F2: {study.best_value:.4f}")
    
    # Train final model with best params
    with mlflow.start_run(run_name="lightgbm_best") as run:
        best_params = study.best_params.copy()
        best_params.update({
            'class_weight': 'balanced',
            'random_state': RANDOM_STATE,
            'verbose': -1,
            'n_jobs': -1
        })
        
        model = LGBMClassifier(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
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
        signature = infer_signature(X_train, model.predict_proba(X_train))
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CyberBullyingModelWrapper(), 
            artifacts=artifacts,                 
            signature=signature,
            pip_requirements=["torch", "transformers", "scikit-learn", "lightgbm"]
        )
        
        mlflow.set_tag("model_type", "lightgbm")
        
        run_id = run.info.run_id
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
    
    # Compare results
    results = [
        ("XGBoost", xgb_run_id, xgb_uuid, xgb_f2),
        ("LightGBM", lgb_run_id, lgb_uuid, lgb_f2),
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
    """Promote best model to production if it beats current champion"""
    print("\n📦 Promoting Best Model to Registry...")
    
    
    bucket_root = os.getenv("MLFLOW_ARTIFACT_LOCATION", "s3://cyberbullying-artifacts/mlflow")
    model_url = f"{bucket_root}/models/{winner_uuid}/artifacts/model"
    
    # Check current production model
    try:
        prod_models = client.get_latest_versions(name=EXPERIMENT_NAME, stages=["Production"])
        if prod_models:
            current_prod = prod_models[0]
            prod_run = client.get_run(current_prod.run_id)
            prod_f2 = prod_run.data.metrics.get('test_f2_score', 0.0)
            print(f"   Current Production F2: {prod_f2:.4f}")
        else:
            prod_f2 = 0.0
            print("   No current production model")
    except Exception as e:
        prod_f2 = 0.0
        print(f"   No production model found: {e}")
    
    # Register new model
    print(f"   Registering model from: {model_url}")
    mv = mlflow.register_model(model_url, name=EXPERIMENT_NAME)
    
    if winner_f2 > prod_f2:
        client.transition_model_version_stage(
            name=EXPERIMENT_NAME,
            version=mv.version,
            stage='Production',
            archive_existing_versions=True
        )
        print(f"   ✅ Model v{mv.version} promoted to PRODUCTION")
        print(f"   📈 Improvement: {winner_f2 - prod_f2:.4f} (+{((winner_f2 - prod_f2)/prod_f2)*100:.2f}%)")
        
        # Save promotion metadata
        promotion_metadata = {
            'version': mv.version,
            'model_type': winner_name,
            'f2_score': float(winner_f2),
            'previous_f2': float(prod_f2),
            'improvement': float(winner_f2 - prod_f2),
            'run_id': winner_id,
            'promoted_at': pd.Timestamp.now().isoformat()
        }
        
        with open(f"{ARTIFACTS_DIR}/production_model_metadata.json", 'w') as f:
            json.dump(promotion_metadata, f, indent=2)
            
    else:
        client.transition_model_version_stage(
            name=EXPERIMENT_NAME,
            version=mv.version,
            stage='Archived',
            archive_existing_versions=False
        )
        print(f"   ⚠️  Model v{mv.version} archived (no improvement)")
        print(f"   Current production model is still better")

