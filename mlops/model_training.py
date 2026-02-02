import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    fbeta_score, classification_report, precision_score, 
    recall_score, roc_auc_score, confusion_matrix, accuracy_score,
    ConfusionMatrixDisplay, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import mlflow.pyfunc

try:
    from prefect import task
except ImportError:
    def task(log_prints=True):
        """Dummy decorator when Prefect is not available"""
        def decorator(func):
            return func
        return decorator
    

class Config:    
    SCRIPT_DIR = Path(__file__).parent.absolute()
    DATA_PATH = SCRIPT_DIR / "../data/training_data_with_history.parquet"
    CACHE_DIR = SCRIPT_DIR / "../cache"
    ARTIFACTS_DIR = SCRIPT_DIR / "../api_service/artifacts"
    
    S3_BUCKET = os.getenv('S3_BUCKET', 's3://cyberbullying-artifacts-victor-obi/mlflow')
    
    # Model parameters
    SVD_COMPONENTS = 128
    RANDOM_STATE = 42
    EXPERIMENT_NAME = "cyberbullying-detection"
    
    TABULAR_FEATURES = [
        'msg_len', 'caps_ratio', 'personal_pronoun_count', 'slur_count',
        'user_bad_ratio_7d', 'user_toxicity_trend',
        'channel_toxicity_ratio', 'hours_since_last_msg', 'is_new_to_channel'
    ]
    
    TEST_SIZE = 0.3
    VAL_SIZE = 0.5  # Of the test set
    
    BETA_SCORE = 2  

    @classmethod
    def setup_directories(cls):
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics"""
    f2_score: float
    f1_score: float
    precision: float
    recall: float
    roc_auc: float
    accuracy: float
    avg_precision: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'f2_score': self.f2_score,
            'f1_score': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'accuracy': self.accuracy,
            'roc_auc': self.roc_auc,
            'avg_precision': self.avg_precision
        }
    
    def __str__(self) -> str:
        return (f"F2: {self.f2_score:.4f} | F1: {self.f1_score:.4f} | "
                f"Precision: {self.precision:.4f} | Recall: {self.recall:.4f} | "
                f"ROC-AUC: {self.roc_auc:.4f} | Accuracy: {self.accuracy:.4f}")
    

@dataclass
class TrainingData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scale_pos_weight: float
    variance_retained: float
    
    def get_shapes_summary(self) -> Dict[str, Any]:
        return {
            'train_size': len(self.X_train),
            'val_size': len(self.X_val),
            'test_size': len(self.X_test),
            'num_features': self.X_train.shape[1],
            'scale_pos_weight': self.scale_pos_weight,
            'variance_retained': self.variance_retained
        }


class ModelType(Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LOGISTIC = "logistic_reg"


class BERTEmbeddingGenerator:    
    def __init__(self, batch_size: int = 32, max_length: int = 128):
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = self._get_device()
        
    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def generate(self, text_list: List[str]) -> np.ndarray:
        from transformers import DistilBertTokenizer, DistilBertModel
        
        print(f"   -> Loading DistilBERT on {self.device}...")
        
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        model.eval()
        
        total = len(text_list)
        embeddings = np.zeros((total, 768), dtype=np.float32)
        
        print(f"   -> Generating embeddings for {total:,} texts...")
        
        try:
            for i in range(0, total, self.batch_size):
                batch_text = text_list[i:i + self.batch_size]
                
                inputs = tokenizer(batch_text, padding=True, truncation=True, max_length=self.max_length,
                                   return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings[i:i + len(batch_embeddings)] = batch_embeddings
                
                # Memory cleanup
                del inputs, outputs, batch_embeddings
                
                if i % 1000 == 0 and i > 0:
                    print(f"      Progress: {i:,}/{total:,} ({i/total*100:.1f}%)")
                    self._cleanup_memory()
        
        finally:
            # Ensure cleanup even on error
            del model, tokenizer
            self._cleanup_memory()
        
        return embeddings
    
    def _cleanup_memory(self):
        import gc
        gc.collect()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()


class DataPreparator:
    """Handles data loading, embedding generation, and feature engineering"""
    def __init__(self, config: Config):
        self.config = config
        self.embedding_generator = BERTEmbeddingGenerator()
        self.variance_retained = 0
        
    @task(log_prints=True)
    def prepare_data(self, force_recompute: bool = False) -> TrainingData:
        """
        Load and prepare all features for training
        
        Args:
            force_recompute: If True, regenerate embeddings even if cached
            
        Returns:
            TrainingData object with all splits
        """
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        df = self._load_raw_data()
        
        embeddings_768 = self._get_embeddings(df, force_recompute)
        
        embeddings_reduced, svd_transformer = self._reduce_dimensions(embeddings_768)
        
        tabular_scaled, scaler = self._process_tabular_features(df)
        
        X, y = self._combine_features(embeddings_reduced, tabular_scaled, df['label'].values)
        
        training_data = self._create_splits(X, y)
        
        self._save_preprocessing_artifacts(svd_transformer, scaler)
        
        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETE")
        print("="*60)
        self._print_summary(training_data)
        
        return training_data
    
    def _load_raw_data(self) -> pd.DataFrame:
        print("\n1. Loading Data...")
        
        if not self.config.DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found: {self.config.DATA_PATH}")
        
        df = pd.read_parquet(self.config.DATA_PATH)
        print(f"   Shape: {df.shape}")
        
        # Validate required columns
        required_cols = ['text', 'label'] + self.config.TABULAR_FEATURES
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Class distribution
        class_dist = df['label'].value_counts(normalize=True)
        print(f"   Class Distribution:")
        print(f"      Safe (0):  {class_dist[0]*100:.2f}%")
        print(f"      Toxic (1): {class_dist[1]*100:.2f}%")
        
        scale_pos_weight = (df['label'] == 0).sum() / (df['label'] == 1).sum()
        print(f"   Scale_pos_weight: {scale_pos_weight:.2f}")
        
        return df
    
    def _get_embeddings(self, df: pd.DataFrame, force_recompute: bool) -> np.ndarray:
        """Generate or load cached embeddings"""
        from cache import EmbeddingCache
        
        print("\n2. Text Embeddings (DistilBERT)...")
        
        # Determine if we should use S3
        local_cache_exists = (
            (self.config.CACHE_DIR / 'bert_embeddings.pkl').exists() and
            (self.config.CACHE_DIR / 'cache_metadata.json').exists()
        )
        
        use_s3 = not local_cache_exists
        s3_bucket = self.config.S3_BUCKET if use_s3 else None
        
        if local_cache_exists:
            print("   📁 Local cache detected")
        else:
            print("   🪣 No local cache - will sync with S3")
        
        cache = EmbeddingCache(
            cache_dir=str(self.config.CACHE_DIR),
            s3_bucket=s3_bucket,
            use_s3=use_s3,
            s3_prefix="embeddings-cache"
        )
        
        if not force_recompute:
            try:
                print("   -> Loading from cache...")
                embeddings, metadata = cache.load_embeddings()
                print(f"   ✅ Loaded {len(embeddings):,} cached embeddings")
                return embeddings
            except Exception as e:
                print(f"   ⚠️  Cache load failed: {e}")
                print("   -> Falling back to generation...")
        
        # Generate new embeddings
        print("   -> Generating new embeddings...")
        raw_text = df['text'].astype(str).tolist()
        embeddings = self.embedding_generator.generate(raw_text)
        
        # Save to cache
        try:
            cache.save_embeddings(
                embeddings,
                str(self.config.DATA_PATH),
                additional_info={'num_texts': len(raw_text)}
            )
            print("   ✅ Embeddings cached successfully")
        except Exception as e:
            print(f"   ⚠️  Failed to cache embeddings: {e}")
        
        return embeddings
    
    def _reduce_dimensions(self, embeddings: np.ndarray) -> Tuple[np.ndarray, TruncatedSVD]:
        """Apply dimensionality reduction"""
        print(f"\n3. Dimensionality Reduction (768 -> {self.config.SVD_COMPONENTS})...")
        
        svd = TruncatedSVD(
            n_components=self.config.SVD_COMPONENTS,
            random_state=self.config.RANDOM_STATE
        )
        embeddings_reduced = svd.fit_transform(embeddings)
        
        variance_retained = svd.explained_variance_ratio_.sum()
        print(f"   ✅ Variance Retained: {variance_retained*100:.2f}%")
        self.variance_retained = variance_retained
        return embeddings_reduced, svd
    
    def _process_tabular_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
        """Scale tabular features"""
        print("\n4. Processing Tabular Features...")
        
        tabular_data = df[self.config.TABULAR_FEATURES].values
        
        scaler = StandardScaler()
        tabular_scaled = scaler.fit_transform(tabular_data)
        
        print(f"   ✅ Scaled {len(self.config.TABULAR_FEATURES)} features")
        
        return tabular_scaled, scaler
    
    def _combine_features(self, embeddings: np.ndarray, tabular: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print("\n5. Feature Fusion...")
        
        X = np.hstack([embeddings, tabular])
        
        print(f"   Final Shape: {X.shape}")
        print(f"   - Text features: {embeddings.shape[1]}")
        print(f"   - Behavioral features: {tabular.shape[1]}")
        
        return X, labels
    
    def _create_splits(self, X: np.ndarray, y: np.ndarray) -> TrainingData:
        """Create train/val/test splits"""
        print("\n6. Creating Data Splits...")
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        # Second split: val vs test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.config.VAL_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y_temp
        )
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        print(f"   Train: {len(X_train):,} samples")
        print(f"   Val:   {len(X_val):,} samples")
        print(f"   Test:  {len(X_test):,} samples")
        
        # Calculate variance retained (placeholder - should come from SVD)
        # variance_retained = 0.95  # This should be passed from _reduce_dimensions
        
        return TrainingData(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            scale_pos_weight=scale_pos_weight,
            variance_retained=self.variance_retained
        )
    
    def _save_preprocessing_artifacts(self, svd: TruncatedSVD, scaler: StandardScaler):
        """Save preprocessing transformers"""
        print("\n7. Saving Preprocessing Artifacts...")
        
        joblib.dump(svd, self.config.ARTIFACTS_DIR / "svd_transformer.pkl")
        joblib.dump(scaler, self.config.ARTIFACTS_DIR / "scaler.pkl")
        
        print("   ✅ Saved SVD and Scaler")
    
    def _print_summary(self, data: TrainingData):
        """Print data preparation summary"""
        summary = data.get_shapes_summary()
        
        print(f"\n   Training set:   {summary['train_size']:,} samples")
        print(f"   Validation set: {summary['val_size']:,} samples")
        print(f"   Test set:       {summary['test_size']:,} samples")
        print(f"   Features:       {summary['num_features']}")
        print(f"   Pos weight:     {summary['scale_pos_weight']:.2f}")


class CyberBullyingModelWrapper(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        """Load model artifacts"""
        import joblib
        from transformers import DistilBertTokenizer, DistilBertModel
        import torch
        
        # Load sklearn artifacts
        self.model = joblib.load(context.artifacts["model_path"])
        self.svd = joblib.load(context.artifacts["svd_path"])
        self.scaler = joblib.load(context.artifacts["scaler_path"])
        self.threshold = joblib.load(context.artifacts["threshold_path"])
        
        # Load DistilBERT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.bert_model.eval()
    
    def _get_bert_embeddings(self, text_list: List[str]) -> np.ndarray:
        """Generate BERT embeddings"""
        import torch
        
        inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    def predict(self, context, model_input: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        import numpy as np
        
        # Extract text and generate embeddings
        text_data = model_input['text'].astype(str).tolist()
        vectors_768 = self._get_bert_embeddings(text_data)
        vectors_128 = self.svd.transform(vectors_768)
        
        # Process tabular features
        tabular_data = model_input.drop(columns=['text']).values
        tabular_scaled = self.scaler.transform(tabular_data)
        
        # Combine features
        X_final = np.hstack([vectors_128, tabular_scaled])
        
        # Predict
        probs = self.model.predict_proba(X_final)[:, 1]
        predictions = (probs >= self.threshold).astype(int)
        
        return predictions, probs
    

class MetricsCalculator:
    """Calculate and log model metrics"""
    
    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> ModelMetrics:
        """Calculate all evaluation metrics"""
        return ModelMetrics(
            f2_score=fbeta_score(y_true, y_pred, beta=2),
            f1_score=fbeta_score(y_true, y_pred, beta=1),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_true, y_proba),
            accuracy=accuracy_score(y_true, y_pred),
            avg_precision=average_precision_score(y_true, y_proba)
        )
    
    @staticmethod
    def optimize_threshold(model, X_val: np.ndarray, y_val: np.ndarray, beta: int = 2) -> Tuple[float, float]:
        """Find optimal classification threshold"""
        y_proba = model.predict_proba(X_val)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        f_scores = [
            fbeta_score(y_val, (y_proba >= t).astype(int), beta=beta)
            for t in thresholds
        ]
        
        optimal_idx = np.argmax(f_scores)
        return thresholds[optimal_idx], f_scores[optimal_idx]
    
    @staticmethod
    def log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_dir: Path) -> Path:
        """Generate and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Safe', 'Toxic']
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap='Blues', ax=ax)
        plt.title(f'{model_name} Confusion Matrix')
        
        cm_path = save_dir / f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return cm_path
    
    @staticmethod
    def log_feature_importance(model,model_name: str, feature_names: List[str], save_dir: Path, top_n: int = 20) -> Optional[Tuple[Path, Path]]:
        """Generate and save feature importance plots"""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        elif hasattr(model, "coef_"):
            coef = model.coef_

            # Binary classification → shape (1, n_features)
            # Multiclass → shape (n_classes, n_features)
            if coef.ndim == 2:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)

        else:
            # Model does not support feature importance
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.tight_layout()
        
        fi_plot_path = save_dir / f"feature_importance_{model_name}.png"
        plt.savefig(fi_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save CSV
        fi_csv_path = save_dir / f"feature_importance_{model_name}.csv"
        importance_df.to_csv(fi_csv_path, index=False)
        
        return fi_plot_path, fi_csv_path


class HyperparameterOptimizer:
    """Handles Optuna-based hyperparameter optimization"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_xgboost_objective(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, scale_pos_weight: float):
        def objective(trial):
            import optuna
            
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
                'random_state': self.config.RANDOM_STATE,
                'n_jobs': 1
            }
            
            model = XGBClassifier(**params)
            model.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            preds = model.predict(X_val)
            f2 = fbeta_score(y_val, preds, beta=2)
            
            # Log additional metrics
            mlflow.log_metric('f1_score', fbeta_score(y_val, preds, beta=1))
            mlflow.log_metric('recall', recall_score(y_val, preds))
            mlflow.log_metric('precision', precision_score(y_val, preds))
            mlflow.log_metric('accuracy', accuracy_score(y_val, preds))
            
            return f2
        
        return objective
    
    def create_lightgbm_objective(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
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
                'random_state': self.config.RANDOM_STATE,
                'verbose': -1,
                'n_jobs': 1
            }
            
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            
            preds = model.predict(X_val)
            f2 = fbeta_score(y_val, preds, beta=2)
            
            # Log additional metrics
            mlflow.log_metric('f1_score', fbeta_score(y_val, preds, beta=1))
            mlflow.log_metric('recall', recall_score(y_val, preds))
            mlflow.log_metric('precision', precision_score(y_val, preds))
            mlflow.log_metric('accuracy', accuracy_score(y_val, preds))
            
            return f2
        
        return objective
    
    def create_logistic_objective(self, X_train: np.ndarray, y_train: np.ndarray,X_val: np.ndarray, y_val: np.ndarray):
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.001, 10.0, log=True),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga']),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'max_iter': 1000,
                'random_state': self.config.RANDOM_STATE
            }
            
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            f2 = fbeta_score(y_val, preds, beta=2)
            
            mlflow.log_metric('f1_score', fbeta_score(y_val, preds, beta=1))
            mlflow.log_metric('recall', recall_score(y_val, preds))
            mlflow.log_metric('precision', precision_score(y_val, preds))
            mlflow.log_metric('accuracy', accuracy_score(y_val, preds))
            
            return f2
        
        return objective

class ModelTrainer:
    """Handles model training with MLflow tracking"""
    
    def __init__(self, config: Config):
        self.config = config
        self.optimizer = HyperparameterOptimizer(config)
        self.metrics_calc = MetricsCalculator()
        
    @task(log_prints=True)
    def train_model(self, model_type: ModelType, data: TrainingData, n_trials: int = 50) -> Tuple[str, str, float]:
        """
        Train a model with hyperparameter optimization
        
        Args:
            model_type: Type of model to train
            data: Training data splits
            n_trials: Number of Optuna trials
            
        Returns:
            Tuple of (run_id, run_uuid, f2_score)
        """
        import optuna
        from optuna.integration.mlflow import MLflowCallback
        
        print(f"\n{'='*60}")
        print(f"TRAINING {model_type.value.upper()}")
        print("="*60)
        
        # Create Optuna study
        study = optuna.create_study(direction='maximize', study_name=f'{model_type.value}_optimization', sampler=optuna.samplers.TPESampler(seed=self.config.RANDOM_STATE))
        
        # Get appropriate objective function
        if model_type == ModelType.XGBOOST:
            objective = self.optimizer.create_xgboost_objective(
                data.X_train, data.y_train, data.X_val, data.y_val,
                data.scale_pos_weight)
        elif model_type == ModelType.LIGHTGBM:
            objective = self.optimizer.create_lightgbm_objective(
                data.X_train, data.y_train, data.X_val, data.y_val)
        else:  # LOGISTIC
            objective = self.optimizer.create_logistic_objective(
                data.X_train, data.y_train, data.X_val, data.y_val)
        
        # Optimize
        mlflow_callback = MLflowCallback(metric_name='f2_score', create_experiment=False,mlflow_kwargs={"nested": True})
        
        study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_callback], show_progress_bar=True)
        
        print(f"\n   Best F2 from optimization: {study.best_value:.4f}")
        
        # Close any active runs
        if mlflow.active_run():
            mlflow.end_run()
        
        # Train final model with best params
        run_id, run_uuid, test_f2 = self._train_final_model(
            model_type, study.best_params, data
        )
        
        return run_id, run_uuid, test_f2
    
    def _train_final_model(self, model_type: ModelType, best_params: Dict[str, Any], data: TrainingData) -> Tuple[str, str, float]:        
        with mlflow.start_run(run_name=f"{model_type.value}_best") as run:
            # Create model with best params
            model = self._create_model(model_type, best_params, data.scale_pos_weight)
            
            # Train
            if model_type == ModelType.XGBOOST:
                model.fit(
                    data.X_train, data.y_train,
                    eval_set=[(data.X_val, data.y_val)],
                    verbose=False
                )
            elif model_type == ModelType.LIGHTGBM:
                model.fit(
                    data.X_train, data.y_train,
                    eval_set=[(data.X_val, data.y_val)]
                )
            else:
                model.fit(data.X_train, data.y_train)
            
            optimal_threshold, _ = self.metrics_calc.optimize_threshold(
                model, data.X_val, data.y_val, beta=self.config.BETA_SCORE
            )
            
            y_test_proba = model.predict_proba(data.X_test)[:, 1]
            y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
            
            test_metrics = self.metrics_calc.calculate(
                data.y_test, y_test_pred, y_test_proba
            )
            
            self._log_to_mlflow(
                model_type, model, best_params, optimal_threshold,
                test_metrics, data.y_test, y_test_pred
            )
            
            self._save_and_log_model(model_type, model, optimal_threshold)
            
            run_id = run.info.run_id
            run_uuid = run.info.artifact_uri.split('/')[-2]
            
            print(f"\n   ✅ Test Metrics: {test_metrics}")
            
            return run_id, run_uuid, test_metrics.f2_score
    
    def _create_model(self, model_type: ModelType, params: Dict[str, Any], scale_pos_weight: float):
        params = params.copy()
        
        if model_type == ModelType.XGBOOST:
            params.update({
                'scale_pos_weight': scale_pos_weight,
                'eval_metric': 'aucpr',
                'random_state': self.config.RANDOM_STATE,
                'n_jobs': 1
            })
            return XGBClassifier(**params)
        
        elif model_type == ModelType.LIGHTGBM:
            params.update({
                'class_weight': 'balanced',
                'random_state': self.config.RANDOM_STATE,
                'verbose': -1,
                'n_jobs': 1
            })
            return LGBMClassifier(**params)
        
        else:  # LOGISTIC
            params['random_state'] = self.config.RANDOM_STATE
            return LogisticRegression(**params)
    
    def _log_to_mlflow(self, model_type: ModelType, model, params: Dict[str, Any], threshold: float, metrics: ModelMetrics, y_true: np.ndarray, y_pred: np.ndarray):
        mlflow.log_params(params)
        mlflow.log_param('optimal_threshold', threshold)
        mlflow.log_param('model_type', model_type.value)
        
        mlflow.log_metrics({f'test_{k}': v for k, v in metrics.to_dict().items()})
        
        cm_path = self.metrics_calc.log_confusion_matrix(
            y_true, y_pred, model_type.value, self.config.ARTIFACTS_DIR
        )
        mlflow.log_artifact(str(cm_path))
        
        feature_names = (
            [f'svd_{i}' for i in range(self.config.SVD_COMPONENTS)] +
            self.config.TABULAR_FEATURES
        )
        
        fi_paths = self.metrics_calc.log_feature_importance(
            model, model_type.value, feature_names, self.config.ARTIFACTS_DIR
        )
        
        if fi_paths:
            mlflow.log_artifact(str(fi_paths[0]))
            mlflow.log_artifact(str(fi_paths[1]))
        
        # Set tags
        mlflow.set_tag("model_type", model_type.value)
        mlflow.set_tag("f2_score", f"{metrics.f2_score:.4f}")
    
    def _save_and_log_model(self, model_type: ModelType, model, threshold: float):
        # Save artifacts locally
        model_path = self.config.ARTIFACTS_DIR / f"{model_type.value}_model.pkl"
        threshold_path = self.config.ARTIFACTS_DIR / f"{model_type.value}_threshold.pkl"
        
        joblib.dump(model, model_path)
        joblib.dump(threshold, threshold_path)
        
        # Create artifact map for MLflow
        artifacts = {
            "model_path": str(model_path),
            "threshold_path": str(threshold_path),
            "svd_path": str(self.config.ARTIFACTS_DIR / "svd_transformer.pkl"),
            "scaler_path": str(self.config.ARTIFACTS_DIR / "scaler.pkl")
        }
        
        # Create input example
        input_example = pd.DataFrame([{
            "text": "This is a sample message for testing",
            "msg_len": 24,
            "caps_ratio": 0.0,
            "personal_pronoun_count": 1,
            "slur_count": 0,
            "user_bad_ratio_7d": 0.0,
            "user_toxicity_trend": 0.0,
            "channel_toxicity_ratio": 0.05,
            "hours_since_last_msg": 1.0,
            "is_new_to_channel": 0
        }])
        
        output_example = np.array([0])
        signature = infer_signature(input_example, output_example)
        
        # Determine pip requirements based on model type
        base_requirements = ["torch", "transformers==4.57.6", "scikit-learn", "joblib"]
        
        if model_type == ModelType.XGBOOST:
            base_requirements.append("xgboost")
        elif model_type == ModelType.LIGHTGBM:
            base_requirements.append("lightgbm")
        
        # Log model as PyFunc
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CyberBullyingModelWrapper(),
            artifacts=artifacts,
            signature=signature,
            code_path=[str(Path(__file__).absolute())],
            pip_requirements=base_requirements
        )

@task(log_prints=True)
def run_all_experiments(data: TrainingData, n_trials: int = 50, models_to_train: Optional[List[ModelType]] = None) -> Tuple[str, str, float, str]:
    if models_to_train is None:
        models_to_train = list(ModelType)
    
    print("\n" + "="*60)
    print("STARTING MODEL TOURNAMENT")
    print("="*60)
    
    config = Config()
    trainer = ModelTrainer(config)
    
    results = []
    
    for model_type in models_to_train:
        try:
            run_id, run_uuid, f2_score = trainer.train_model(
                model_type, data, n_trials
            )
            results.append((model_type.value, run_id, run_uuid, f2_score))
        except Exception as e:
            print(f"\n Error training {model_type.value}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        raise RuntimeError("All model training attempts failed!")
    
    # Sort by F2 score
    results.sort(key=lambda x: x[3], reverse=True)
    winner_name, winner_id, winner_uuid, winner_f2 = results[0]
    
    # Print results
    print("\n" + "="*60)
    print("TOURNAMENT RESULTS")
    print("="*60)
    for name, run_id, uuid, f2 in results:
        emoji = "🏆" if name == winner_name else "  "
        print(f"{emoji} {name:15s} | F2: {f2:.4f} | Run: {run_id}")
    print("="*60)
    print(f"🏆 WINNER: {winner_name} with F2-Score: {winner_f2:.4f}")
    print("="*60)
    
    return winner_id, winner_uuid, winner_f2, winner_name


# MODEL PROMOTION
@task(log_prints=True)
def promote_best_model(winner_id: str, winner_uuid: str, winner_f2: float, winner_name: str, client: MlflowClient, experiment_name: str = "cyberbullying-detection") -> int:
    """
    Promote best model to production in registry
    
    Args:
        winner_id: MLflow run ID of winning model
        winner_uuid: UUID of winning model
        winner_f2: F2 score of winning model
        winner_name: Name of winning model
        client: MLflow client
        experiment_name: Name of experiment
        
    Returns:
        Version number of registered model
    """
    print("\n" + "="*60)
    print("MODEL PROMOTION")
    print("="*60)
    
    model_uri = f"runs:/{winner_id}/model"
    print(f"\n   Model URI: {model_uri}")
    
    prod_f2 = 0.0
    try:
        prod_models = client.get_latest_versions(
            name=experiment_name,
            stages=['Production']
        )
        
        if prod_models:
            current_prod = prod_models[0]
            if current_prod.run_id:
                prod_run = client.get_run(current_prod.run_id)
                prod_f2 = prod_run.data.metrics.get('test_f2_score', 0.0)
                print(f"   Current Production F2: {prod_f2:.4f}")
            else:
                print("    Production model has no run_id")
        else:
            print("   No current production model")
    except Exception as e:
        print(f"     Could not fetch production metrics: {e}")
    
    # Register model
    try:
        mv = mlflow.register_model(model_uri, name=experiment_name)
        print(f"    Registered as version {mv.version}")
    except Exception as e:
        print(f"    Registration failed: {e}")
        raise
    
    if winner_f2 > prod_f2:
        try:
            client.transition_model_version_stage(
                name=experiment_name,
                version=mv.version,
                stage='Production',
                archive_existing_versions=True
            )
            
            client.set_model_version_tag(experiment_name, mv.version, "model_type", winner_name)
            client.set_model_version_tag(experiment_name, mv.version, "f2_score", str(winner_f2))
            client.set_model_version_tag(experiment_name, mv.version, "run_id", winner_id)
            
            if prod_f2 > 0:
                improvement = ((winner_f2 - prod_f2) / prod_f2) * 100
            else:
                improvement = 100.0
            
            print(f"\n    Promoted v{mv.version} to Production")
            print(f"   Improvement: +{improvement:.2f}%")
            print(f"   Model: {winner_name}")
            print(f"   F2 Score: {winner_f2:.4f}")
            
        except Exception as e:
            print(f"    Promotion failed: {e}")
            raise
    else:
        try:
            client.transition_model_version_stage(
                name=experiment_name,
                version=mv.version,
                stage='Archived',
                archive_existing_versions=False
            )
            print(f"\n     Archived v{mv.version}")
            print(f"   Reason: Did not beat production ({winner_f2:.4f} vs {prod_f2:.4f})")
        except Exception as e:
            print(f"     Archival failed: {e}")
    
    print("="*60)
    
    return mv.version


def load_and_prepare_data(force_recompute: bool = False) -> TrainingData:
    """Convenience wrapper for data preparation"""
    config = Config()
    config.setup_directories()
    
    preparator = DataPreparator(config)
    return preparator.prepare_data(force_recompute)
