import pandas as pd
import numpy as np
import os
import json
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
from .utils import calculate_text_features
from prefect import task, flow
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import IntervalSchedule

# CONFIGURATION
LOGS_PATH = os.getenv("LOGS_PATH", "../data/raw_logs.jsonl")
MASTER_DATA_PATH = os.getenv("MASTER_DATA_PATH", "../data/training_data_with_history.parquet")
BACKUP_PATH = os.getenv("BACKUP_PATH", "../data/training_data_backup.parquet")
ARCHIVE_PATH = os.getenv("ARCHIVE_PATH", "../data/archives")
FEATURE_CONFIG_PATH = os.getenv("FEATURE_CONFIG", "../config/features.json")

# Data quality thresholds
MIN_TEXT_LENGTH = 3
MAX_TEXT_LENGTH = 5000
MIN_NEW_SAMPLES = 10  
MAX_NULL_RATIO = 0.3  

# Feature engineering
TOXIC_KEYWORDS = {
    'slurs': ['trash', 'scum', 'garbage', 'loser', 'idiot', 'stupid', 'dumb'],
    'threats': ['kill', 'die', 'hurt', 'attack', 'destroy'],
    'harassment': ['ugly', 'fat', 'worthless', 'pathetic', 'waste']
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# DATA LOADING AND VALIDATION
@task(name="Load and Validate Logs", log_prints=True, retries=2, retry_delay_seconds=30)
def load_and_validate_logs() -> Optional[pd.DataFrame]:
    if not os.path.exists(LOGS_PATH):
        logger.warning(f"⚠️  No logs found at {LOGS_PATH}")
        return None
    
    # Check file size
    file_size = os.path.getsize(LOGS_PATH)
    if file_size == 0:
        logger.warning("⚠️  Log file is empty")
        return None
    
    logger.info(f"📂 Loading logs from {LOGS_PATH} ({file_size} bytes)")
    
    try:
        # Read JSONL with error handling
        new_data = pd.read_json(LOGS_PATH, lines=True)
        
        if new_data.empty:
            logger.warning("⚠️  No data in log file")
            return None
        
        logger.info(f"✅ Loaded {len(new_data)} raw records")
        
        # Validate required columns
        required_cols = ['text', 'user_id', 'timestamp']
        missing_cols = [col for col in required_cols if col not in new_data.columns]
        
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Filter by text length
        new_data = new_data[
            (new_data['text'].str.len() >= MIN_TEXT_LENGTH) &
            (new_data['text'].str.len() <= MAX_TEXT_LENGTH)
        ].copy()
        
        logger.info(f"✅ {len(new_data)} records after length filtering")
        
        # Determine label source
        if 'verified_label' in new_data.columns:
            # Use human labels (highest quality)
            valid_data = new_data[new_data['verified_label'].notna()].copy()
            valid_data['label'] = valid_data['verified_label'].astype(int)
            valid_data['label_source'] = 'human'
            logger.info(f"✅ Using {len(valid_data)} human-verified labels")
            
        elif 'prediction' in new_data.columns:
            # Use model predictions (pseudo-labeling)
            valid_data = new_data[new_data['prediction'].notna()].copy()
            valid_data['label'] = valid_data['prediction'].astype(int)
            valid_data['label_source'] = 'model'
            logger.warning(f"⚠️  Using {len(valid_data)} model predictions as pseudo-labels")
            
        else:
            logger.error("❌ No label column found (need 'verified_label' or 'prediction')")
            return None
        
        if len(valid_data) < MIN_NEW_SAMPLES:
            logger.warning(f"⚠️  Only {len(valid_data)} samples (minimum: {MIN_NEW_SAMPLES})")
            return None
        
        # Add metadata
        valid_data['ingested_at'] = datetime.now().isoformat()
        
        # Check for nulls
        null_ratio = valid_data.isnull().mean().mean()
        if null_ratio > MAX_NULL_RATIO:
            logger.warning(f"⚠️  High null ratio: {null_ratio:.2%} (threshold: {MAX_NULL_RATIO:.2%})")
        
        # Deduplicate by text hash
        if 'text_hash' in valid_data.columns:
            before = len(valid_data)
            valid_data = valid_data.drop_duplicates(subset=['text_hash'])
            after = len(valid_data)
            if before != after:
                logger.info(f"🔄 Removed {before - after} duplicate samples")
        
        logger.info(f"✅ Validated {len(valid_data)} samples ready for ingestion")
        
        return valid_data
        
    except Exception as e:
        logger.error(f"❌ Failed to load logs: {e}", exc_info=True)
        return None

# FEATURE ENGINEERING
@task(name="Calculate Features", log_prints=True)
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"🔧 Engineering features for {len(df)} samples...")
    
    df = df.copy()
    
    logger.info("   -> Applying centralized text feature logic...")
    
    feature_dicts = df['text'].apply(calculate_text_features).tolist()
    
    feature_df = pd.DataFrame(feature_dicts)
    
   
    df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    
   
    history_features = [
        'user_bad_ratio_7d',
        'user_bad_ratio_30d',
        'user_toxicity_trend',
        'user_msg_count_7d',
        'channel_toxicity_ratio',
        'hours_since_last_msg',
        'is_new_to_channel',
        'user_report_count'
    ]
    
    for feature in history_features:
        if feature not in df.columns:
            if 'ratio' in feature or 'trend' in feature:
                df[feature] = 0.0
            elif 'count' in feature:
                df[feature] = 0
            elif 'hours' in feature:
                df[feature] = 1.0
            elif 'is_' in feature:
                df[feature] = 0
    
    if 'timestamp' in df.columns:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        
        # User message frequency
        user_msg_counts = df.groupby('user_id').size()
        df['user_msg_count_total'] = df['user_id'].map(user_msg_counts).fillna(0)
        
        # User toxicity rate (if labels exist in this batch)
        if 'label' in df.columns:
            user_toxic_rates = df.groupby('user_id')['label'].mean()
            df['user_toxicity_rate'] = df['user_id'].map(user_toxic_rates).fillna(0)
    
    # ========== TEMPORAL FEATURES ==========
    if 'timestamp' in df.columns:
        df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    logger.info(f"✅ Feature engineering complete: {len(df.columns)} features")
    
    return df


# DATA QUALITY CHECKS
@task(name="Validate Data Quality",log_prints=True)
def validate_data_quality(df: pd.DataFrame) -> bool:
    logger.info("🔍 Running data quality checks...")
    
    issues = []
    
    # Check label distribution
    label_dist = df['label'].value_counts(normalize=True)
    logger.info(f"Label distribution:\n{label_dist}")
    
    minority_class_ratio = label_dist.min()
    if minority_class_ratio < 0.1:
        issues.append(f"Severe class imbalance: minority class at {minority_class_ratio:.2%}")
    
    null_counts = df.isnull().sum()
    high_null_features = null_counts[null_counts > len(df) * 0.2]
    
    if not high_null_features.empty:
        issues.append(f"Features with >20% nulls: {list(high_null_features.index)}")
    
    # Check for constant features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_features = [col for col in numeric_cols if df[col].nunique() == 1]
    
    if constant_features:
        issues.append(f"Constant features (no variance): {constant_features}")
    
    # Check for outliers in key features
    key_features = ['msg_len', 'caps_ratio', 'slur_count']
    for feature in key_features:
        if feature in df.columns:
            q99 = df[feature].quantile(0.99)
            outliers = (df[feature] > q99).sum()
            if outliers > len(df) * 0.05:
                logger.warning(f"⚠️  {feature}: {outliers} outliers (>{q99:.2f})")
    
    # Report issues
    if issues:
        logger.warning(f"⚠️  Data quality issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        
        # Decide whether to proceed
        # For now, we proceed with warnings but this could be a hard stop
        return True
    else:
        logger.info(" All data quality checks passed")
        return True

# DATA MERGING
@task(name="Merge and Save", log_prints=True, retries=2)
def merge_and_save(new_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if new_df is None or new_df.empty:
        logger.warning("  No new data to merge")
        return {"status": "skipped", "reason": "no_new_data"}
    
    stats = {
        "new_samples": len(new_df),
        "merge_timestamp": datetime.now().isoformat()
    }
    
    try:
        # Load master dataset
        if os.path.exists(MASTER_DATA_PATH):
            logger.info(f" Loading master dataset from {MASTER_DATA_PATH}")
            master_df = pd.read_parquet(MASTER_DATA_PATH)
            stats["master_size_before"] = len(master_df)
            logger.info(f" Loaded {len(master_df)} existing samples")
        else:
            logger.warning(f"  No master dataset found, creating new one")
            master_df = pd.DataFrame()
            stats["master_size_before"] = 0
        
        # Align columns
        if not master_df.empty:
            # Get common columns
            common_cols = list(set(master_df.columns) & set(new_df.columns))
            
            # Add missing columns to new_df with defaults
            for col in master_df.columns:
                if col not in new_df.columns:
                    if master_df[col].dtype == 'object':
                        new_df[col] = None
                    else:
                        new_df[col] = 0
            
            # Align column order
            new_df = new_df[master_df.columns]
        
        # Merge
        combined_df = pd.concat([master_df, new_df], ignore_index=True)
        
        # Deduplicate
        if 'text_hash' in combined_df.columns:
            dedup_col = 'text_hash'
        else:
            dedup_col = ['text', 'user_id', 'timestamp']
        
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=dedup_col, keep='last')
        after_dedup = len(combined_df)
        
        stats["duplicates_removed"] = before_dedup - after_dedup
        stats["master_size_after"] = after_dedup
        
        logger.info(f"🔄 Removed {stats['duplicates_removed']} duplicates")
        
        # Create backup
        if not master_df.empty:
            os.makedirs(os.path.dirname(BACKUP_PATH), exist_ok=True)
            
            # Timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_backup = BACKUP_PATH.replace('.parquet', f'_{timestamp}.parquet')
            
            master_df.to_parquet(timestamped_backup)
            logger.info(f"💾 Created backup: {timestamped_backup}")
            
            # Also keep latest backup
            master_df.to_parquet(BACKUP_PATH)
        
        # Save merged dataset
        combined_df.to_parquet(MASTER_DATA_PATH)
        logger.info(f"✅ Saved merged dataset: {len(combined_df)} total samples (+{len(new_df)} new)")
        
        # Clear processed logs
        if os.path.exists(LOGS_PATH):
            # Archive instead of deleting
            os.makedirs(ARCHIVE_PATH, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_file = os.path.join(ARCHIVE_PATH, f"logs_{timestamp}.jsonl")
            
            os.rename(LOGS_PATH, archive_file)
            logger.info(f"📦 Archived logs to {archive_file}")
            
            # Create new empty log file
            open(LOGS_PATH, 'w').close()
        
        stats["status"] = "success"
        return stats
        
    except Exception as e:
        logger.error(f"❌ Merge failed: {e}", exc_info=True)
        stats["status"] = "failed"
        stats["error"] = str(e)
        return stats

# DATA VERSIONING
@task(name="Version Control Data",log_prints=True,retries=1)
def version_control_data() -> bool:
    logger.info("📦 Versioning data with DVC...")
    
    try:
        # Check if DVC is initialized
        if not os.path.exists('.dvc'):
            logger.warning("⚠️  DVC not initialized, skipping versioning")
            return False
        
        # Add to DVC
        result = subprocess.run(
            ["dvc", "add", MASTER_DATA_PATH],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"✅ DVC add successful")
        
        # Push to remote
        result = subprocess.run(["dvc", "push"], capture_output=True, text=True, check=True)
        logger.info(f"✅ DVC push successful")
        
        # Commit .dvc file to Git
        dvc_file = f"{MASTER_DATA_PATH}.dvc"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        subprocess.run(["git", "add", dvc_file], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Data ingestion: {timestamp}"],
            check=True
        )
        
        logger.info("✅ Data versioned and committed to Git")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Version control failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Version control error: {e}")
        return False

# DRIFT DETECTION
@task(name="Detect Data Drift", log_prints=True)
def detect_data_drift(new_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare new data distribution with master dataset to detect drift.
    
    Checks:
    - Feature distribution shifts (KS test)
    - Label distribution changes
    - New vocabulary (unseen words)
    """
    logger.info("🔍 Checking for data drift...")
    
    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": False,
        "details": {}
    }
    
    try:
        if not os.path.exists(MASTER_DATA_PATH):
            logger.info("No master dataset for comparison")
            return drift_report
        
        master_df = pd.read_parquet(MASTER_DATA_PATH)
        
        # Label distribution drift
        new_label_dist = new_df['label'].value_counts(normalize=True)
        master_label_dist = master_df['label'].value_counts(normalize=True)
        
        label_drift = abs(new_label_dist - master_label_dist).max()
        drift_report["details"]["label_distribution_drift"] = float(label_drift)
        
        if label_drift > 0.1:  # 10% threshold
            logger.warning(f"⚠️  Label distribution drift detected: {label_drift:.2%}")
            drift_report["drift_detected"] = True
        
        # Feature distribution drift (simple version)
        numeric_features = ['msg_len', 'caps_ratio', 'slur_count']
        
        for feature in numeric_features:
            if feature in new_df.columns and feature in master_df.columns:
                new_mean = new_df[feature].mean()
                master_mean = master_df[feature].mean()
                
                relative_change = abs(new_mean - master_mean) / (master_mean + 1e-8)
                
                if relative_change > 0.3:  # 30% threshold
                    logger.warning(f"⚠️  {feature} drift: {relative_change:.2%} change")
                    drift_report["details"][f"{feature}_drift"] = float(relative_change)
                    drift_report["drift_detected"] = True
        
        if drift_report["drift_detected"]:
            logger.warning("⚠️  Data drift detected - consider retraining model")
        else:
            logger.info("✅ No significant drift detected")
        
        return drift_report
        
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        return drift_report

# MAIN FLOW
@flow(name="Data Ingestion Pipeline", log_prints=True)
def data_ingestion_flow():
    """
    Main orchestration flow for data ingestion.
    
    Steps:
    1. Load and validate raw logs
    2. Engineer features
    3. Run quality checks
    4. Merge with master dataset
    5. Version with DVC
    6. Detect drift
    7. Trigger retraining if needed
    """
    logger.info("🚀 Starting data ingestion pipeline...")
    
    new_data = load_and_validate_logs()
    
    if new_data is None:
        logger.info("✅ No new data to process")
        return
    
    processed_data = calculate_features(new_data)
    
    quality_ok = validate_data_quality(processed_data)
    
    if not quality_ok:
        logger.error("Data quality checks failed - aborting ingestion")
        return
    
    merge_stats = merge_and_save(processed_data)
    
    if merge_stats["status"] != "success":
        logger.error("Merge failed - aborting pipeline")
        return
    
    version_control_data()
    
    
    logger.info("\n" + "="*60)
    logger.info("📊 INGESTION SUMMARY")
    logger.info("="*60)
    logger.info(f"New samples ingested: {merge_stats['new_samples']}")
    logger.info(f"Total dataset size: {merge_stats['master_size_after']}")
    logger.info(f"Duplicates removed: {merge_stats['duplicates_removed']}")
    logger.info("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Ingestion Pipeline")
    parser.add_argument("--schedule", action="store_true", help="Deploy with 2-week schedule")
    parser.add_argument("--interval-days", type=int, default=14, help="Scheduling interval in days (default: 14)")
    
    args = parser.parse_args()
    
    if args.schedule:
        logger.info(f"📅 Deploying with {args.interval_days}-day schedule...")
        
        deployment = Deployment.build_from_flow(
            flow=data_ingestion_flow,
            name="biweekly-data-ingestion",
            schedule=IntervalSchedule(interval=timedelta(days=args.interval_days))
        )
        
        deployment.apply()
        logger.info("Deployment created successfully")
    else:
        # Run once
        data_ingestion_flow()

# # cyberbullying-bot/
# # │
# # ├── .gitignore                   # Standard python gitignore
# # ├── README.md                    # Architecture diagram & setup instructions
# # ├── requirements-dev.txt         # Dev tools (pytest, black, flake8)
# # │
# # ├── 📂 api_service/              # [Service B] The "Brain" (FastAPI + Model)
# # │   ├── app.py                   # Main FastAPI entrypoint
# # │   ├── config.py                # Env vars (Feast path, Model path)
# # │   ├── schemas.py               # Pydantic models (Input/Output validation)
# # │   ├── core/
# # │   │   ├── model.py             # DistilBERT + SVM loading & prediction logic
# # │   │   ├── rules.py             # Regex/Hardcoded rule engine
# # │   │   └── logger.py            # Structured JSON logger setup
# # │   ├── feature_store/           # Feast Configuration
# # │   │   ├── feature_store.yaml   # Connection to Offline(Parquet)/Online(SQLite)
# # │   │   └── definitions.py       # Entity & Feature View definitions
# # │   ├── artifacts/               # The "Artifact Store" (Git LFS tracked)
# # │   │   ├── svm_model_v1.pkl     # Trained Classifier
# # │   │   └── distilbert_config/   # (Optional) Tokenizer files
# # │   ├── Dockerfile               # Build instruction for API container
# # │   └── requirements.txt         # Dependencies (fastapi, torch, transformers)
# # │
# # ├── 📂 bot_service/              # [Service A] The "Enforcer" (Discord Bot)
# # │   ├── bot.py                   # Main Discord Bot entrypoint
# # │   ├── config.py                # Env vars (Discord Token, API URL)
# # │   ├── cogs/                    # Modular Bot Commands
# # │   │   ├── moderation.py        # The listener (on_message) logic
# # │   │   └── admin.py             # Ops commands (!ops inject_drift)
# # │   ├── Dockerfile               # Build instruction for Bot container
# # │   └── requirements.txt         # Dependencies (discord.py, requests)
# # │
# # ├── 📂 mlops/                    # The "Level 4" Automation Scripts
# # │   ├── generate_data.py         # Script to create fake user history (for Feast)
# # │   ├── train.py                 # Script to train SVM from scratch
# # │   ├── drift_monitor.py         # Script running Evidently AI checks
# # │   └── retrain_flow.py          # Prefect flow (Drift -> Train -> Deploy)
# # │
# # ├── 📂 copilot/                  # AWS Infrastructure (Auto-generated)
# # │   ├── api/                     # Manifest for Backend Service
# # │   └── bot/                     # Manifest for Worker Service
# # │
# # └── 📂 data/                     # Local Data Lake (Git Ignored)
# #     ├── raw_logs.json            # Appended logs from API
# #     ├── registry.db              # SQLite Online Store (Feast)
# #     └── offline_store/           # Parquet files
# #         └── user_stats.parquet