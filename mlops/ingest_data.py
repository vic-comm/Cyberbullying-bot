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
import asyncio
from prefect import task, flow
import asyncpg
from mlops.feature_store import FeatureStore
# CONFIGURATION
import boto3
LOGS_PATH = os.getenv("LOGS_PATH", "data/raw_logs.jsonl")
MASTER_DATA_PATH = os.getenv("MASTER_DATA_PATH", "data/training_data_with_history.parquet")
BACKUP_PATH = os.getenv("BACKUP_PATH", "data/training_data_backup.parquet")
ARCHIVE_PATH = os.getenv("ARCHIVE_PATH", "data/archives")
FEATURE_CONFIG_PATH = os.getenv("FEATURE_CONFIG", "config/features.json")
BUCKET_NAME = os.getenv("BUCKET_NAME") 
S3_MASTER_KEY = os.getenv("S3_MASTER_KEY", "data/training_data_with_history.parquet")
DATABASE_URL = os.getenv("DATABASE_URL")
INGESTION_LOOKBACK_HOURS = int(os.getenv("INGESTION_LOOKBACK_HOURS", "24")) 
PLATFORMS_TO_INGEST = os.getenv("PLATFORMS_TO_INGEST", "discord,slack,whatsapp").split(",")
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

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'ingestion.log')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FEATURE_CONFIG_PATH, exist_ok=True)
os.makedirs(ARCHIVE_PATH, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def fetch_logs_from_supabase(
    lookback_hours: int = 24,
    platforms: List[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch recent logs from Supabase for training data ingestion.
    
    Replaces reading from JSONL files - now queries PostgreSQL directly.
    """
    if not DATABASE_URL:
        logger.error("❌ DATABASE_URL not set - cannot fetch logs")
        return None
    
    platforms = platforms or PLATFORMS_TO_INGEST
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    
    logger.info(f"📂 Fetching logs from Supabase...")
    logger.info(f"   Platforms: {platforms}")
    logger.info(f"   Since: {cutoff.isoformat()}")
    
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Query logs with quality filters
        query = """
            SELECT 
                l.id,
                l.user_id,
                l.server_id,
                l.platform,
                l.message as text,
                l.toxicity_score,
                l.severity,
                l.action_taken,
                l.timestamp,
                l.explanation,
                l.metadata
            FROM logs l
            WHERE l.timestamp > $1
              AND l.platform = ANY($2)
              AND l.toxicity_score IS NOT NULL
              AND l.severity IS NOT NULL
              AND LENGTH(l.message) >= $3
              AND LENGTH(l.message) <= $4
            ORDER BY l.timestamp DESC
        """
        
        rows = await conn.fetch(
            query,
            cutoff,
            platforms,
            MIN_TEXT_LENGTH,
            MAX_TEXT_LENGTH
        )
        
        await conn.close()
        
        if not rows:
            logger.warning("⚠️ No logs found in specified time window")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in rows])
        
        logger.info(f"✅ Fetched {len(df)} logs from database")
        
        # Create label from severity
        severity_to_label = {
            'SAFE': 0,
            'UNCERTAIN': 0,
            'LOW': 1,
            'MEDIUM': 1,
            'HIGH': 1
        }
        
        df['label'] = df['severity'].map(severity_to_label)
        df['label_source'] = 'production'  # From actual bot decisions
        
        # Add text hash for deduplication
        df['text_hash'] = df['text'].apply(lambda x: hash(x))
        
        # Platform distribution
        platform_dist = df['platform'].value_counts()
        logger.info(f"   Platform distribution: {platform_dist.to_dict()}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch logs from Supabase: {e}", exc_info=True)
        return None

# DATA LOADING AND VALIDATION
@task(name="Load and Validate Logs", log_prints=True, retries=2, retry_delay_seconds=30)
def load_and_validate_logs() -> Optional[pd.DataFrame]:
    """
    Load logs from Supabase (replaces JSONL loading).
    """
    logger.info(f"📂 Loading logs from Supabase (last {INGESTION_LOOKBACK_HOURS}h)...")
    
    # Run async fetch in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        df = loop.run_until_complete(
            fetch_logs_from_supabase(
                lookback_hours=INGESTION_LOOKBACK_HOURS,
                platforms=PLATFORMS_TO_INGEST
            )
        )
    finally:
        loop.close()
    
    if df is None or df.empty:
        logger.warning("⚠️ No data available for ingestion")
        return None
    
    logger.info(f"✅ Loaded {len(df)} raw records from database")
    
    # Validate data quality
    return validate_incoming_data(df)

# FEATURE ENGINEERING
@task(name="Calculate Features", log_prints=True)
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"🔧 Engineering features for {len(df)} samples...")
    
    df = df.copy()
    
    logger.info("   -> Applying centralized text feature logic...")
    
    feature_dicts = df['text'].apply(calculate_text_features).tolist()
    
    feature_df = pd.DataFrame(feature_dicts)
    
   
    df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    
    def extract_metadata_features(row):
        meta = row.get('metadata')
        if not isinstance(meta, dict):
            return pd.Series()
        
        # Extract specific features logged by the bot
        return pd.Series({
            'user_bad_ratio_7d': meta.get('user_bad_ratio_7d', 0.0),
            'user_toxicity_trend': meta.get('user_toxicity_trend', 0.0),
            'channel_toxicity_ratio': meta.get('channel_toxicity_ratio', 0.0),
            'is_new_to_channel': int(meta.get('is_new_to_channel', 0))
        })

    # Apply extraction
    if 'metadata' in df.columns:
        meta_features = df.apply(extract_metadata_features, axis=1)
        df = pd.concat([df, meta_features], axis=1)
        
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


@task(name="Validate Data Quality", log_prints=True)
def validate_data_quality(df: pd.DataFrame) -> bool:
    """
    Acts as a quality gate. 
    Returns False (BLOCKING) for Broken Pipelines or Attacks.
    Returns True (WARNING) for Class Imbalance or Concept Drift.
    """
    logger.info("🔍 Running strict data quality checks...")
    
    blocking_issues = []
    warnings = []
    
    # ── 1. CRITICAL: DATA QUALITY DRIFT (Broken Pipeline) ────────────────
    # Check A: Essential Columns are Missing/Null
    critical_cols = ['text', 'user_id', 'timestamp']
    for col in critical_cols:
        if col not in df.columns:
            blocking_issues.append(f"❌ Missing critical column: {col}")
        elif df[col].isnull().any():
            null_count = df[col].isnull().sum()
            blocking_issues.append(f"❌ Nulls found in critical column '{col}': {null_count} rows")

    # Check B: Text Content Integrity (Empty strings or whitespace)
    if 'text' in df.columns:
        empty_text_count = df[df['text'].str.strip() == ''].shape[0]
        if empty_text_count > 0:
            blocking_issues.append(f"❌ Found {empty_text_count} rows with empty/whitespace text")

    # Check C: Message Length Corruption (e.g., all 0s)
    if 'msg_len' in df.columns:
        if (df['msg_len'] == 0).all():
             blocking_issues.append("❌ Critical: 'msg_len' is 0 for ALL rows (Pipeline bug?)")

    # Check D: Spam/Repetition Attack (Identical content spam)
    # If >50% of the dataset is duplicates of the same 1 message
    if 'text' in df.columns and len(df) > 50:
        most_common_msg = df['text'].mode()[0]
        repetition_count = (df['text'] == most_common_msg).sum()
        repetition_rate = repetition_count / len(df)
        
        if repetition_rate > 0.5:  # Threshold: 50% identical messages
            blocking_issues.append(f"❌ ADVERSARIAL ATTACK DETECTED: {repetition_rate:.1%} of data is identical spam.")
            logger.error(f"   Spam content sample: '{most_common_msg[:50]}...'")

    # Check E: Bot/Script Flooding (One user sending >30% of all data)
    if 'user_id' in df.columns and len(df) > 50:
        top_user_share = df['user_id'].value_counts(normalize=True).iloc[0]
        if top_user_share > 0.3: # Threshold: 1 user sent 30% of batch
            blocking_issues.append(f"❌ ADVERSARIAL ATTACK: Single user sent {top_user_share:.1%} of all messages.")

    # Check F: Class Imbalance (Warning only)
    if 'label' in df.columns:
        label_dist = df['label'].value_counts(normalize=True)
        minority_class_ratio = label_dist.min()
        if minority_class_ratio < 0.05: # Stricter 5%
            warnings.append(f"⚠️ Severe class imbalance: Minority class at {minority_class_ratio:.2%}")

    # Check G: High Nulls in Non-Critical Features
    null_counts = df.isnull().sum()
    high_null_features = null_counts[null_counts > len(df) * 0.2]
    high_null_features = high_null_features.drop(labels=critical_cols, errors='ignore') # Ignore criticals already checked
    if not high_null_features.empty:
        warnings.append(f"⚠️ High nulls in features: {list(high_null_features.index)}")
    
    # Print Warnings (Don't stop)
    for w in warnings:
        logger.warning(w)

    # Print Blocking Issues (STOP PIPELINE)
    if blocking_issues:
        logger.error("🛑 CRITICAL DATA QUALITY FAILURE - INGESTION ABORTED")
        for issue in blocking_issues:
            logger.error(issue)
        return False  # BLOCK INGESTION

    logger.info("✅ Data Quality & Adversarial Checks Passed")
    return True   # ALLOW INGESTION

def pull_master_data():
    if os.path.exists(MASTER_DATA_PATH):
        logger.info("✅ Master data already exists locally.")
        return

    logger.info("📉 Pulling Master Data (Remote -> Local)...")
    
    try:
        subprocess.run(["dvc", "pull", MASTER_DATA_PATH, "--force"], check=True, capture_output=True)
        logger.info("✅ DVC Pull successful")
        return
    except Exception as e:
        logger.warning(f"⚠️ DVC Pull failed: {e}")

    if BUCKET_NAME:
        try:
            logger.info("🔄 Attempting direct S3 download...")
            s3 = boto3.client('s3')
            os.makedirs(os.path.dirname(MASTER_DATA_PATH), exist_ok=True)
            s3.download_file(BUCKET_NAME, S3_MASTER_KEY, MASTER_DATA_PATH)
            logger.info("✅ Direct S3 Download successful")
        except Exception as e:
            logger.warning(f"❌ Direct S3 Download failed: {e}")

def push_master_data():
    logger.info("📈 Pushing Master Data (Local -> Remote)...")
    
    try:
        subprocess.run(["dvc", "add", MASTER_DATA_PATH], check=True, capture_output=True)
        subprocess.run(["dvc", "push", MASTER_DATA_PATH], check=True, capture_output=True)
        logger.info("✅ DVC Push successful")
        return
    except Exception as e:
        logger.warning(f"⚠️ DVC Push failed: {e}")

    if BUCKET_NAME:
        try:
            logger.info("🔄 Attempting direct S3 upload...")
            s3 = boto3.client('s3')
            s3.upload_file(MASTER_DATA_PATH, BUCKET_NAME, S3_MASTER_KEY)
            logger.info("✅ Direct S3 Upload successful")
        except Exception as e:
            logger.error(f"❌ Direct S3 Upload failed: {e}")


def merge_and_save(new_df: pd.DataFrame) -> Dict[str, Any]:
    stats = {"new_samples": len(new_df), "status": "pending"}
    
    try:
        # A. DOWNLOAD HISTORY
        pull_master_data()
        
        # B. LOAD
        if os.path.exists(MASTER_DATA_PATH):
            master_df = pd.read_parquet(MASTER_DATA_PATH)
            stats["master_size_before"] = len(master_df)
        else:
            master_df = pd.DataFrame()
            stats["master_size_before"] = 0
            
        # C. ALIGN COLUMNS (Fix mismatch errors)
        if not master_df.empty:
            for col in master_df.columns:
                if col not in new_df.columns:
                    new_df[col] = 0 if pd.api.types.is_numeric_dtype(master_df[col]) else None
            new_df = new_df[master_df.columns.intersection(new_df.columns)]

        # D. MERGE
        combined_df = pd.concat([master_df, new_df], ignore_index=True)
        
        # E. DEDUPLICATE (Critical step)
        # Drop duplicates based on hash, keep the LAST (newest) one
        if 'text_hash' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['text_hash'], keep='last')
            
        stats["master_size_after"] = len(combined_df)
        stats["duplicates_removed"] = (stats["master_size_before"] + len(new_df)) - len(combined_df)

        # F. SAVE LOCAL
        combined_df.to_parquet(MASTER_DATA_PATH)
        
        # G. UPLOAD HISTORY
        push_master_data()
        
        # H. CLEANUP LOCAL LOGS (To keep container clean)
        if os.path.exists(LOGS_PATH):
            os.remove(LOGS_PATH)
            
        stats["status"] = "success"
        return stats

    except Exception as e:
        logger.error(f"❌ Merge failed: {e}", exc_info=True)
        stats["status"] = "failed"
        return stats
    
def validate_incoming_data(new_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Validates the DataFrame passed from the Orchestrator.
    """
    if new_data is None or new_data.empty:
        logger.warning("⚠️ Received empty DataFrame for validation")
        return None
    
    logger.info(f"🔍 Validating {len(new_data)} raw records...")
    
    try:
        # Validate required columns
        required_cols = ['text', 'user_id', 'timestamp']
        missing_cols = [col for col in required_cols if col not in new_data.columns]
        
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Filter by text length
        valid_data = new_data[
            (new_data['text'].str.len() >= MIN_TEXT_LENGTH) &
            (new_data['text'].str.len() <= MAX_TEXT_LENGTH)
        ].copy()
        
        logger.info(f"✅ {len(valid_data)} records passed length checks")
        
        # Determine label source (Human vs Model)
        if 'verified_label' in valid_data.columns:
            valid_data = valid_data[valid_data['verified_label'].notna()].copy()
            valid_data['label'] = valid_data['verified_label'].astype(int)
            valid_data['label_source'] = 'human'
            logger.info(f"✅ Using {len(valid_data)} human-verified labels")
            
        elif 'prediction' in valid_data.columns:
            valid_data = valid_data[valid_data['prediction'].notna()].copy()
            valid_data['label'] = valid_data['prediction'].astype(int)
            valid_data['label_source'] = 'model'
            logger.warning(f"⚠️ Using {len(valid_data)} model predictions as pseudo-labels")
            
        # Fallback: Try to use 'label' column if it exists (e.g. from sync script)
        elif 'label' in valid_data.columns:
             valid_data['label_source'] = 'existing'
             logger.info(f"✅ Using {len(valid_data)} existing labels")
        else:
            logger.error("❌ No label column found (need 'verified_label', 'prediction', or 'label')")
            return None
        
        if len(valid_data) < MIN_NEW_SAMPLES:
            logger.warning(f"⚠️ Only {len(valid_data)} valid samples (minimum: {MIN_NEW_SAMPLES})")
            return None
        
        # Add metadata
        valid_data['ingested_at'] = datetime.now().isoformat()
        
        # Deduplicate by text hash within the batch
        if 'text_hash' in valid_data.columns:
            before = len(valid_data)
            valid_data = valid_data.drop_duplicates(subset=['text_hash'])
            after = len(valid_data)
            if before != after:
                logger.info(f"🔄 Removed {before - after} duplicate samples inside this batch")
        
        return valid_data
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}", exc_info=True)
        return None
    
# MAIN FLOW
@flow(name="Data Ingestion Pipeline", log_prints=True)
def data_ingestion_flow(new_data: pd.DataFrame = None):
    logger.info("🚀 Starting data ingestion pipeline...")
    if not new_data:
        new_data = load_and_validate_logs()
    else:
        new_data = validate_incoming_data(new_data)
    

    if new_data is None:
        logger.info("✅ No new data to process")
        return {"status": "skipped", "reason": "no_new_data"}
    
    processed_data = calculate_features(new_data)
    
    quality_ok = validate_data_quality(processed_data)
    
    if not quality_ok:
        logger.error("Data quality checks failed - aborting ingestion")
        return {"status": "failed", "reason": "quality_check_failed"}
    
    merge_stats = merge_and_save(processed_data)
    
    if merge_stats["status"] != "success":
        logger.error("Merge failed - aborting pipeline")
        return merge_stats
    
    
    logger.info("🔄 Syncing updated features to Redis...")
    fs = FeatureStore()
    fs.sync_offline_to_online(
        parquet_path=MASTER_DATA_PATH,
        feature_group_name="user_toxicity",
        version="prod",
        entity_key="user_id"
    )
    logger.info("✅ Feature Store sync complete")

    logger.info("\n" + "="*60)
    logger.info("📊 INGESTION SUMMARY")
    logger.info("="*60)
    logger.info(f"New samples ingested: {merge_stats['new_samples']}")
    logger.info(f"Total dataset size: {merge_stats['master_size_after']}")
    logger.info(f"Duplicates removed: {merge_stats['duplicates_removed']}")
    logger.info("="*60)

    return merge_stats


if __name__ == "__main__":
    data_ingestion_flow()

