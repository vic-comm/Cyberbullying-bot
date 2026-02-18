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
import pandas as pd
import asyncio
import asyncpg
from datetime import datetime, timedelta
import logging
from typing import Optional, List
from prefect import task, flow
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

async def fetch_training_data_stratified(
    lookback_hours: int = 24,
    platforms: List[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch training data with stratified sampling + admin reviews.
    
    Returns:
    - 5% of high-confidence safe messages (anchors)
    - 5% of high-confidence toxic messages (anchors)
    - 100% of admin-reviewed disputes (corrections)
    - 100% of admin-reviewed uncertain messages (edge cases)
    """
    if not DATABASE_URL:
        logger.error("❌ DATABASE_URL not set")
        return None
    
    platforms = platforms or PLATFORMS_TO_INGEST
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    
    logger.info(f"📂 Fetching training data with admin-reviewed feedback...")
    logger.info(f"   Strategy: 5% anchors + 100% admin-reviewed")
    
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # ───────────────────────────────────────────────────────
        # PART 1: HIGH-CONFIDENCE ANCHORS (5% sample)
        # ───────────────────────────────────────────────────────
        
        anchors_query = """
        WITH high_conf_messages AS (
            SELECT 
                l.id,
                l.user_id,
                l.server_id,
                l.platform,
                l.message as text,
                l.toxicity_score,
                l.severity,
                l.timestamp,
                l.metadata,
                
                -- Label from severity
                CASE 
                    WHEN l.severity IN ('LOW', 'MEDIUM', 'HIGH') THEN 1
                    ELSE 0
                END AS label,
                
                'anchor' as source_type,
                'model_prediction' as label_source
                
            FROM logs l
            WHERE l.timestamp > $1
              AND l.platform = ANY($2)
              AND l.toxicity_score IS NOT NULL
              AND LENGTH(l.message) >= $3
              AND LENGTH(l.message) <= $4
              AND (
                  l.toxicity_score < $5  -- High conf safe
                  OR 
                  l.toxicity_score > $6  -- High conf toxic
              )
              AND random() < $7  -- 5% sample
        )
        SELECT * FROM high_conf_messages
        """
        
        anchors = await conn.fetch(
            anchors_query,
            cutoff,
            platforms,
            MIN_TEXT_LENGTH,
            MAX_TEXT_LENGTH,
            LOW_CONFIDENCE_THRESHOLD,   # 0.3
            HIGH_CONFIDENCE_THRESHOLD,  # 0.7
            HIGH_CONFIDENCE_SAMPLE_RATE # 0.05
        )
        
        anchors_df = pd.DataFrame([dict(row) for row in anchors])
        logger.info(f"   Anchors: {len(anchors_df)} messages (5% sample)")
        
        # ───────────────────────────────────────────────────────
        # PART 2: ADMIN-REVIEWED FEEDBACK (100%)
        # ───────────────────────────────────────────────────────
        
        feedback_query = """
        SELECT 
            l.id,
            l.user_id,
            l.server_id,
            l.platform,
            l.message as text,
            l.toxicity_score,
            l.severity,
            l.timestamp,
            l.metadata,
            
            -- Use ADMIN's final decision as label
            f.final_label as label,
            
            'feedback' as source_type,
            CASE 
                WHEN f.admin_decision = 'agree_with_user' THEN 'admin_corrected'
                WHEN f.admin_decision = 'agree_with_model' THEN 'admin_approved'
                ELSE 'admin_custom'
            END as label_source,
            
            f.admin_decision,
            f.reviewed_by,
            f.dispute_reason
            
        FROM logs l
        JOIN feedback f ON l.id = f.log_id
        WHERE l.timestamp > $1
          AND l.platform = ANY($2)
          AND f.admin_reviewed = TRUE  -- Only admin-reviewed
          AND f.used_in_training = FALSE  -- Not yet used
          AND LENGTH(l.message) >= $3
          AND LENGTH(l.message) <= $4
        """
        
        feedback_rows = await conn.fetch(
            feedback_query,
            cutoff,
            platforms,
            MIN_TEXT_LENGTH,
            MAX_TEXT_LENGTH
        )
        
        feedback_df = pd.DataFrame([dict(row) for row in feedback_rows])
        logger.info(f"   Feedback: {len(feedback_df)} admin-reviewed disputes")
        
        # ───────────────────────────────────────────────────────
        # PART 3: ADMIN-REVIEWED UNCERTAIN (100%)
        # ───────────────────────────────────────────────────────
        
        # Note: Uncertain messages admin-reviewed are already in feedback table
        # This query would get uncertain messages NOT yet reviewed by anyone
        # You might want to exclude these from training until reviewed
        
        # For now, we only use:
        # - Anchors (high confidence, model prediction trusted)
        # - Admin-reviewed feedback (admin made final call)
        
        # ───────────────────────────────────────────────────────
        # COMBINE
        # ───────────────────────────────────────────────────────
        
        if anchors_df.empty and feedback_df.empty:
            logger.warning("⚠️ No training data available")
            await conn.close()
            return None
        
        # Concatenate
        if not anchors_df.empty and not feedback_df.empty:
            combined_df = pd.concat([anchors_df, feedback_df], ignore_index=True)
        elif not anchors_df.empty:
            combined_df = anchors_df
        else:
            combined_df = feedback_df
        
        # Mark feedback as used
        if not feedback_df.empty:
            feedback_ids = feedback_df['id'].tolist()
            await conn.execute('''
                UPDATE feedback 
                SET used_in_training = TRUE 
                WHERE log_id = ANY($1)
            ''', feedback_ids)
            logger.info(f"   Marked {len(feedback_ids)} feedback items as used")
        
        await conn.close()
        
        # Add text hash for deduplication
        combined_df['text_hash'] = combined_df['text'].apply(lambda x: hash(x))
        
        # ───────────────────────────────────────────────────────
        # STATISTICS
        # ───────────────────────────────────────────────────────
        
        logger.info(f"✅ Fetched {len(combined_df)} total training examples")
        logger.info(f"   Breakdown:")
        source_dist = combined_df['source_type'].value_counts()
        for source, count in source_dist.items():
            logger.info(f"     {source}: {count} ({count/len(combined_df):.1%})")
        
        label_dist = combined_df['label_source'].value_counts()
        logger.info(f"   Label sources:")
        for source, count in label_dist.items():
            logger.info(f"     {source}: {count}")
        
        # Label balance
        label_balance = combined_df['label'].value_counts()
        logger.info(f"   Label balance:")
        logger.info(f"     Safe (0): {label_balance.get(0, 0)}")
        logger.info(f"     Toxic (1): {label_balance.get(1, 0)}")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch training data: {e}", exc_info=True)
        return None

# DATA LOADING AND VALIDATION
@task(name="Load Training Data", log_prints=True, retries=2, retry_delay_seconds=30)
def load_and_validate_logs() -> Optional[pd.DataFrame]:
    """
    Load training data with admin-reviewed feedback integration.
    """
    logger.info(f"📂 Loading training data (last {INGESTION_LOOKBACK_HOURS}h)...")
    logger.info(f"   Mode: Stratified sampling + admin-reviewed feedback")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        df = loop.run_until_complete(
            fetch_training_data_stratified(
                lookback_hours=INGESTION_LOOKBACK_HOURS,
                platforms=PLATFORMS_TO_INGEST
            )
        )
    finally:
        loop.close()
    
    if df is None or df.empty:
        logger.warning("⚠️ No training data available")
        return None
    
    logger.info(f"✅ Loaded {len(df)} records")
    
    # Validate
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
    Enhanced validation with poisoning detection.
    """
    logger.info("🔍 Running data quality & security checks...")
    
    blocking_issues = []
    warnings = []
    
    # ─── CRITICAL: Required columns ───────────────────────────
    critical_cols = ['text', 'user_id', 'timestamp', 'label', 'label_source']
    for col in critical_cols:
        if col not in df.columns:
            blocking_issues.append(f"❌ Missing column: {col}")
        elif df[col].isnull().any():
            null_count = df[col].isnull().sum()
            blocking_issues.append(f"❌ Nulls in '{col}': {null_count}")
    
    # ─── SECURITY: Check label source distribution ────────────
    if 'label_source' in df.columns:
        # Ensure we have SOME admin-reviewed data
        admin_reviewed = df[df['label_source'].str.contains('admin', na=False)]
        admin_ratio = len(admin_reviewed) / len(df) if len(df) > 0 else 0
        
        if admin_ratio < 0.05 and len(df) > 100:
            warnings.append(
                f"⚠️ Only {admin_ratio:.1%} of data is admin-reviewed (expected 10%+)"
            )
    
    # ─── SECURITY: Detect spam/repetition ─────────────────────
    if 'text' in df.columns and len(df) > 50:
        most_common = df['text'].mode()[0] if not df['text'].mode().empty else ""
        repetition_count = (df['text'] == most_common).sum()
        repetition_rate = repetition_count / len(df)
        
        if repetition_rate > 0.5:
            blocking_issues.append(
                f"❌ SPAM ATTACK: {repetition_rate:.1%} identical messages"
            )
    
    # ─── SECURITY: Detect single-user flooding ────────────────
    if 'user_id' in df.columns and len(df) > 50:
        top_user_share = df['user_id'].value_counts(normalize=True).iloc[0]
        if top_user_share > 0.3:
            blocking_issues.append(
                f"❌ FLOODING: Single user sent {top_user_share:.1%} of data"
            )
    
    # ─── WARNING: Class imbalance ──────────────────────────────
    if 'label' in df.columns:
        label_dist = df['label'].value_counts(normalize=True)
        minority_ratio = label_dist.min()
        if minority_ratio < 0.05:
            warnings.append(
                f"⚠️ Severe class imbalance: {minority_ratio:.1%}"
            )
    
    # Print results
    for w in warnings:
        logger.warning(w)
    
    if blocking_issues:
        logger.error("🛑 INGESTION BLOCKED - CRITICAL ISSUES")
        for issue in blocking_issues:
            logger.error(issue)
        return False
    
    logger.info("✅ Data quality checks passed")
    return True

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
    fs.sync_from_supabase(
        database_url=DATABASE_URL,
        feature_group_name="user_toxicity",
        version="prod",
        lookback_days=30,
        ttl_days=7
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

