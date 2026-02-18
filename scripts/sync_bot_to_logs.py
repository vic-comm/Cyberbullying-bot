import sqlite3
import json
import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any
from mlops.utils import calculate_text_features, compute_text_hash

# Configuration
BOT_DB_PATH = os.getenv("BOT_DB_PATH", "bot_memory.db")
RAW_LOGS_PATH = os.getenv("LOGS_PATH", "data/raw_logs.jsonl")
SYNC_LOOKBACK_HOURS = int(os.getenv("SYNC_LOOKBACK_HOURS", "24"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_from_bot_db(lookback_hours: int = 86700) -> List[Dict[str, Any]]:
    """
    Extract new records from bot SQLite database.
    """
    if not os.path.exists(BOT_DB_PATH):
        logger.error(f"❌ Bot database not found at: {BOT_DB_PATH}")
        return []
    
    logger.info(f"📂 Connecting to {BOT_DB_PATH}...")
    
    try:
        conn = sqlite3.connect(BOT_DB_PATH)
        conn.row_factory = sqlite3.Row
        
        # Calculate cutoff time
        cutoff = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()
        
        
        query = """
        SELECT 
            user_id,
            message AS text,
            timestamp,
            severity,
            toxicity_score AS confidence,
            action_taken,
            metadata
        FROM logs
        WHERE 
            timestamp > ?
        ORDER BY timestamp DESC
        """
        
        rows = conn.execute(query, (cutoff,)).fetchall()
        logger.info(f"✅ Extracted {len(rows)} records from bot database")
        
        conn.close()
        
        # Convert to list of dicts
        records = []
        for row in rows:
            records.append(dict(row))
        
        return records

    except Exception as e:
        logger.error(f"❌ Database Extraction Error: {e}")
        return []

def enrich_with_features(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add calculated features and standardize labels.
    """
    logger.info(f"🔧 Enriching {len(records)} records...")
    
    enriched = []
    for record in records:
        text = record.get('text', '')
        if not text:
            continue
            
        # 1. Derive Label from Severity (Since 'verified_label' column doesn't exist)
        # We assume HIGH/MEDIUM/LOW = Toxic (1), SAFE/UNCERTAIN = Safe (0)
        # In a real system, you'd want manual verification here.
        severity = record.get('severity', 'UNCERTAIN')
        if severity in ['HIGH', 'MEDIUM', 'LOW']:
            label = 1
            label_source = 'model_derived'
        else:
            label = 0 # Treating Uncertain as Safe for now, or you could skip them
            label_source = 'model_derived'

        # Optional: Use metadata if you store reports there
        metadata = json.loads(record.get('metadata', '{}'))
        report_count = metadata.get('report_count', 0)
        
        # 2. Calculate static features
        features = calculate_text_features(text)
        
        # 3. Build final record
        enriched_record = {
            'text': text,
            'user_id': record['user_id'],
            'timestamp': str(record['timestamp']),
            'label': label,
            'label_source': label_source,
            'confidence': record.get('confidence'),
            'severity': severity,
            'report_count': report_count,
            'text_hash': compute_text_hash(text),
            'ingested_at': datetime.now().isoformat(),
            **features 
        }
        
        enriched.append(enriched_record)
    
    return enriched

def append_to_logs(records: List[Dict[str, Any]]) -> int:
    """
    Append records to raw_logs.jsonl for ingestion.
    """
    if not records:
        return 0
    
    os.makedirs(os.path.dirname(RAW_LOGS_PATH), exist_ok=True)
    
    # Load existing hashes to prevent duplicates
    existing_hashes = set()
    if os.path.exists(RAW_LOGS_PATH):
        try:
            with open(RAW_LOGS_PATH, 'r') as f:
                for line in f:
                    try:
                        log = json.loads(line)
                        if 'text_hash' in log:
                            existing_hashes.add(log['text_hash'])
                    except: continue
        except Exception as e:
            logger.warning(f"Could not read existing logs: {e}")

    # Filter out duplicates
    new_records = [r for r in records if r['text_hash'] not in existing_hashes]
    
    if not new_records:
        logger.info(f"⚠️ All {len(records)} records are duplicates.")
        return 0
    
    # Append to file
    with open(RAW_LOGS_PATH, 'a') as f:
        for record in new_records:
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"✅ Appended {len(new_records)} new records to {RAW_LOGS_PATH}")
    return len(new_records)

def sync_bot_to_logs(lookback_hours: int = None) -> int:
    lookback_hours = lookback_hours or SYNC_LOOKBACK_HOURS
    
    records = extract_from_bot_db(lookback_hours)
    if not records:
        return 0
        
    enriched = enrich_with_features(records)
    count = append_to_logs(enriched)
    
    return count

if __name__ == "__main__":
    sync_bot_to_logs()