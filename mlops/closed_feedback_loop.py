import sqlite3
import pandas as pd
import numpy as np
import os
import argparse
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
import hashlib
from .utils import calculate_text_features
# Rich terminal formattin
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

DB_PATH = os.getenv("BOT_DB_PATH", "bot_memory.db")
LOGS_PATH = os.getenv("LOGS_PATH", "data/raw_logs.jsonl")
AUDIT_PATH = os.getenv("AUDIT_PATH", "data/labeling_audit.jsonl")
FEATURE_CONFIG_PATH = os.getenv("FEATURE_CONFIG", "config/features.json")

UNCERTAINTY_THRESHOLD = 0.3  # |confidence - 0.5| < 0.3
MIN_REPORTS_FOR_REVIEW = 2   
DAYS_LOOKBACK = 7            

TOXIC_KEYWORDS = {
    'slurs': ['trash', 'scum', 'garbage', 'loser', 'idiot', 'stupid', 'dumb'],
    'threats': ['kill', 'die', 'hurt', 'attack', 'destroy'],
    'harassment': ['ugly', 'fat', 'worthless', 'pathetic', 'waste']
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feedback_loop.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# UTILITY FUNCTIONS
def calculate_static_features(text: str) -> Dict[str, Any]:
    text = str(text)
    words = text.lower().split()
    
    features = {
        # Basic text stats
        "msg_len": len(text),
        "word_count": len(words),
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0.0,
        
        # Character patterns
        "caps_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0.0,
        "punctuation_ratio": sum(1 for c in text if c in '!?.,;:') / len(text) if text else 0.0,
        "emoji_count": sum(1 for c in text if ord(c) > 127000),  # Rough emoji detection
        
        # Linguistic features
        "personal_pronoun_count": sum(1 for w in words if w in ['i', 'me', 'my', 'mine', 'you', 'your', 'yours']),
        "question_mark_count": text.count('?'),
        "exclamation_count": text.count('!'),
        "all_caps_words": sum(1 for w in words if w.isupper() and len(w) > 1),
        
        # Toxicity signals
        "slur_count": sum(1 for w in words if any(s in w for s in TOXIC_KEYWORDS['slurs'])),
        "threat_count": sum(1 for w in words if any(t in w for t in TOXIC_KEYWORDS['threats'])),
        "harassment_count": sum(1 for w in words if any(h in w for h in TOXIC_KEYWORDS['harassment'])),
        
        # Initialize user history features (will be computed separately)
        "user_bad_ratio_7d": 0.0,
        "user_bad_ratio_30d": 0.0,
        "user_toxicity_trend": 0.0,
        "user_msg_count_7d": 0.0,
        "channel_toxicity_ratio": 0.0,
        "hours_since_last_msg": 1.0,
        "is_new_to_channel": 0,
        "user_report_count": 0
    }
    
    return features

def compute_text_hash(text: str) -> str:
    """Create unique identifier for deduplication"""
    return hashlib.md5(text.encode()).hexdigest()

def load_existing_labels() -> set:
    labeled_hashes = set()
    
    if os.path.exists(LOGS_PATH):
        try:
            df = pd.read_json(LOGS_PATH, lines=True)
            if 'text_hash' in df.columns:
                labeled_hashes = set(df['text_hash'].values)
        except Exception as e:
            logger.warning(f"Could not load existing labels: {e}")
    
    logger.info(f"Loaded {len(labeled_hashes)} previously labeled samples")
    return labeled_hashes

# DATABASE OPERATIONS
def fetch_candidates(lookback_days: int = DAYS_LOOKBACK, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Extract high-value labeling candidates from production database.
    
    Priority scoring:
    1. User-reported messages (human flags)
    2. Model uncertainty (confidence near 0.5)
    3. Edge cases (unusual feature combinations)
    4. Recent samples (temporal diversity)
    
    Returns:
        List of candidate dicts sorted by priority
    """
    if not os.path.exists(DB_PATH):
        logger.error(f"Database not found at {DB_PATH}")
        return []
    
    logger.info(f"Fetching candidates from {DB_PATH}...")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Calculate date threshold
    cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    
    # Complex query to get diverse candidates
    query = """
    WITH scored_candidates AS (
        SELECT 
            user_id,
            message,
            severity,
            confidence,
            action_taken,
            timestamp,
            channel_id,
            report_count,
            -- Priority scoring
            CASE 
                WHEN report_count >= ? THEN 100  -- User reports highest priority
                WHEN action_taken = 'FLAGGED_REVIEW' THEN 80
                WHEN severity = 'UNCERTAIN' THEN 60
                WHEN ABS(confidence - 0.5) < ? THEN 40  -- Near decision boundary
                ELSE 20
            END as priority_score
        FROM logs
        WHERE timestamp > ?
            AND message IS NOT NULL
            AND LENGTH(message) > 10  -- Filter very short messages
        ORDER BY priority_score DESC, timestamp DESC
        LIMIT ?
    )
    SELECT * FROM scored_candidates
    ORDER BY priority_score DESC, RANDOM()  -- Add randomness for diversity
    """
    
    rows = conn.execute(
        query,
        (MIN_REPORTS_FOR_REVIEW, UNCERTAINTY_THRESHOLD, cutoff_date, limit * 2)
    ).fetchall()
    
    conn.close()
    
    # Convert to dicts and deduplicate
    candidates = []
    seen_hashes = load_existing_labels()
    
    for row in rows:
        text_hash = compute_text_hash(row['message'])
        
        # Skip if already labeled
        if text_hash in seen_hashes:
            continue
        
        candidates.append({
            'user_id': row['user_id'],
            'message': row['message'],
            'severity': row['severity'],
            'confidence': row['confidence'],
            'action_taken': row['action_taken'],
            'timestamp': row['timestamp'],
            'channel_id': row['channel_id'],
            'report_count': row['report_count'] if row['report_count'] else 0,
            'priority_score': row['priority_score'],
            'text_hash': text_hash
        })
        
        if len(candidates) >= limit:
            break
    
    logger.info(f"Found {len(candidates)} unique candidates for labeling")
    return candidates

def mark_as_reviewed(conn: sqlite3.Connection, user_id: str, message: str):
    """Mark samples as reviewed in database to prevent re-labeling"""
    try:
        conn.execute("""
            UPDATE logs 
            SET action_taken = 'HUMAN_REVIEWED',
                review_timestamp = ?
            WHERE user_id = ? AND message = ?
        """, (datetime.now().isoformat(), user_id, message))
        conn.commit()
    except Exception as e:
        logger.warning(f"Could not mark as reviewed: {e}")

# ============================================================================
# INTERACTIVE LABELING
# ============================================================================
def display_candidate(candidate: Dict[str, Any], index: int, total: int):
    """Pretty-print candidate info"""
    if RICH_AVAILABLE:
        table = Table(title=f"Sample {index}/{total}", show_header=False)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Message", candidate['message'])
        table.add_row("User ID", candidate['user_id'])
        table.add_row("Channel", candidate['channel_id'])
        table.add_row("Timestamp", candidate['timestamp'])
        table.add_row("Confidence", f"{candidate['confidence']:.2f}" if candidate['confidence'] else "N/A")
        table.add_row("Reports", str(candidate['report_count']))
        table.add_row("Priority", f"{candidate['priority_score']}/100")
        
        console.print(table)
    else:
        print(f"\n{'='*60}")
        print(f"Sample {index}/{total}")
        print(f"{'='*60}")
        print(f"💬 Message: {candidate['message']}")
        print(f"👤 User: {candidate['user_id']}")
        print(f"📍 Channel: {candidate['channel_id']}")
        print(f"🕒 Time: {candidate['timestamp']}")
        print(f"🎯 Confidence: {candidate['confidence']:.2f}" if candidate['confidence'] else "N/A")
        print(f"🚩 Reports: {candidate['report_count']}")
        print(f"⭐ Priority: {candidate['priority_score']}/100")

def get_label_input() -> Optional[str]:
    """Get label from user with validation"""
    prompt = "\n👉 Label? [1=Toxic, 0=Safe, u=Uncertain, s=Skip, q=Quit, b=Back]: "
    
    while True:
        choice = input(prompt).strip().lower()
        
        if choice in ['1', '0', 'u', 's', 'q', 'b']:
            return choice
        
        print("❌ Invalid input. Please use: 1, 0, u, s, q, or b")

def interactive_labeling(
    candidates: List[Dict[str, Any]],
    batch_mode: bool = False
) -> List[Dict[str, Any]]:
    """
    Present candidates to human labeler with interactive CLI.
    
    Args:
        candidates: List of samples to label
        batch_mode: If True, save after each label (for interruption safety)
        
    Returns:
        List of labeled records ready for training
    """
    labeled_data = []
    label_counts = defaultdict(int)
    
    print("\n" + "="*70)
    print("🏷️  HUMAN FEEDBACK SESSION")
    print("="*70)
    print("Instructions:")
    print("  [1] = Toxic/Harmful (cyberbullying, threats, harassment)")
    print("  [0] = Safe/Benign (normal conversation)")
    print("  [u] = Uncertain (need more context)")
    print("  [s] = Skip (come back to this later)")
    print("  [q] = Quit (save progress and exit)")
    print("  [b] = Back (relabel previous sample)")
    print("="*70 + "\n")
    
    i = 0
    history = []  # For "back" functionality
    
    while i < len(candidates):
        candidate = candidates[i]
        
        display_candidate(candidate, i + 1, len(candidates))
        choice = get_label_input()
        
        # Handle navigation
        if choice == 'q':
            print("\n💾 Saving progress and quitting...")
            break
        
        if choice == 's':
            print("⏭️  Skipped\n")
            i += 1
            continue
        
        if choice == 'b':
            if history:
                i = max(0, i - 1)
                print("⬅️  Going back...\n")
                # Remove last label if it was saved
                if labeled_data and labeled_data[-1]['text_hash'] == candidates[i]['text_hash']:
                    labeled_data.pop()
            else:
                print("⚠️  Already at first sample\n")
            continue
        
        # Convert label
        if choice == 'u':
            label = -1  # Special marker for uncertain
            label_str = "uncertain"
        else:
            label = int(choice)
            label_str = "toxic" if label == 1 else "safe"
        
        # Calculate features
        features = calculate_text_features(candidate['message'])
        
        # Create training record
        entry = {
            "text": candidate['message'],
            "user_id": candidate['user_id'],
            "channel_id": candidate['channel_id'],
            "timestamp": candidate['timestamp'],
            "verified_label": label,
            "labeler_confidence": 1.0,  # Could add confidence rating
            "labeled_at": datetime.now().isoformat(),
            "model_prediction": None,  # Filled if available
            "model_confidence": candidate.get('confidence'),
            "text_hash": candidate['text_hash'],
            "priority_score": candidate['priority_score'],
            "report_count": candidate['report_count'],
            **features
        }
        
        labeled_data.append(entry)
        history.append(i)
        label_counts[label_str] += 1
        
        print(f"✅ Labeled as '{label_str}'\n")
        
        # Auto-save in batch mode
        if batch_mode:
            save_labels([entry], append=True)
        
        i += 1
    
    # Show summary
    print("\n" + "="*70)
    print("📊 LABELING SUMMARY")
    print("="*70)
    print(f"Total labeled: {len(labeled_data)}")
    for label, count in label_counts.items():
        print(f"  {label.capitalize()}: {count}")
    print("="*70 + "\n")
    
    return labeled_data

# ============================================================================
# DATA PERSISTENCE
# ============================================================================
def save_labels(labeled_data: List[Dict[str, Any]], append: bool = True):
    """
    Save labeled data to JSONL file.
    Also maintains an audit trail of all labeling decisions.
    """
    if not labeled_data:
        logger.info("No new labels to save")
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
    
    mode = 'a' if append else 'w'
    
    try:
        # Save to main logs file
        with open(LOGS_PATH, mode) as f:
            for record in labeled_data:
                f.write(json.dumps(record) + "\n")
        
        logger.info(f"💾 Saved {len(labeled_data)} labels to {LOGS_PATH}")
        
        # Save audit trail (includes metadata)
        with open(AUDIT_PATH, 'a') as f:
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
                "num_labels": len(labeled_data),
                "label_distribution": dict(pd.Series([r['verified_label'] for r in labeled_data]).value_counts()),
            }
            f.write(json.dumps(audit_entry) + "\n")
        
        logger.info(f"📝 Audit trail updated: {AUDIT_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to save labels: {e}")
        raise

def update_bot_database(labeled_data: List[Dict[str, Any]]):
    """Mark reviewed samples in the bot database"""
    if not labeled_data:
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        for record in labeled_data:
            mark_as_reviewed(conn, record['user_id'], record['text'])
        
        conn.close()
        logger.info(f"✅ Marked {len(labeled_data)} samples as reviewed in bot DB")
        
    except Exception as e:
        logger.warning(f"Could not update bot database: {e}")

# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Human-in-the-Loop Feedback Collection for Cyberbullying Detection"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of samples to label (default: 50)"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DAYS_LOOKBACK,
        help=f"Days to look back for candidates (default: {DAYS_LOOKBACK})"
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Save after each label (safer for long sessions)"
    )
    parser.add_argument(
        "--priority-threshold",
        type=int,
        default=40,
        help="Minimum priority score to include (0-100, default: 40)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Show statistics without labeling"
    )
    
    args = parser.parse_args()
    
    # Fetch candidates
    candidates = fetch_candidates(
        lookback_days=args.lookback_days,
        limit=args.limit
    )
    
    if not candidates:
        print("✅ No new candidates to label!")
        return
    
    # Filter by priority
    candidates = [c for c in candidates if c['priority_score'] >= args.priority_threshold]
    
    if not candidates:
        print(f"✅ No candidates above priority threshold {args.priority_threshold}!")
        return
    
    # Show statistics
    if args.stats_only:
        print("\n📊 CANDIDATE STATISTICS")
        print("="*60)
        df = pd.DataFrame(candidates)
        print(f"Total candidates: {len(df)}")
        print(f"\nPriority distribution:")
        print(df['priority_score'].value_counts().sort_index(ascending=False))
        print(f"\nAction types:")
        print(df['action_taken'].value_counts())
        print(f"\nReport counts:")
        print(df['report_count'].value_counts())
        return
    
    # Interactive labeling
    labeled_data = interactive_labeling(candidates, batch_mode=args.batch_mode)
    
    # Save results
    if labeled_data:
        save_labels(labeled_data, append=True)
        update_bot_database(labeled_data)
        print(f"\n Feedback loop closed! {len(labeled_data)} new training samples collected.")
    else:
        print("\n  No samples were labeled.")

if __name__ == "__main__":
    main()


# [Discord User]
#              │
#       (Sends Message)
#              ▼
#       [Bot / API]  ──────────▶  (Uncertain Predictions)
#              │                              │
#              │                              ▼
#     (Logs to SQLite)            [bot_memory.db]
#              │                              │
#              │                   (3. close_feedback_loop.py)
#              │                              │
#              ▼                              ▼
#     [Raw Production Logs] ◀───── [Human Labeler (You)]
#              │
#     (4. ingest_data.py)
#              │
#              ▼
#     [Master Dataset (Parquet)] ──▶ [DVC / Git]
#              │
#     (5. pipeline.py)
#              │
#              ▼
#     [New Model Version] ──▶ [Deploy to API]