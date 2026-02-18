import sqlite3
import json
import os
from datetime import datetime
import pandas as pd

# CONFIGURATION
DB_PATH = "bot_memory.db"
OUTPUT_LOGS = "data/raw_logs.jsonl"

def export_db():
    print(f"🔌 Connecting to {DB_PATH}...")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database {DB_PATH} not found.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Query your exact table schema
        query = """
        SELECT 
            user_id,
            message as text,
            timestamp,
            severity,
            toxicity_score,
            action_taken
        FROM logs
        """
        
        df = pd.read_sql_query(query, conn)
        print(f"📊 Extracted {len(df)} records from 'logs' table.")

        if df.empty:
            print("⚠️ Table is empty. No logs to export.")
            return

        # --- DATA TRANSFORMATION ---
        
        # 1. Format Timestamp (ISO 8601 string)
        # SQLite often stores dates as strings, but we ensure standardization
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

        # 2. Derive Labels for Training
        # The pipeline needs 'verified_label' (0 or 1). 
        # We derive this from your 'severity' column.
        def derive_label(severity):
            if severity in ['HIGH', 'MEDIUM', 'LOW']:
                return 1  # Toxic
            return 0      # Safe / Uncertain

        df['verified_label'] = df['severity'].apply(derive_label)

        # 3. Add prediction score (optional but helpful metadata)
        df['prediction'] = df['verified_label'] # Fallback for ingestion logic

        # 4. Generate Text Hash (for deduplication in the pipeline)
        # We hash the text + timestamp to ensure unique IDs
        df['text_hash'] = df.apply(lambda x: str(hash(f"{x['text']}{x['timestamp']}")), axis=1)

        # --- EXPORT ---
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(OUTPUT_LOGS), exist_ok=True)
        
        print(f"📝 Writing to {OUTPUT_LOGS}...")
        
        # Select only the columns the pipeline expects
        export_df = df[['text', 'user_id', 'timestamp', 'verified_label', 'text_hash', 'toxicity_score']]
        
        with open(OUTPUT_LOGS, 'a') as f:
            for _, row in export_df.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")

        print(f"✅ Success! Exported {len(export_df)} logs.")
        print("🚀 Next Step: Run 'python mlops/ingest_data.py'")

    except sqlite3.Error as e:
        print(f"❌ SQLite Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    export_db()