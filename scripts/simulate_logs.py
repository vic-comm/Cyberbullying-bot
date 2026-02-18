import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path

PARQUET_PATH = "data/training_data_with_history.parquet"
LOG_PATH = "data/raw_logs.jsonl"

def simulate_logs():
    if not os.path.exists(PARQUET_PATH):
        print(f"❌ Error: Could not find {PARQUET_PATH}")
        return

    print(f"📂 Reading from {PARQUET_PATH}...")
    
    try:
        df = pd.read_parquet(PARQUET_PATH)
    except Exception as e:
        print(f"❌ Failed to read parquet file: {e}")
        return

    sample_size = min(20, len(df))
    sample = df.sample(n=sample_size, random_state=42)

    log_df = sample[[
        "text",
        "user_id",
        "timestamp"
    ]].copy()

    if "label" in sample.columns:
        log_df["verified_label"] = sample["label"]
    else:
        import numpy as np
        log_df["prediction"] = np.random.randint(0, 2, size=len(log_df))

    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"]).dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    
    log_df["text_hash"] = log_df["text"].apply(lambda x: str(hash(x + datetime.now().isoformat())))

    Path("data").mkdir(exist_ok=True)

    print(f"📝 Writing {len(log_df)} records to {LOG_PATH}...")
    with open(LOG_PATH, "w") as f:
        for _, row in log_df.iterrows():
            record = row.to_dict()
            f.write(json.dumps(record) + "\n")

    print(f"✅ Success! Created {LOG_PATH} with {len(log_df)} simulated log entries.")
    print("   You can now run your pipeline: python mlops/ingest_data.py")

if __name__ == "__main__":
    simulate_logs()