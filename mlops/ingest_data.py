import pandas as pd
import os
import json
import subprocess
from datetime import datetime
from prefect import task, flow
from prefect.server.schemas.schedules import IntervalSchedule

# Configuration
LOGS_PATH = "../data/raw_logs.jsonl"
MASTER_DATA_PATH = "../data/training_data_with_history.parquet"
BACKUP_PATH = "../data/training_data_backup.parquet"

@task(log_prints=True)
def load_and_validate_logs():
    """
    Reads the raw logs.
    Assumes logs have: 'text', 'timestamp', 'user_id', and optionally 'verified_label'
    """
    if not os.path.exists(LOGS_PATH):
        print("⚠️ No new logs found.")
        return None

    # Read JSON Lines
    new_data = pd.read_json(LOGS_PATH, lines=True)
    
    if new_data.empty:
        print("⚠️ Log file is empty.")
        return None

    # FILTER: Only keep logs that have a manual label (Human in the loop)
    # If you want to use the Bot's prediction as the label (Risky!), skip this.
    if 'verified_label' in new_data.columns:
        valid_data = new_data.dropna(subset=['verified_label']).copy()
        valid_data['label'] = valid_data['verified_label'] # Override prediction with truth
    else:
        print("Using model predictions as labels (Pseudo-labeling)")
        valid_data = new_data.rename(columns={'prediction': 'label'})

    print(f"Loaded {len(valid_data)} new valid samples.")
    return valid_data

@task(log_prints=True)
def calculate_features(df):
    # 1. Message-Level Features
    df['msg_len'] = df['text'].astype(str).apply(len)
    df['caps_ratio'] = df['text'].astype(str).apply(lambda x: sum(1 for c in x if c.isupper())/len(x) if len(x)>0 else 0)
    
    # Simple bad word count (You should import BAD_WORDS from a config file)
    bad_words = ["trash", "kill", "die", "ugly", "stupid"] 
    df['slur_count'] = df['text'].astype(str).apply(lambda x: sum(1 for w in bad_words if w in x.lower()))
    
    # 2. Mocking History Features (Since logs usually don't have full 7-day history context)
    # In a real system, you would query the Feature Store here.
    # For this script, we assume the API logged these values, or we fill defaults.
    cols_needed = ['user_bad_ratio_7d', 'user_toxicity_trend', 'channel_toxicity_ratio', 
                   'hours_since_last_msg', 'is_new_to_channel']
    
    for col in cols_needed:
        if col not in df.columns:
            df[col] = 0.0 # Default value for missing context
            
    # Ensure column order matches master
    return df

@task(log_prints=True)
def merge_and_save(new_df):
    if new_df is None: return

    # Load master
    master_df = pd.read_parquet(MASTER_DATA_PATH)
    
    # Align columns
    common_cols = master_df.columns.intersection(new_df.columns)
    new_df_aligned = new_df[common_cols]
    
    # Append
    combined_df = pd.concat([master_df, new_df_aligned], ignore_index=True)
    
    # Deduplicate (prevent adding the same log twice)
    combined_df.drop_duplicates(subset=['text', 'user_id', 'timestamp'], inplace=True)
    
    # Save Backup
    master_df.to_parquet(BACKUP_PATH)
    
    # Save New Master
    combined_df.to_parquet(MASTER_DATA_PATH)
    print(f"✅ Merged. Total size: {len(combined_df)} rows (+{len(new_df_aligned)} new).")
    
    # Clear the logs file (so we don't re-ingest next time)
    # open(LOGS_PATH, 'w').close() 

@task(log_prints=True)
def version_control_data():
    """
    Commits the new dataset to DVC and Git.
    """
    print("📦 Versioning data with DVC...")
    
    # 1. DVC Add
    subprocess.run(["dvc", "add", MASTER_DATA_PATH], check=True)
    
    # 2. DVC Push (Upload to S3)
    subprocess.run(["dvc", "push"], check=True) 
    
    # 3. Git Commit (Track the .dvc file)
    dvc_file = f"{MASTER_DATA_PATH}.dvc"
    timestamp = datetime.now().strftime("%Y-%m-%d")
    subprocess.run(["git", "add", dvc_file], check=True)
    subprocess.run(["git", "commit", "-m", f"Auto-ingest logs: {timestamp}"], check=True)
    
    print("✅ Data versioned and committed to Git.")

@flow(name="Data Ingestion Pipeline")
def data_ingestion_flow():
    new_data = load_and_validate_logs()
    if new_data is not None:
        processed_data = calculate_features(new_data)
        merge_and_save(processed_data)
        version_control_data()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", action="store_true", help="Deploy 2-week schedule")
    args = parser.parse_args()
    
    if args.schedule:
        data_ingestion_flow.serve(
            name="biweekly-data-ingestion",
            interval=1209600 # 2 weeks in seconds
        )
    else:
        data_ingestion_flow()