import pandas as pd
import json
import os
from datetime import datetime

from evidently.report import Report
from evidently.metric_preset import TextEvals, DataDriftPreset
from evidently.descriptors import TextLength, OOV, NonLetterCharacterPercentage

# Configuration
REFERENCE_DATA_PATH = "data/training_data_with_history.parquet"
CURRENT_LOGS_PATH = "data/raw_logs.jsonl"
REPORT_OUTPUT_PATH = "reports/drift_report.html"

def load_data():
    # 1. Load Reference (Training Data)
    reference = pd.read_parquet(REFERENCE_DATA_PATH)
    
    # 2. Load Current (Production Logs)
    if not os.path.exists(CURRENT_LOGS_PATH):
        print("⚠️ No production logs found.")
        return None, None
        
    current = pd.read_json(CURRENT_LOGS_PATH, lines=True)
    
    # Ensure columns match
    # evidently expects the text column to have the same name
    return reference, current

def generate_report(reference, current):
    print("📊 Generating Data Drift Report...")
    
    # We use a preset designed for NLP
    # It calculates stats like: Text Length, Out-of-Vocabulary % (OOV), etc.
    text_report = Report(metrics=[
        DataDriftPreset(
            columns=["text"], 
            embeddings=None, 
            embeddings_drift_method=None,
            drift_share=0.5
        ), 
        TextEvals(column_name="text", descriptors=[
            TextLength(), 
            OOV(), 
            NonLetterCharacterPercentage()
        ])
    ])
    
    text_report.run(reference_data=reference, current_data=current)
    
    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    text_report.save_html(REPORT_OUTPUT_PATH)
    print(f"✅ Report saved to {REPORT_OUTPUT_PATH}")

if __name__ == "__main__":
    ref, cur = load_data()
    if cur is not None and not cur.empty:
        generate_report(ref, cur)