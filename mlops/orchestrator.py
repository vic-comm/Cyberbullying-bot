import logging
from datetime import datetime
import pandas as pd
import os
from prefect import flow, get_run_logger

# Import your sub-flows
import boto3
from mlops.drift_pipeline import drift_detection_flow
from mlops.ingest_data import data_ingestion_flow
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv("BUCKET_NAME") 
    
# @flow(name="MLOps Master Pipeline", log_prints=True)
def mlops_orchestrator():
    logger.info("🚀 Starting MLOps Master Pipeline...")
    logger.info("🛡️ STEP 2: Checking Data Drift...")
    drift_decision = drift_detection_flow()
    
    drift_severity = drift_decision.get("severity", "none")
    logger.info(f"   Drift Severity: {drift_severity.upper()}")

    # ── STEP 3: INGESTION (Validate & Merge) ──────────────────────
    logger.info("💾 STEP 3: Ingesting Data...")
    ingest_stats = data_ingestion_flow()
    
    ingestion_status = ingest_stats.get("status")

    if ingestion_status == "failed":
        logger.error(" Ingestion Failed (Likely Quality/Adversarial Issue).")
        logger.error("   Action: Check logs. Data was NOT merged.")
        return # STOP HERE - Do not retrain on garbage

    elif ingestion_status == "skipped":
        logger.warning(" Ingestion Skipped (No valid data found).")
        return
    
    logger.info(" STEP 4: Retraining Decision...")
    
    if drift_severity in ["high", "critical"]:
        logger.warning(f" {drift_severity.upper()} Drift detected in valid data.")
        logger.warning("   Triggering Automatic Retraining...")
        
        
        from mlops.pipeline import main_flow
        main_flow(use_dvc=True)
        
    elif drift_severity == "medium":
        logger.info("Medium Drift. Adding to schedule (no immediate retrain).")
        
    else:
        logger.info("✅ Low Drift. Model is stable.")

    logger.info("✅ Pipeline Completed Successfully.")

if __name__ == "__main__":
    mlops_orchestrator()


