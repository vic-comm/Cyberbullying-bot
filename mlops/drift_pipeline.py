import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import subprocess
import os
import asyncio
import logging
import asyncpg
from typing import Dict, Any, List
import mlflow
from dotenv import load_dotenv
import pandas as pd
import boto3
import requests
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset, TextEvals
from evidently.descriptors import (
    TextLength,
    OOVWordsPercentage,
    NonLetterCharacterPercentage,
    TriggerWordsPresent,
    RegExp,
    Sentiment,
)
from evidently import Dataset
from evidently import DataDefinition
from scipy import stats

from prefect import task, flow, get_run_logger
# from prometheus_client import Gauge, Counter, push_to_gateway
load_dotenv()
# CONFIGURATION
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/training_data_with_history.parquet")
CURRENT_LOGS_PATH = os.getenv("CURRENT_LOGS_PATH", "data/raw_logs.jsonl")
REPORT_OUTPUT_PATH = os.getenv("REPORT_OUTPUT_PATH", "reports/drift_report.html")
METRICS_OUTPUT_PATH = os.getenv("METRICS_OUTPUT_PATH", "reports/drift_metrics.json")
HISTORICAL_METRICS_PATH = os.getenv("HISTORICAL_METRICS_PATH", "reports/drift_history.jsonl")
DRIFT_LOOKBACK_HOURS = int(os.getenv("DRIFT_LOOKBACK_HOURS", "24"))  # Check last 24h
PLATFORMS_TO_CHECK = os.getenv("PLATFORMS_TO_CHECK", "discord,slack,whatsapp").split(",")
# Alert configuration
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
PAGERDUTY_TOKEN = os.getenv("PAGERDUTY_TOKEN")
ALERT_EMAIL = os.getenv("ALERT_EMAIL")
# PROMETHEUS_GATEWAY = os.getenv("PROMETHEUS_GATEWAY", "localhost:9091")
BUCKET_NAME = os.getenv("BUCKET_NAME")
S3_REFERENCE_KEY = os.getenv("S3_MASTER_KEY", "data/training_data_with_history.parquet")
S3_LOGS_KEY = os.getenv("S3_LOGS_KEY", "logs/raw_logs.jsonl")
DATABASE_URL = os.getenv("DATABASE_URL") 
# Drift thresholds
DRIFT_THRESHOLDS = {
    "dataset_drift_share": 0.3,  # 30% of features drifting
    "feature_drift_score": 0.1,   # Individual feature drift threshold
    "label_distribution_shift": 0.15,  # 15% change in class balance
    "text_length_shift": 0.2,     # 20% change in avg text length
    "oov_rate_shift": 0.25,       # 25% increase in out-of-vocabulary
    "performance_drop": 0.05      # 5% drop in model metrics
}

MIN_SAMPLES_FOR_DRIFT = 100

# Toxic keywords for monitoring
TOXIC_KEYWORDS = [
    'kill', 'die', 'hurt', 'trash', 'ugly', 'stupid', 
    'hate', 'loser', 'idiot', 'dumb', 'waste'
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/drift_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_from_s3(s3_key: str, local_path: str) -> bool:
    """Helper to safely download from S3"""
    if not BUCKET_NAME:
        return False
    try:
        s3 = boto3.client('s3')
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        logger.info(f"✅ Downloaded s3://{BUCKET_NAME}/{s3_key}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ S3 Download failed for {s3_key}: {e}")
        return False

# @task(name="Load Reference Data", log_prints=True)
def load_data() -> Optional[pd.DataFrame]:
    logger.info("📂 Loading reference data (Training Baseline)...")
    
    # 1. Check Local existence
    if os.path.exists(REFERENCE_DATA_PATH):
        logger.info("   Found local reference file.")
    else:
        # 2. Try DVC Pull (Best for versioning)
        dvc_success = False
        if os.path.exists('.dvc'):
            logger.info("   Attempting DVC pull...")
            try:
                subprocess.run(["dvc", "pull", REFERENCE_DATA_PATH], check=True, capture_output=True)
                dvc_success = True
                logger.info("✅ DVC pull successful")
            except Exception:
                logger.warning("⚠️ DVC pull failed (Common in Docker)")
        
        # 3. Fallback to S3 Direct Download (Best for Fargate)
        if not dvc_success:
            logger.info("   Attempting direct S3 download...")
            if not download_from_s3(S3_REFERENCE_KEY, REFERENCE_DATA_PATH):
                logger.error("❌ Could not load reference data from DVC or S3")
                return None

    try:
        reference = pd.read_parquet(REFERENCE_DATA_PATH)
        logger.info(f"✅ Loaded reference data: {len(reference)} samples")
        
        # Validate
        if 'text' not in reference.columns:
            logger.error("❌ Reference data missing 'text' column")
            return None
            
        return reference
        
    except Exception as e:
        logger.error(f"❌ Failed to read reference parquet: {e}")
        return None

async def fetch_current_logs_from_supabase(
    lookback_hours: int = 24,
    platforms: List[str] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch recent production logs from Supabase for drift detection.    
    """
    if not DATABASE_URL:
        logger.error("❌ DATABASE_URL not set")
        return None
    
    platforms = platforms or PLATFORMS_TO_CHECK
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    
    logger.info(f"📝 Fetching current logs from Supabase...")
    logger.info(f"   Platforms: {platforms}")
    logger.info(f"   Since: {cutoff.isoformat()}")
    
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # Query production logs
        query = """
            SELECT 
                l.message as text,
                l.user_id,
                l.server_id,
                l.platform,
                l.toxicity_score,
                l.severity,
                l.timestamp,
                l.metadata
            FROM logs l
            WHERE l.timestamp > $1
              AND l.platform = ANY($2)
              AND l.toxicity_score IS NOT NULL
              AND LENGTH(l.message) >= 3
              AND LENGTH(l.message) <= 5000
            ORDER BY l.timestamp DESC
        """
        
        rows = await conn.fetch(query, cutoff, platforms)
        await conn.close()
        
        if not rows:
            logger.warning(f"⚠️ No current logs found in last {lookback_hours}h")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in rows])
        
        logger.info(f"✅ Fetched {len(df)} current logs")
        
        # Create label from severity
        df['toxicity_score'] = df['toxicity_score'].astype(float)
        severity_to_label = {
            'SAFE': 0, 'UNCERTAIN': 0,
            'LOW': 1, 'MEDIUM': 1, 'HIGH': 1
        }
        df['label'] = df['severity'].map(severity_to_label)
        
        # Add text hash for deduplication
        df['text_hash'] = df['text'].apply(lambda x: hash(x))
        
        # Deduplicate
        before = len(df)
        df = df.drop_duplicates(subset=['text_hash'])
        if len(df) < before:
            logger.info(f"🔄 Removed {before - len(df)} duplicates")
        
        # Platform stats
        platform_dist = df['platform'].value_counts()
        logger.info(f"   Platform distribution: {platform_dist.to_dict()}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch from Supabase: {e}", exc_info=True)
        return None

def load_current_data() -> Optional[pd.DataFrame]:
    """
    Load current production data from Supabase (sync wrapper).
    """
    logger.info("📝 Loading current production data...")
    
    # Run async fetch in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        df = loop.run_until_complete(
            fetch_current_logs_from_supabase(
                lookback_hours=DRIFT_LOOKBACK_HOURS,
                platforms=PLATFORMS_TO_CHECK
            )
        )
    finally:
        loop.close()
    
    if df is None or df.empty:
        logger.warning("⚠️ No current data available")
        return None
    
    if len(df) < MIN_SAMPLES_FOR_DRIFT:
        logger.warning(
            f"⚠️ Low sample size: {len(df)} (min: {MIN_SAMPLES_FOR_DRIFT}) "
            f"- drift detection may be unreliable"
        )
    
    logger.info(f"✅ Loaded {len(df)} current samples")
    return df

# STATISTICAL DRIFT TESTS
def kolmogorov_smirnov_test(ref_data: pd.Series, cur_data: pd.Series) -> Dict[str, float]:
    try:
        statistic, p_value = stats.ks_2samp(ref_data.dropna(), cur_data.dropna())
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drifted": p_value < 0.05  # 5% significance level
        }
    except Exception as e:
        logger.warning(f"KS test failed: {e}")
        return {"statistic": None, "p_value": None, "drifted": None}

def chi_square_test(ref_data: pd.Series, cur_data: pd.Series) -> Dict[str, float]:
    try:
        # Get value counts
        ref_counts = ref_data.value_counts()
        cur_counts = cur_data.value_counts()
        
        # Align categories
        all_categories = set(ref_counts.index) | set(cur_counts.index)
        ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
        cur_aligned = [cur_counts.get(cat, 0) for cat in all_categories]
        
        # Perform test
        statistic, p_value = stats.chisquare(cur_aligned, ref_aligned)
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drifted": p_value < 0.05
        }
    except Exception as e:
        logger.warning(f"Chi-square test failed: {e}")
        return {"statistic": None, "p_value": None, "drifted": None}

def population_stability_index(ref_data: pd.Series, cur_data: pd.Series, bins: int = 10) -> float:
    """
    Calculate PSI (Population Stability Index) for numerical features.
    
    PSI Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change
    - PSI >= 0.2: Significant change (drift)
    
    Returns:
        PSI score
    """
    try:
        # Create bins based on reference data
        ref_clean = ref_data.dropna()
        cur_clean = cur_data.dropna()
        
        # Define bins
        _, bin_edges = np.histogram(ref_clean, bins=bins)
        
        # Calculate distributions
        ref_dist = np.histogram(ref_clean, bins=bin_edges)[0] / len(ref_clean)
        cur_dist = np.histogram(cur_clean, bins=bin_edges)[0] / len(cur_clean)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        ref_dist = ref_dist + epsilon
        cur_dist = cur_dist + epsilon
        
        # Calculate PSI
        psi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
        
        return float(psi)
        
    except Exception as e:
        logger.warning(f"PSI calculation failed: {e}")
        return None

# @task(name="Detect Statistical Drift", log_prints=True)
def detect_statistical_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame
) -> Dict[str, Any]:
    """
    Perform comprehensive statistical drift detection.
    
    Tests applied:
    - KS test for numerical features
    - Chi-square for categorical features
    - PSI for distribution shifts
    - Label distribution comparison
    
    Returns:
        Dict with drift metrics per feature
    """
    logger.info("🔬 Running statistical drift tests...")
    
    drift_results = {
        "timestamp": datetime.now().isoformat(),
        "reference_samples": len(reference),
        "current_samples": len(current),
        "features": {},
        "overall_drift_detected": False,
        "drifted_features": []
    }
    
    # Numerical features to test
    numerical_features = ['msg_len', 'caps_ratio', 'slur_count', 'word_count']
    
    for feature in numerical_features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        
        logger.info(f"  Testing {feature}...")
        
        # KS Test
        ks_result = kolmogorov_smirnov_test(reference[feature], current[feature])
        
        # PSI
        psi_score = population_stability_index(reference[feature], current[feature])
        
        # Mean shift
        ref_mean = reference[feature].mean()
        cur_mean = current[feature].mean()
        mean_shift = abs(cur_mean - ref_mean) / (ref_mean + 1e-10)
        
        feature_result = {
            "ks_test": ks_result,
            "psi": psi_score,
            "mean_shift": float(mean_shift),
            "ref_mean": float(ref_mean),
            "cur_mean": float(cur_mean),
            "drifted": ks_result.get("drifted") or (psi_score and psi_score > 0.2)
        }
        
        drift_results["features"][feature] = feature_result
        
        if feature_result["drifted"]:
            drift_results["drifted_features"].append(feature)
            logger.warning(f"    Drift detected in {feature}")
    
    # Label distribution drift
    if 'label' in reference.columns and 'label' in current.columns:
        logger.info("  Testing label distribution...")
        
        ref_label_dist = reference['label'].value_counts(normalize=True)
        cur_label_dist = current['label'].value_counts(normalize=True)
        
        # Calculate max difference
        label_shift = abs(cur_label_dist - ref_label_dist).max()
        
        drift_results["label_distribution"] = {
            "shift": float(label_shift),
            "ref_dist": ref_label_dist.to_dict(),
            "cur_dist": cur_label_dist.to_dict(),
            "drifted": label_shift > DRIFT_THRESHOLDS["label_distribution_shift"]
        }
        
        if label_shift > DRIFT_THRESHOLDS["label_distribution_shift"]:
            drift_results["drifted_features"].append("label_distribution")
            logger.warning(f"    Label distribution shift: {label_shift:.2%}")
    
    # Overall drift decision
    drift_share = len(drift_results["drifted_features"]) / max(len(drift_results["features"]), 1)
    drift_results["drift_share"] = drift_share
    drift_results["overall_drift_detected"] = drift_share > DRIFT_THRESHOLDS["dataset_drift_share"]
    
    if drift_results["overall_drift_detected"]:
        logger.warning(f" DRIFT DETECTED: {len(drift_results['drifted_features'])} features drifted")
        # DRIFT_DETECTED.labels(severity="high").inc()
    else:
        logger.info(" No significant statistical drift detected")
    
    # Update Prometheus metrics
    # DRIFT_SCORE.labels(metric_type="statistical").set(drift_share)
    
    return drift_results


# @task(name="Generate Evidently Report", log_prints=True)
def generate_evidently_report(reference: pd.DataFrame, current: pd.DataFrame) -> Dict[str, Any]:

    logger.info("📊 Generating Evidently drift report...")
    
    try:
        schema = DataDefinition(text_columns=["text"])
        ref_ds = Dataset.from_pandas(reference, data_definition=schema)
        cur_ds = Dataset.from_pandas(current, data_definition=schema)
        
        text_descriptors = [
            TextLength(column_name='text'),
            OOVWordsPercentage(column_name="text"),
            NonLetterCharacterPercentage(column_name="text"),
            TriggerWordsPresent(column_name="text", words_list=TOXIC_KEYWORDS),
            RegExp(column_name="text", reg_exp=r"[!?]{2,}"),
            Sentiment(column_name="text"),
        ]
        
        report = Report(metrics=[
            DataSummaryPreset(include_tests=True),
            DataDriftPreset(drift_share=DRIFT_THRESHOLDS["dataset_drift_share"]),
            TextEvals(descriptors=text_descriptors),
        ], include_tests=True)
        
        report_snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)
        
        os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
        report_snapshot.save_html(REPORT_OUTPUT_PATH)
        logger.info(f"✅ HTML report saved: {REPORT_OUTPUT_PATH}")
        
        report_dict = report_snapshot.dict()
        
        drifted_features = []
        drift_share = 0.0
        dataset_drift_detected = False
        
        metrics_list = report_dict.get("metrics", [])
        
        for metric_entry in metrics_list:
            metric_type = metric_entry.get("metric", "")
            result = metric_entry.get("result", {})
            
            if "DataDriftTable" in metric_type or "drift_by_columns" in result:
                drift_by_columns = result.get("drift_by_columns", {})
                
                for col_name, col_stats in drift_by_columns.items():
                    if col_stats.get("drift_detected", False):
                        drifted_features.append(col_name)
                
                drift_share = result.get("share_of_drifted_columns", 0.0)
                dataset_drift_detected = result.get("dataset_drift", False)
                
                break 
        
        if drift_share == 0.0 and not drifted_features:
            for metric_entry in metrics_list:
                result = metric_entry.get("result", {})
                
                if "number_of_drifted_columns" in result:
                    total_cols = result.get("number_of_columns", 1)
                    drifted_cols = result.get("number_of_drifted_columns", 0)
                    drift_share = drifted_cols / total_cols if total_cols > 0 else 0.0
                    dataset_drift_detected = result.get("dataset_drift", False)
                    
                    if "drift_by_columns" in result:
                        for col, stats in result["drift_by_columns"].items():
                            if stats.get("drift_detected"):
                                drifted_features.append(col)
        
        result = {
            "drift_share": float(drift_share),
            "drifted_features": drifted_features,
            "drifted_count": len(drifted_features),
            "dataset_drift_detected": bool(dataset_drift_detected or 
                                          drift_share >= DRIFT_THRESHOLDS["dataset_drift_share"]),
            "report_path": REPORT_OUTPUT_PATH,
            "timestamp": datetime.now().isoformat(),
        }
        
        logger.info(
            f" Evidently drift complete | "
            f"share={drift_share:.2%} | "
            f"drifted={len(drifted_features)} features"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌Evidently report failed: {e}", exc_info=True)
        return {
            "drift_share": 0.0,
            "drifted_features": [],
            "drifted_count": 0,
            "dataset_drift_detected": False,
            "report_path": None,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }

# @task(name="Make Drift Decision", log_prints=True)
def make_drift_decision(
    statistical_results: Dict[str, Any],
    evidently_results: Dict[str, Any],
) -> Dict[str, Any]:

    logger.info("🎯 Making drift decision based on combined analysis...")

    SEVERITY_LEVELS = {
        "none": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }

    SEVERITY_ACTIONS = {
        "none": "no_action_needed",
        "low": "continue_monitoring",
        "medium": "monitor_closely",
        "high": "retrain_recommended",
        "critical": "immediate_retrain",
    }

    stat_features = set(statistical_results.get("drifted_features", []))
    evid_features = set(evidently_results.get("drifted_features", []))
    all_features = stat_features | evid_features

    stat_share = statistical_results.get("drift_share", 0.0)
    evid_share = evidently_results.get("drift_share", 0.0)

    combined_share = max(stat_share, evid_share)

    feature_pressure = len(all_features)

    if combined_share >= 0.5:
        severity = "critical"
    elif combined_share >= 0.3:
        severity = "high"
    elif combined_share >= 0.15:
        severity = "medium"
    elif combined_share >= 0.05:
        severity = "low"
    else:
        severity = "none"

    severity_score = SEVERITY_LEVELS[severity]

    if feature_pressure >= 10:
        severity_score = max(severity_score, SEVERITY_LEVELS["high"])
    elif feature_pressure >= 5:
        severity_score = max(severity_score, SEVERITY_LEVELS["medium"])

    label_drift_detected = (
        statistical_results
        .get("label_distribution", {})
        .get("drifted", False)
    )

    if label_drift_detected:
        severity_score = max(severity_score, SEVERITY_LEVELS["high"])

    severity = next(
        k for k, v in SEVERITY_LEVELS.items() if v == severity_score
    )
    action = SEVERITY_ACTIONS[severity]

    drift_detected = combined_share >= DRIFT_THRESHOLDS["dataset_drift_share"]

    decision = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": drift_detected,
        "severity": severity,
        "action": action,
        "metrics": {
            "combined_drift_share": combined_share,
            "statistical_drift_share": stat_share,
            "evidently_drift_share": evid_share,
            "total_drifted_features": feature_pressure,
            "label_drift_detected": label_drift_detected,
        },
        "drifted_features": {
            "all": list(all_features),
            "statistical_only": list(stat_features - evid_features),
            "evidently_only": list(evid_features - stat_features),
            "both": list(stat_features & evid_features),
        },
    }

    logger.info("📊 Drift Decision Summary")
    logger.info(f"  Severity: {severity.upper()}")
    logger.info(f"  Action: {action}")
    logger.info(f"  Combined drift share: {combined_share:.1%}")
    logger.info(f"  Drifted features: {feature_pressure}")

    # DRIFT_SCORE.labels(metric_type="combined").set(combined_share)

    # if drift_detected:
    #     DRIFT_DETECTED.labels(severity=severity).inc()

    return decision

# ALERTING
def send_slack_alert(drift_results: Dict[str, Any]):
    """Send drift alert to Slack"""
    if not SLACK_WEBHOOK_URL:
        logger.info("⚠️  Slack webhook not configured, skipping alert")
        return
    
    try:
        import requests
        
        message = {
            "text": "🚨 *Data Drift Detected*",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🚨 Data Drift Alert"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Drift Share:* {drift_results['drift_share']:.1%}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Drifted Features:* {len(drift_results['drifted_features'])}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Features:* {', '.join(drift_results['drifted_features'])}"
                    }
                }
            ]
        }
        
        response = requests.post(SLACK_WEBHOOK_URL, json=message)
        response.raise_for_status()
        
        logger.info("✅ Slack alert sent")
        
    except Exception as e:
        logger.error(f"❌ Failed to send Slack alert: {e}")

# @task(name="Send Alerts", log_prints=True)
def send_alerts(drift_results: Dict[str, Any]):
    """
    Send alerts via configured channels if drift detected.
    """
    if not drift_results.get("overall_drift_detected"):
        logger.info("No alerts needed - no drift detected")
        return
    
    logger.info("📢 Sending drift alerts...")
    
    # Slack
    send_slack_alert(drift_results)
    
    # Email (placeholder)
    if ALERT_EMAIL:
        logger.info(f"📧 Email alert would be sent to {ALERT_EMAIL}")
        # Implement email sending here
    
    # PagerDuty (placeholder)
    if PAGERDUTY_TOKEN:
        logger.info("📟 PagerDuty alert would be triggered")
        # Implement PagerDuty integration here

# METRICS PERSISTENCE
# @task(name="Save Drift Metrics", log_prints=True)
def save_drift_metrics(drift_results: Dict[str, Any]):
    """
    Save drift metrics to JSON and historical log.
    """
    try:
        # Save latest metrics
        os.makedirs(os.path.dirname(METRICS_OUTPUT_PATH), exist_ok=True)
        with open(METRICS_OUTPUT_PATH, 'w') as f:
            json.dump(drift_results, f, indent=2)
        
        logger.info(f"✅ Metrics saved to {METRICS_OUTPUT_PATH}")
        
        # Append to historical log
        os.makedirs(os.path.dirname(HISTORICAL_METRICS_PATH), exist_ok=True)
        with open(HISTORICAL_METRICS_PATH, 'a') as f:
            f.write(json.dumps(drift_results) + '\n')
        
        # Push to Prometheus if configured
        # if PROMETHEUS_GATEWAY:
        #     try:
        #         push_to_gateway(
        #             PROMETHEUS_GATEWAY,
        #             job='drift_detection',
        #             registry=DRIFT_SCORE._metrics
        #         )
        #         logger.info("✅ Metrics pushed to Prometheus")
        #     except Exception as e:
        #         logger.warning(f"⚠️  Could not push to Prometheus: {e}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save metrics: {e}")


def drift_detection_flow(current_data: pd.Dataframe = None):
    start_time = datetime.now()
    logger = get_run_logger()
    logger.info("🚀 Starting drift detection pipeline...")
    logger.info(f"Timestamp: {start_time.isoformat()}")
    logger.info("=" * 60)

    mlflow_run = None

    try:
        dagshub_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(dagshub_uri)
        logger.info(f"📊 MLflow tracking: {dagshub_uri}")
        
        experiment_name = "Drift_Monitoring_v3"
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        run_name = f"drift_check_{start_time.strftime('%Y%m%d_%H%M%S')}"
        mlflow_run = mlflow.start_run(run_name=run_name)
        logger.info(f"🔬 MLflow run: {mlflow_run.info.run_id}")

        # STEP 1: Load Data
        logger.info("\n📂 STEP 1: Loading datasets...")
        reference = load_data()
        if not current_data:
            current = load_current_data()
        else:
            current = current_data

        common_cols = list(set(reference.columns) & set(current.columns))
        missing_in_current = set(reference.columns) - set(current.columns)
        missing_in_reference = set(current.columns) - set(reference.columns)
        
        if missing_in_current:
            logger.warning(f"     Columns in reference but not current: {list(missing_in_current)[:5]}...")
        if missing_in_reference:
            logger.info(f"     New columns in current: {list(missing_in_reference)[:5]}...")
        
        logger.info(f"   Using {len(common_cols)} common columns for comparison")
        
        # Filter to common columns
        reference = reference[common_cols]
        current = current[common_cols]

        if reference is None:
            logger.error("❌ No reference data available")
            mlflow.log_param("status", "failed")
            mlflow.log_param("reason", "no_reference_data")
            mlflow.end_run(status="FAILED")
            return {"status": "failed", "reason": "no_reference_data"}

        if current is None or current.empty:
            logger.warning("⚠️ No current data to analyze")
            mlflow.log_param("status", "skipped")
            mlflow.log_param("reason", "no_current_data")
            mlflow.end_run(status="FINISHED")
            return {"status": "skipped", "reason": "no_current_data"}

        logger.info(f"✅ Loaded {len(reference)} reference + {len(current)} current samples")
        
        # Log data metrics
        mlflow.log_param("reference_samples", len(reference))
        mlflow.log_param("current_samples", len(current))
        mlflow.log_param("pipeline_version", "1.0.0")

        # STEP 2: Statistical Drift Detection
        statistical_results = detect_statistical_drift(reference, current)
        
        logger.info(
            f"Statistical: {statistical_results.get('drift_share', 0):.1%} drift share, "
            f"{len(statistical_results.get('drifted_features', []))} features"
        )
        
        # Log statistical metrics
        mlflow.log_metric("statistical_drift_share", statistical_results.get("drift_share", 0.0))
        mlflow.log_metric("statistical_drifted_count", len(statistical_results.get("drifted_features", [])))
        
        # Log per-feature drift scores (PSI, KS statistic)
        for feature, stats in statistical_results.get("features", {}).items():
            if stats.get("psi") is not None:
                mlflow.log_metric(f"psi_{feature}", stats["psi"])
            if stats.get("mean_shift") is not None:
                mlflow.log_metric(f"mean_shift_{feature}", stats["mean_shift"])

        # STEP 3: Evidently Drift Detection
        evidently_results = generate_evidently_report(reference, current)

        # Graceful degradation
        if evidently_results.get("error"):
            logger.warning(f"⚠️ Evidently unavailable: {evidently_results['error']} — using stats only")
            mlflow.log_param("evidently_status", "failed")
            mlflow.log_param("evidently_error", str(evidently_results.get("error"))[:250])
            
            evidently_results = {
                "drift_share": 0.0,
                "drifted_features": [],
                "drifted_count": 0,
                "dataset_drift_detected": False,
                "report_path": None,
                "status": "unavailable",
            }
        else:
            logger.info(
                f"Evidently: {evidently_results.get('drift_share', 0):.1%} drift share, "
                f"{evidently_results.get('drifted_count', 0)} features"
            )
            
            # Log Evidently metrics
            mlflow.log_param("evidently_status", "success")
            mlflow.log_metric("evidently_drift_share", evidently_results.get("drift_share", 0.0))
            mlflow.log_metric("evidently_drifted_count", evidently_results.get("drifted_count", 0))

        # STEP 3.5: Upload Evidently Report to MLflow/Dagshub
        report_path = evidently_results.get("report_path")
        if report_path and Path(report_path).exists() and not evidently_results.get("error"):
            try:
                # Log the HTML report as artifact
                mlflow.log_artifact(report_path, artifact_path="drift_reports")
                logger.info(f"✅ Evidently report uploaded to MLflow: {report_path}")
                
                # Log metrics output as well (if exists)
                metrics_path = os.getenv("METRICS_OUTPUT_PATH", "reports/drift_metrics.json")
                if Path(metrics_path).exists():
                    mlflow.log_artifact(metrics_path, artifact_path="metrics")
                    logger.info(f"✅ Metrics JSON uploaded: {metrics_path}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to upload artifacts to MLflow: {e}")
        elif report_path:
            logger.warning(f"⚠️ Report path exists but file not found: {report_path}")

        # STEP 4: Combined Drift Decision
        drift_decision = make_drift_decision(
            statistical_results=statistical_results,
            evidently_results=evidently_results,
        )
        
        # Log decision metrics
        mlflow.log_param("drift_detected", drift_decision.get("drift_detected", False))
        mlflow.log_param("severity", drift_decision.get("severity", "none"))
        mlflow.log_param("action", drift_decision.get("action", "no_action_needed"))
        
        mlflow.log_metric("combined_drift_share", drift_decision["metrics"]["combined_drift_share"])
        mlflow.log_metric("total_drifted_features", drift_decision["metrics"]["total_drifted_features"])
        
        # Log drifted feature names as tags (for filtering in UI)
        drifted_features = drift_decision.get("drifted_features", {}).get("all", [])
        if drifted_features:
            mlflow.set_tag("drifted_features", ",".join(drifted_features[:10]))  # Limit to 10
            mlflow.log_dict(drift_decision["drifted_features"], "drifted_features.json")

        # STEP 5: Persist Metrics
        logger.info("💾 Saving drift metrics...")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        complete_results = {
            "pipeline_metadata": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "pipeline_version": "1.0.0",
                "mlflow_run_id": mlflow_run.info.run_id if mlflow_run else None,
                "mlflow_experiment": experiment_name,
            },
            "decision": drift_decision,
            "statistical": statistical_results,
            "evidently": evidently_results,
        }

        save_drift_metrics(complete_results)
        
        # Log pipeline duration
        mlflow.log_metric("pipeline_duration_seconds", duration)

        # STEP 6: Alerts (severity-gated)
        if drift_decision["severity"] in {"high", "critical"}:
            logger.info(f"Triggering alerts (severity: {drift_decision['severity']})")
            mlflow.set_tag("alert_triggered", "true")
            send_alerts(drift_decision)
        else:
            logger.info(f" No alerts needed (severity: {drift_decision['severity']})")
            mlflow.set_tag("alert_triggered", "false")

        # Summary Report
        logger.info("\n" + "=" * 60)
        logger.info("📊 DRIFT DETECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Reference samples: {len(reference):,}")
        logger.info(f"Current samples: {len(current):,}")
        logger.info(f"Drift detected: {'YES' if drift_decision['drift_detected'] else 'NO'}")
        logger.info(f"Severity: {drift_decision['severity'].upper()}")
        logger.info(f"Action: {drift_decision['action']}")
        logger.info(f"Drifted features: {drift_decision['metrics']['total_drifted_features']}")
        
        if drifted_features:
            logger.info(f"  → {', '.join(drifted_features)}")
        
        if evidently_results.get("report_path"):
            logger.info(f"Evidently report: {evidently_results['report_path']}")
        
        if mlflow_run:
            logger.info(f"MLflow run ID: {mlflow_run.info.run_id}")
        
        logger.info(f"Duration: {duration:.2f}s")
        logger.info("=" * 60)
        
        # Mark MLflow run as successful
        mlflow.end_run(status="FINISHED")

        return drift_decision

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        
        # Log error to MLflow
        if mlflow_run:
            mlflow.log_param("error", str(e)[:250])
            mlflow.set_tag("status", "failed")
            mlflow.end_run(status="FAILED")
        
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": start_time.isoformat(),
            "duration_seconds": (datetime.now() - start_time).total_seconds()
        }


if __name__ == "__main__":
    logger.info("🚀 Starting Drift Detection Job...")
    
    try:
        status = drift_detection_flow()
        logger.info(f"✅ Drift Check Complete. Status: {status}")
        
    except Exception as e:
        logger.error(f"❌ Drift Check Failed: {e}")
        exit(1) 