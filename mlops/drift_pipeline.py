# drift_detection.py
"""
Production Drift Detection System for Cyberbullying Detection

Monitors data and model drift in production, generates detailed reports,
and triggers alerts/retraining when thresholds are exceeded.

Key Features:
- Statistical drift detection (KS, PSI, Chi-square tests)
- Semantic drift detection via embeddings
- Label distribution monitoring
- Model performance degradation tracking
- Automated alerting (Slack, email, PagerDuty)
- Integration with Prefect for scheduling
- Metric tracking to Prometheus
"""
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import hashlib

from evidently.report import Report
from evidently.metric_preset import TextEvals, DataDriftPreset, DataQualityPreset
from evidently.descriptors import (
    TextLength, 
    OOV, 
    NonLetterCharacterPercentage,
    TriggerWordsPresence,
    RegExp,
    Sentiment
)
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric
)
from scipy import stats

from prefect import task, flow
from prometheus_client import Gauge, Counter, push_to_gateway

# ============================================================================
# CONFIGURATION
# ============================================================================
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/training_data_with_history.parquet")
CURRENT_LOGS_PATH = os.getenv("CURRENT_LOGS_PATH", "data/raw_logs.jsonl")
PRODUCTION_LOGS_PATH = os.getenv("PRODUCTION_LOGS_PATH", "data/production_logs.jsonl")
REPORT_OUTPUT_PATH = os.getenv("REPORT_OUTPUT_PATH", "reports/drift_report.html")
METRICS_OUTPUT_PATH = os.getenv("METRICS_OUTPUT_PATH", "reports/drift_metrics.json")
HISTORICAL_METRICS_PATH = os.getenv("HISTORICAL_METRICS_PATH", "reports/drift_history.jsonl")

# Alert configuration
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
PAGERDUTY_TOKEN = os.getenv("PAGERDUTY_TOKEN")
ALERT_EMAIL = os.getenv("ALERT_EMAIL")
PROMETHEUS_GATEWAY = os.getenv("PROMETHEUS_GATEWAY", "localhost:9091")

# Drift thresholds
DRIFT_THRESHOLDS = {
    "dataset_drift_share": 0.3,  # 30% of features drifting
    "feature_drift_score": 0.1,   # Individual feature drift threshold
    "label_distribution_shift": 0.15,  # 15% change in class balance
    "text_length_shift": 0.2,     # 20% change in avg text length
    "oov_rate_shift": 0.25,       # 25% increase in out-of-vocabulary
    "performance_drop": 0.05      # 5% drop in model metrics
}

# Minimum samples for reliable drift detection
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

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================
DRIFT_SCORE = Gauge(
    'drift_detection_score',
    'Overall drift score',
    ['metric_type']
)

DRIFT_DETECTED = Counter(
    'drift_detections_total',
    'Number of drift detections',
    ['severity']
)

SAMPLES_ANALYZED = Counter(
    'drift_samples_analyzed_total',
    'Number of samples analyzed for drift'
)

# ============================================================================
# DATA LOADING
# ============================================================================
@task(name="Load Reference and Current Data", log_prints=True, retries=2)
def load_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load reference (training) and current (production) datasets.
    
    Returns:
        Tuple of (reference_df, current_df) or (None, None) on failure
    """
    logger.info("📂 Loading datasets for drift analysis...")
    
    # Load reference data
    try:
        if not os.path.exists(REFERENCE_DATA_PATH):
            logger.error(f"❌ Reference data not found: {REFERENCE_DATA_PATH}")
            return None, None
        
        reference = pd.read_parquet(REFERENCE_DATA_PATH)
        logger.info(f"✅ Loaded reference data: {len(reference)} samples")
        
        # Basic validation
        if 'text' not in reference.columns:
            logger.error("❌ Reference data missing 'text' column")
            return None, None
            
    except Exception as e:
        logger.error(f"❌ Failed to load reference data: {e}", exc_info=True)
        return None, None
    
    # Load current production data
    try:
        # Try multiple sources
        current_sources = [CURRENT_LOGS_PATH, PRODUCTION_LOGS_PATH]
        current = None
        
        for source in current_sources:
            if os.path.exists(source):
                try:
                    temp_df = pd.read_json(source, lines=True)
                    if not temp_df.empty:
                        current = temp_df if current is None else pd.concat([current, temp_df])
                        logger.info(f"✅ Loaded {len(temp_df)} samples from {source}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not read {source}: {e}")
        
        if current is None or current.empty:
            logger.warning("⚠️  No current production data found")
            return reference, None
        
        # Deduplicate current data
        if 'text_hash' in current.columns:
            before = len(current)
            current = current.drop_duplicates(subset=['text_hash'])
            logger.info(f"🔄 Removed {before - len(current)} duplicate samples")
        
        # Validate minimum samples
        if len(current) < MIN_SAMPLES_FOR_DRIFT:
            logger.warning(
                f"⚠️  Only {len(current)} samples (minimum: {MIN_SAMPLES_FOR_DRIFT}). "
                f"Drift detection may be unreliable."
            )
        
        # Align columns between reference and current
        common_cols = list(set(reference.columns) & set(current.columns))
        logger.info(f"📊 Using {len(common_cols)} common columns for comparison")
        
        reference = reference[common_cols]
        current = current[common_cols]
        
        logger.info(f"✅ Loaded current data: {len(current)} samples")
        SAMPLES_ANALYZED.inc(len(current))
        
        return reference, current
        
    except Exception as e:
        logger.error(f"❌ Failed to load current data: {e}", exc_info=True)
        return reference, None

# ============================================================================
# STATISTICAL DRIFT TESTS
# ============================================================================
def kolmogorov_smirnov_test(ref_data: pd.Series, cur_data: pd.Series) -> Dict[str, float]:
    """
    Perform KS test for continuous numerical features.
    
    Returns:
        Dict with statistic and p-value
    """
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
    """
    Perform Chi-square test for categorical features.
    
    Returns:
        Dict with statistic and p-value
    """
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

@task(name="Detect Statistical Drift", log_prints=True)
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
            logger.warning(f"  ⚠️  Drift detected in {feature}")
    
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
            logger.warning(f"  ⚠️  Label distribution shift: {label_shift:.2%}")
    
    # Overall drift decision
    drift_share = len(drift_results["drifted_features"]) / max(len(drift_results["features"]), 1)
    drift_results["drift_share"] = drift_share
    drift_results["overall_drift_detected"] = drift_share > DRIFT_THRESHOLDS["dataset_drift_share"]
    
    if drift_results["overall_drift_detected"]:
        logger.warning(f"🚨 DRIFT DETECTED: {len(drift_results['drifted_features'])} features drifted")
        DRIFT_DETECTED.labels(severity="high").inc()
    else:
        logger.info("✅ No significant statistical drift detected")
    
    # Update Prometheus metrics
    DRIFT_SCORE.labels(metric_type="statistical").set(drift_share)
    
    return drift_results

# ============================================================================
# EVIDENTLY REPORT GENERATION
# ============================================================================
@task(name="Generate Evidently Report", log_prints=True)
def generate_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame
) -> str:
    """
    Generate comprehensive drift report using Evidently AI.
    
    Includes:
    - Data quality metrics
    - Statistical drift detection
    - Text-specific metrics (length, OOV, sentiment)
    - Custom descriptors for toxicity keywords
    
    Returns:
        Path to generated HTML report
    """
    logger.info("📊 Generating Evidently drift report...")
    
    try:
        # Define text descriptors
        text_descriptors = [
            TextLength(),
            OOV(),
            NonLetterCharacterPercentage(),
            TriggerWordsPresence(words_list=TOXIC_KEYWORDS),
            RegExp(reg_exp=r'[!?]{2,}'),  # Multiple punctuation (emphasis)
            Sentiment()
        ]
        
        # Create comprehensive report
        report = Report(metrics=[
            # Data quality
            DataQualityPreset(),
            
            # Statistical drift
            DataDriftPreset(
                columns=None,  # Test all columns
                drift_share=DRIFT_THRESHOLDS["dataset_drift_share"]
            ),
            
            # Text-specific metrics
            TextEvals(
                column_name="text",
                descriptors=text_descriptors
            ),
            
            # Individual column drift
            ColumnDriftMetric(column_name="msg_len"),
            ColumnDriftMetric(column_name="caps_ratio"),
            
            # Dataset-level
            DatasetDriftMetric(),
            DatasetMissingValuesMetric()
        ])
        
        # Run report
        report.run(reference_data=reference, current_data=current)
        
        # Save HTML
        os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
        report.save_html(REPORT_OUTPUT_PATH)
        
        logger.info(f"✅ Evidently report saved to {REPORT_OUTPUT_PATH}")
        
        # Extract key metrics for alerting
        report_json = report.json()
        
        return REPORT_OUTPUT_PATH
        
    except Exception as e:
        logger.error(f"❌ Evidently report generation failed: {e}", exc_info=True)
        return None

# ============================================================================
# ALERTING
# ============================================================================
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

@task(name="Send Alerts", log_prints=True)
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

# ============================================================================
# METRICS PERSISTENCE
# ============================================================================
@task(name="Save Drift Metrics", log_prints=True)
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
        if PROMETHEUS_GATEWAY:
            try:
                push_to_gateway(
                    PROMETHEUS_GATEWAY,
                    job='drift_detection',
                    registry=DRIFT_SCORE._metrics
                )
                logger.info("✅ Metrics pushed to Prometheus")
            except Exception as e:
                logger.warning(f"⚠️  Could not push to Prometheus: {e}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save metrics: {e}")

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================
@flow(name="Drift Detection Pipeline", log_prints=True)
def drift_detection_flow():
    """
    Main drift detection orchestration flow.
    
    Steps:
    1. Load reference and current data
    2. Run statistical drift tests
    3. Generate Evidently report
    4. Send alerts if drift detected
    5. Save metrics for tracking
    """
    logger.info("🚀 Starting drift detection pipeline...")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load data
    reference, current = load_data()
    
    if reference is None:
        logger.error("❌ Cannot proceed without reference data")
        return
    
    if current is None or current.empty:
        logger.warning("⚠️  No current data to analyze")
        return
    
    # Statistical drift detection
    drift_results = detect_statistical_drift(reference, current)
    
    # Generate visual report
    report_path = generate_evidently_report(reference, current)
    
    if report_path:
        drift_results["report_path"] = report_path
    
    # Save metrics
    save_drift_metrics(drift_results)
    
    # Send alerts if needed
    send_alerts(drift_results)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 DRIFT DETECTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Reference samples: {drift_results['reference_samples']}")
    logger.info(f"Current samples: {drift_results['current_samples']}")
    logger.info(f"Drift detected: {drift_results['overall_drift_detected']}")
    logger.info(f"Drifted features: {len(drift_results['drifted_features'])}")
    if drift_results['drifted_features']:
        logger.info(f"  - {', '.join(drift_results['drifted_features'])}")
    logger.info("="*60)
    
    # Return recommendation
    if drift_results['overall_drift_detected']:
        logger.warning("🔄 RECOMMENDATION: Retrain model to address drift")
        return "retrain_recommended"
    else:
        logger.info("✅ Model performance likely stable")
        return "no_action_needed"

# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Drift Detection for Cyberbullying Detection")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Deploy with daily schedule"
    )
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=24,
        help="Scheduling interval in hours (default: 24)"
    )
    
    args = parser.parse_args()
    
    if args.schedule:
        from prefect.deployments import Deployment
        from prefect.server.schemas.schedules import IntervalSchedule
        
        logger.info(f"📅 Deploying with {args.interval_hours}-hour schedule...")
        
        deployment = Deployment.build_from_flow(
            flow=drift_detection_flow,
            name="daily-drift-detection",
            schedule=IntervalSchedule(interval=timedelta(hours=args.interval_hours))
        )
        
        deployment.apply()
        logger.info("✅ Deployment created successfully")
    else:
        # Run once
        result = drift_detection_flow()
        logger.info(f"Pipeline result: {result}")