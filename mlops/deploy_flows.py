from prefect import deploy
from mlops.orchestrator import mlops_orchestrator
from mlops.ingest_data import data_ingestion_flow
from mlops.drift_pipeline import drift_detection_flow
from mlops.pipeline import main_flow
from datetime import timedelta
WORK_POOL_NAME = "bully-pool"

print(" Deploying Flows to Prefect Cloud...")

deploy(
    mlops_orchestrator.to_deployment(
        name='production-orchestrator',
        work_pool_name=WORK_POOL_NAME,
        interval=timedelta(weeks=1),
        parameters={'lookback_hours': 168},
        tags=["prod", "orchestrator"],
        description="Main MLOps loop: Sync -> Drift -> Retrain?"
    ),

    data_ingestion_flow.to_deployment(
        name='manual-ingestion',
        work_pool_name=WORK_POOL_NAME,
        tags=["maintenance"],
        description="Manually force data ingestion"
    ),

    main_flow.to_deployment(
        name="manual-retraining",
        work_pool_name=WORK_POOL_NAME,
        tags=["maintenance"],
        interval=timedelta(weeks=2),
        description="Manually force full model retraining"
    ),

    drift_detection_flow.to_deployment(
        name="manual-drift-check",
        work_pool_name=WORK_POOL_NAME,
        tags=["monitoring"],
        description="Manually trigger a drift report generation"
    )
)
