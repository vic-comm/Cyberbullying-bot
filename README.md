# README.md - Created for Cyberbullying Bot Project
#### **1. Architecture Diagram**

Use draw.io or Excalidraw:
```
┌─────────────────────────────────────────────────────────┐
│                   USER INTERFACES                        │
│  [Discord]  [Slack]  [WhatsApp]  [Telegram (WIP)]      │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   Bot Service       │
          │   - Event handling  │
          │   - Moderation      │
          │   - Commands        │
          └──────────┬──────────┘
                     │ REST API
          ┌──────────▼──────────┐
          │  Inference API      │
          │  - FastAPI          │
          │  - Model serving    │
          │  - LIME explainer   │
          └─┬─────────┬─────────┘
            │         │
    ┌───────▼──┐  ┌──▼────────┐
    │ Supabase │  │   Redis   │
    │ (Logs)   │  │ (Features)│
    └───┬──────┘  └───────────┘
        │
    ┌───▼──────────────────┐
    │   MLOps Pipeline     │
    │  - Data ingestion    │
    │  - Drift detection   │
    │  - Feature sync      │
    └──────────┬───────────┘
               │
    ┌──────────▼──────────┐
    │  Training Script    │
    │  - DistilBERT       │
    │  - XGBoost          │
    │  - MLflow tracking  │
    └─────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
└──────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │ Supabase PostgreSQL │
                    │   (logs table)      │
                    │  - Single source    │
                    │    of truth         │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
    ┌─────────────────────┐       ┌──────────────────────┐
    │ TRAINING PATH       │       │ INFERENCE PATH       │
    └─────────────────────┘       └──────────────────────┘
                │                             │
                ▼                             ▼
    Fetch individual messages     Calculate user statistics
                │                             │
                ▼                             ▼
    Calculate message features    GROUP BY user_id
    (msg_len, caps_ratio)                   │
                │                             ▼
                ▼                   ┌──────────────────────┐
    ┌─────────────────────┐         │      Redis           │
    │  Parquet (S3)       │         │  (Feature Store)     │
    │  - 10M messages     │         │  - 10K users         │
    │  - One row per msg  │         │  - One key per user  │
    └──────────┬──────────┘         └──────────┬───────────┘
               │                               │
               ▼                               ▼
    ┌─────────────────────┐         ┌──────────────────────┐
    │ Training Script     │         │  Inference API       │
    │ (Monthly)           │         │  (Real-time)         │
    └─────────────────────┘         └──────────────────────┘

    NEVER CONNECTED ←────────────────────→ NEVER CONNECTED


    # How does the new model get deployed?
# - Copy to api_service/models/?
# - Update MLflow model registry?
# - Restart inference API?
```

---

## FINAL ARCHITECTURE
```
┌──────────────────────────────────────────────────────────┐
│              ORCHESTRATOR (Daily 3 AM)                   │
└──────────────────────────────────────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    ┌────▼─────┐  ┌─────▼──────┐  ┌───▼────────┐
    │ Pre-check│  │   Drift    │  │ Ingestion  │
    │ (100+    │  │ Detection  │  │  Pipeline  │
    │  logs?)  │  │            │  │            │
    └────┬─────┘  └─────┬──────┘  └───┬────────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
              ┌─────────▼──────────┐
              │  High/Critical     │
              │  Drift Detected?   │
              └─────────┬──────────┘
                        │
                  ┌─────▼─────┐
                  │ Retrain   │
                  │ (Optional)│
                  └───────────┘