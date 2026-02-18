# 🛡️ AntiBully Bot

**AI-Powered Multi-Platform Content Moderation with Explainable Decisions**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![Discord.py](https://img.shields.io/badge/discord.py-2.3-blue.svg)](https://discordpy.readthedocs.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://postgresql.org)
[![Redis](https://img.shields.io/badge/Redis-7+-red.svg)](https://redis.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.8-blue.svg)](https://mlflow.org)

> **Production-grade moderation bot** using hybrid ML (DistilBERT + XGBoost) with LIME explainability, admin-reviewed feedback loop, and automated drift detection. Built for Discord, Slack, and WhatsApp.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Data Flow](#-data-flow)
- [ML Pipeline](#-ml-pipeline)
- [Feedback Loop](#-feedback-loop)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [MLOps](#-mlops)
- [Contributing](#-contributing)

---

## 🎯 Overview

AntiBully Bot is an intelligent content moderation system that combines:
- **Hybrid ML Model**: DistilBERT (text embeddings) + XGBoost (user context features)
- **Explainable AI**: LIME-generated explanations for every decision
- **Admin Feedback Loop**: Human-in-the-loop corrections improve model accuracy
- **Multi-Platform**: Single ML backend serves Discord, Slack, WhatsApp bots
- **Production MLOps**: Automated drift detection, retraining, and deployment

### Why This Exists

Traditional moderation tools either:
1. Use simple keyword filters (easy to bypass)
2. Use black-box AI (no transparency)
3. Can't learn from mistakes (frozen models)

AntiBully solves all three by combining state-of-the-art NLP, explainable AI, and continuous learning.

---

## ✨ Key Features

### 🤖 Intelligent Moderation
- **Context-Aware**: Uses user history (violation rate, tenure) + channel toxicity
- **Multi-Level Severity**: LOW/MEDIUM/HIGH classification with configurable actions
- **Strike System**: Graduated penalties (warn → timeout → kick → ban)
- **Configurable**: Per-server thresholds, actions, and message templates

### 🔍 Explainable AI
- **LIME Integration**: Shows which words contributed to toxicity score
- **User Dashboard**: Users can see why they were flagged via `!explain` command
- **Admin Dashboard**: Admins review uncertain cases with full context
- **Dispute System**: Users dispute → admin reviews → model learns

### 📊 Production MLOps
- **Drift Detection**: Evidently monitors data/model drift weekly
- **Automated Retraining**: Triggers on high drift or monthly schedule
- **Feature Store**: Redis caches user features for <5ms inference
- **Versioning**: DVC tracks data, MLflow tracks models
- **CI/CD Ready**: Docker containers, Railway.app deployment

### 🔐 Security & Privacy
- **Double-Gated Feedback**: User disputes require admin approval
- **Spam Protection**: Detects coordinated attacks, flooding, repetition
- **Protected Patterns**: Slurs never overridden by user feedback
- **Audit Trail**: Every action logged with timestamps and admin IDs

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                             │
│   Discord Bot  │  Slack Bot  │  WhatsApp Bot  │  Telegram (WIP)   │
└────────────┬───────────────────────────────────────────────────────┘
             │
             │ WebSocket/REST
             ▼
┌────────────────────────────────────────────────────────────────────┐
│                       BOT SERVICE LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Moderation  │  │    Admin     │  │   Feedback   │           │
│  │  (on_message)│  │   Commands   │  │  (!explain)  │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                  │                     │
└─────────┼─────────────────┼──────────────────┼─────────────────────┘
          │                 │                  │
          │ POST /predict   │ GET /config      │ POST /explain
          ▼                 ▼                  ▼
┌────────────────────────────────────────────────────────────────────┐
│                      INFERENCE API (FastAPI)                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │ Toxicity Detector│  │ LIME Explainer   │  │ Feature Enricher│ │
│  │ DistilBERT+XGB  │  │ (word importance)│  │ (Redis lookup)  │ │
│  └──────┬───────────┘  └──────────────────┘  └────────┬────────┘ │
│         │                                               │          │
└─────────┼───────────────────────────────────────────────┼──────────┘
          │                                               │
          │ log_event()                                   │ get_features()
          ▼                                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                  │
│  ┌─────────────────────────────┐  ┌──────────────────────────────┐│
│  │   Supabase PostgreSQL       │  │      Redis (Feature Store)   ││
│  │  ├─ logs                     │  │  ├─ user_toxicity:prod:{id} ││
│  │  ├─ server_configs           │  │  ├─ channel_stats:{id}      ││
│  │  ├─ server_user_violations   │  │  └─ (5ms lookups)           ││
│  │  ├─ feedback (disputes)      │  └──────────────────────────────┘│
│  │  └─ admin_review_queue       │                                  │
│  └─────────────────────────────┘                                  │
└────────────────────────────────────────────────────────────────────┘
          │                                               │
          │ Nightly Sync                                  │ Sync Features
          ▼                                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                       MLOPS PIPELINE                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Data Ingest  │→ │Drift Detector│→ │  Retraining  │           │
│  │ (Stratified) │  │ (Evidently)  │  │  (Monthly)   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│         │                  │                  │                   │
│         ▼                  ▼                  ▼                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ S3 (Parquet) │  │ MLflow Track │  │ DVC Versioning│           │
│  │ Training Data│  │ Experiments  │  │ Data Lineage  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Detailed Component Architecture

### 1. Bot Service (Discord/Slack/WhatsApp)

```
bot_service/
├── bot.py                    # Main entry point
├── config.py                 # Environment variables
├── database.py               # Supabase client
└── cogs/
    ├── moderation.py         # Message filtering
    │   ├── on_message()      # Event listener
    │   ├── execute_action()  # Delete/timeout/kick/ban
    │   └── explain_command() # !explain with feedback buttons
    ├── admin.py              # Configuration
    │   ├── /config           # Interactive setup menu
    │   ├── /pardon           # Reset user strikes
    │   └── /strikes          # View violation history
    └── admin_review.py       # Feedback dashboard (NEW)
        ├── /review           # Review queue (next/by_user/stats/list)
        └── /review_user      # Bulk operations per user
```

**Key Flows:**

#### Message Moderation Flow
```
User sends message
    ↓
on_message() triggered
    ↓
POST /predict (text + user_id + channel_id)
    ↓
API returns: {is_toxic, confidence, severity}
    ↓
if is_toxic:
    ├─ execute_action() based on severity
    ├─ log_event() to Supabase
    └─ send strike warning
else:
    └─ allow message
```

#### User Dispute Flow
```
User types: !explain
    ↓
Fetch latest violation from logs
    ↓
if no explanation:
    ├─ POST /explain (generates LIME)
    └─ Store in logs.explanation
    ↓
Send DM with:
    ├─ LIME visualization (word importance)
    └─ Buttons: [✅ Correct] [❌ Wrong]
    ↓
if user clicks ❌:
    ├─ Open modal: "Why was this wrong?"
    ├─ User types reason
    └─ record_user_dispute() → feedback table
```

---

### 2. Inference API (FastAPI)

```
api_service/
├── app.py                    # FastAPI server
├── models/
│   └── baked_model/          # Serialized model (MLflow)
├── services/
│   ├── toxicity_detector.py  # Main prediction
│   └── explainer.py          # LIME integration
└── utils/
    └── features.py           # Text feature extraction
```

**Endpoints:**

| Endpoint | Method | Purpose | Latency |
|----------|--------|---------|---------|
| `/predict` | POST | Toxicity classification | 50-100ms |
| `/explain` | POST | LIME word importance | 2-4s |
| `/health` | GET | Service status | <10ms |
| `/feedback` | POST | Record user dispute | <50ms |

**Model Architecture:**

```
Input: "you are trash" + user_id + channel_id
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: TEXT FEATURES                                  │
├─────────────────────────────────────────────────────────┤
│ DistilBERT Embeddings (768 dims)                        │
│ + Static Features:                                      │
│   ├─ msg_len, caps_ratio, slur_count                   │
│   ├─ personal_pronoun_count, question_count            │
│   └─ char_repetition, exclamation_count                │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ├─ (768 + 15 features)
                    ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 2: USER CONTEXT ENRICHMENT (Redis)               │
├─────────────────────────────────────────────────────────┤
│ Fetch from Redis:                                       │
│   ├─ user_bad_ratio_7d (% toxic messages)              │
│   ├─ violation_count_7d                                 │
│   ├─ channel_toxicity_ratio                             │
│   ├─ hours_since_last_msg                               │
│   └─ is_new_to_channel                                  │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ├─ (768 + 15 + 5 = 788 features)
                    ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 3: XGBoost Classifier                             │
├─────────────────────────────────────────────────────────┤
│ Input: 788 features                                     │
│ Output: P(toxic) ∈ [0, 1]                               │
│ Thresholds:                                             │
│   ├─ < 0.3 → SAFE                                       │
│   ├─ 0.3-0.5 → LOW                                      │
│   ├─ 0.5-0.7 → MEDIUM                                   │
│   └─ > 0.7 → HIGH                                       │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
Output: {
  "is_toxic": true,
  "confidence": 0.85,
  "severity": "HIGH",
  "features_used": {...}
}
```

---

### 3. Data Layer

#### Supabase PostgreSQL Schema

```sql
-- Core moderation logs
CREATE TABLE logs (
    id              SERIAL PRIMARY KEY,
    user_id         TEXT NOT NULL,
    server_id       TEXT NOT NULL,
    platform        TEXT NOT NULL,
    message         TEXT NOT NULL,
    toxicity_score  REAL,
    severity        TEXT,
    action_taken    TEXT,
    timestamp       TIMESTAMP DEFAULT NOW(),
    explanation     JSONB,          -- LIME output
    metadata        JSONB
);

-- Server configurations (per-guild settings)
CREATE TABLE server_configs (
    server_id                TEXT PRIMARY KEY,
    platform                 TEXT NOT NULL,
    config                   JSONB NOT NULL,  -- All settings
    updated_at               TIMESTAMP
);

-- User violations (strike tracking)
CREATE TABLE server_user_violations (
    server_id                TEXT NOT NULL,
    user_id                  TEXT NOT NULL,
    platform                 TEXT NOT NULL,
    violation_count          INTEGER DEFAULT 0,
    last_violation_time      TIMESTAMP,
    first_violation_time     TIMESTAMP,
    pardoned                 BOOLEAN DEFAULT FALSE,
    pardoned_at              TIMESTAMP,
    pardoned_by              TEXT,
    pardon_reason            TEXT,
    PRIMARY KEY (server_id, user_id, platform)
);

-- Feedback system (admin-reviewed disputes)
CREATE TABLE feedback (
    id                       SERIAL PRIMARY KEY,
    log_id                   INTEGER REFERENCES logs(id),
    user_id                  TEXT NOT NULL,
    server_id                TEXT NOT NULL,
    platform                 TEXT DEFAULT 'discord',
    
    -- Model prediction
    predicted_label          INTEGER NOT NULL,    -- 1=toxic, 0=safe
    predicted_score          REAL,
    
    -- User dispute
    user_claimed_label       INTEGER NOT NULL,
    dispute_reason           TEXT,
    disputed_at              TIMESTAMP DEFAULT NOW(),
    
    -- Admin review
    admin_reviewed           BOOLEAN DEFAULT FALSE,
    admin_decision           TEXT,                -- 'agree_with_model' | 'agree_with_user'
    final_label              INTEGER,
    reviewed_by              TEXT,
    reviewed_at              TIMESTAMP,
    
    used_in_training         BOOLEAN DEFAULT FALSE
);

-- Indexes for performance
CREATE INDEX idx_logs_user_server ON logs(user_id, server_id, timestamp DESC);
CREATE INDEX idx_logs_severity ON logs(severity, timestamp DESC);
CREATE INDEX idx_feedback_review ON feedback(admin_reviewed, server_id);
CREATE INDEX idx_feedback_training ON feedback(used_in_training, admin_reviewed);
```

#### Redis Feature Store

```
Key Pattern: {feature_group}:{version}:{entity_id}

Example Keys:
user_toxicity:prod:user_123 → {
    "user_bad_ratio_7d": 0.15,
    "violation_count_7d": 3,
    "total_messages_7d": 20,
    "hours_since_last_msg": 2.5,
    "user_toxicity_trend": 0.02
}

channel_stats:prod:channel_456 → {
    "channel_toxicity_ratio": 0.08,
    "total_messages_24h": 150
}

TTL: 7 days (auto-expire)
Sync: Every 6 hours from Supabase
```

---

## 🔄 Data Flow Diagrams

### End-to-End Message Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION MESSAGE FLOW                          │
└─────────────────────────────────────────────────────────────────────┘

User: "you are absolute trash"
    ↓
Discord Bot (on_message)
    ↓
┌─────────────────────────────────────────────┐
│ 1. FEATURE EXTRACTION                       │
├─────────────────────────────────────────────┤
│ Text: "you are absolute trash"              │
│ user_id: "user_123"                         │
│ channel_id: "channel_456"                   │
│ server_id: "server_789"                     │
└───────────────────┬─────────────────────────┘
                    │
                    │ HTTP POST /predict
                    ▼
┌─────────────────────────────────────────────┐
│ 2. INFERENCE API                            │
├─────────────────────────────────────────────┤
│ A. Calculate Text Features:                 │
│    ├─ DistilBERT embeddings                 │
│    ├─ msg_len: 24                           │
│    ├─ caps_ratio: 0.0                       │
│    └─ slur_count: 1 ("trash")               │
│                                             │
│ B. Enrich with Redis:                       │
│    ├─ user_bad_ratio_7d: 0.15               │
│    ├─ violation_count_7d: 3                 │
│    └─ channel_toxicity_ratio: 0.08          │
│                                             │
│ C. Model Prediction:                        │
│    XGBoost(768 text + 20 context features)  │
│    → P(toxic) = 0.85                        │
└───────────────────┬─────────────────────────┘
                    │
                    │ Response: {is_toxic: true, confidence: 0.85, severity: "HIGH"}
                    ▼
┌─────────────────────────────────────────────┐
│ 3. BOT EXECUTES ACTION                      │
├─────────────────────────────────────────────┤
│ if severity == HIGH:                        │
│    ├─ Delete message                        │
│    ├─ Add strike (now 4/7)                  │
│    ├─ Timeout user (60 min)                 │
│    └─ Log to Supabase                       │
└───────────────────┬─────────────────────────┘
                    │
                    │ INSERT INTO logs (...)
                    ▼
┌─────────────────────────────────────────────┐
│ 4. SUPABASE STORAGE                         │
├─────────────────────────────────────────────┤
│ logs table:                                 │
│   id: 12345                                 │
│   message: "you are absolute trash"         │
│   toxicity_score: 0.85                      │
│   severity: HIGH                            │
│   action_taken: timeout_60m_strike_4        │
│   explanation: NULL (generated on demand)   │
└───────────────────┬─────────────────────────┘
                    │
                    │ User types: !explain
                    ▼
┌─────────────────────────────────────────────┐
│ 5. LIME EXPLANATION GENERATION              │
├─────────────────────────────────────────────┤
│ POST /explain                               │
│    ↓                                        │
│ LIME analyzes model:                        │
│   "you"       → +0.08 (neutral)             │
│   "are"       → +0.05 (neutral)             │
│   "absolute"  → +0.12 (intensifier)         │
│   "trash"     → +0.60 (toxic) ← TRIGGER     │
│    ↓                                        │
│ Update logs.explanation = {...}             │
└───────────────────┬─────────────────────────┘
                    │
                    │ DM to user
                    ▼
┌─────────────────────────────────────────────┐
│ 6. USER RECEIVES EXPLANATION                │
├─────────────────────────────────────────────┤
│ Discord DM:                                 │
│   📊 Toxicity Analysis                      │
│   ▓▓▓▓▓▓▓▓░░ 85%                           │
│   Trigger: "trash" (+60%)                   │
│   Buttons: [✅ Correct] [❌ Wrong]          │
│                                             │
│ User clicks: ❌ Wrong                       │
│    ↓                                        │
│ Modal: "Why was this flagged incorrectly?"  │
│ User: "This was sarcasm about a game"      │
│    ↓                                        │
│ INSERT INTO feedback (log_id=12345, ...)    │
└─────────────────────────────────────────────┘
```

---

## 🔁 Feedback Loop Architecture

### The Complete Learning Cycle

```
┌──────────────────────────────────────────────────────────────────┐
│              CONTINUOUS IMPROVEMENT CYCLE                        │
└──────────────────────────────────────────────────────────────────┘

WEEK 1: Model Flags Message
    │
    ├─ User: "this game is trash 🔥"
    ├─ Model: 0.85 toxic → DELETE
    ├─ User: !explain → ❌ "This was praise"
    └─ Stored in feedback table (pending admin review)
    
    ↓

WEEK 1: Admin Reviews Dispute
    │
    ├─ Admin: /review
    ├─ Sees: Model=🔴Toxic | User=✅Safe
    ├─ Reads context: "trash 🔥" = gaming slang for "amazing"
    ├─ Decision: ❌ User Correct (false positive)
    └─ UPDATE feedback SET admin_decision='agree_with_user', final_label=0
    
    ↓

MONTH END: Data Ingestion
    │
    ├─ Fetch training data:
    │   ├─ 5% of high-conf safe (anchors)
    │   ├─ 5% of high-conf toxic (anchors)
    │   ├─ 100% admin-reviewed feedback
    │   └─ Total: 10,000 messages
    │
    ├─ Dataset includes:
    │   ├─ "you are trash" → label=1 (toxic)
    │   └─ "this game is trash 🔥" → label=0 (safe) ← CORRECTED
    │
    └─ Save to S3 Parquet
    
    ↓

MONTH END: Model Retraining
    │
    ├─ Load 10K messages from S3
    ├─ Train XGBoost on corrected labels
    ├─ Model learns:
    │   ├─ "trash" alone → toxic
    │   └─ "trash" + emoji → safe (context matters!)
    ├─ Accuracy improves: 94.2% → 95.1%
    └─ Register in MLflow
    
    ↓

MONTH END: Deployment
    │
    ├─ Copy new model to baked_model/
    ├─ Push to GitHub
    ├─ Railway auto-rebuilds API container
    └─ New model live in production
    
    ↓

WEEK 5: Improved Predictions
    │
    ├─ User: "this game is trash 🔥"
    ├─ New Model: 0.35 uncertain → NO ACTION ✅
    ├─ False positive rate drops: 8% → 6%
    └─ User satisfaction increases
```

### Admin Review Dashboard

```
Admin types: /review
    ↓
┌───────────────────────────────────────────────────────────────┐
│  📋 REVIEW QUEUE                                              │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Item #1                                                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Message: "you're trash at this game lol"                │ │
│  │                                                          │ │
│  │ Model Says: 🔴 Toxic (75% confidence)                   │ │
│  │ User Says:  ✅ Safe                                      │ │
│  │                                                          │ │
│  │ User Reason: "This was friendly banter between friends" │ │
│  │ User History: 0 previous violations                     │ │
│  │ Disputed: 2 hours ago                                   │ │
│  │                                                          │ │
│  │ [✅ Model Correct] [❌ User Correct] [⏭️ Skip] [📝 Note]│ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Pending: 12 items                                            │
│  Reviewed today: 8                                            │
│  Model accuracy (last 30d): 94%                               │
└───────────────────────────────────────────────────────────────┘

Admin clicks: ❌ User Correct
    ↓
UPDATE feedback SET 
    admin_reviewed = TRUE,
    admin_decision = 'agree_with_user',
    final_label = 0
    ↓
Next item appears automatically
```

### Bulk Operations (Time Saver)

```
Admin: /review view:Group by User
    ↓
┌───────────────────────────────────────────────────────────────┐
│  👥 USERS WITH PENDING DISPUTES                               │
├───────────────────────────────────────────────────────────────┤
│  @TrollUser     15 disputes                                   │
│  @GamerKid       8 disputes                                   │
│  @NewUser        3 disputes                                   │
└───────────────────────────────────────────────────────────────┘

Admin: /review_user user:@GamerKid
    ↓
┌───────────────────────────────────────────────────────────────┐
│  📋 Reviewing @GamerKid's 8 Disputes                          │
├───────────────────────────────────────────────────────────────┤
│  1. "this is fire bro"        → Model:🔴 User:✅              │
│  2. "gg ez trash team"        → Model:🔴 User:✅              │
│  3. "lmao you got destroyed"  → Model:🔴 User:✅              │
│  ... (5 more)                                                 │
│                                                               │
│  Pattern: Gaming slang being flagged                          │
│                                                               │
│  [✅ Approve All - Model Correct]                            │
│  [❌ Approve All - Users Correct] ← Click this               │
│  [📋 Review Individually]                                     │
└───────────────────────────────────────────────────────────────┘

Result: All 8 marked as false positives in 5 seconds
        (vs. 4 minutes reviewing one-by-one)
```

---

## 🧠 ML Pipeline Architecture

### Training Data Composition (Strategy 2: Stratified Sampling)

```
┌────────────────────────────────────────────────────────────────┐
│               TRAINING DATA SOURCES                            │
└────────────────────────────────────────────────────────────────┘

FROM Supabase logs (last 30 days):
    │
    ├─ HIGH-CONFIDENCE SAFE (score < 0.3)
    │    ├─ Total: 70,000 messages
    │    ├─ Sample: 5% = 3,500 messages ← ANCHORS
    │    └─ Label: 0 (safe) from model
    │
    ├─ HIGH-CONFIDENCE TOXIC (score > 0.7)
    │    ├─ Total: 10,000 messages
    │    ├─ Sample: 5% = 500 messages ← ANCHORS
    │    └─ Label: 1 (toxic) from model
    │
    └─ ADMIN-REVIEWED FEEDBACK
         ├─ Total: 200 disputes reviewed
         ├─ Sample: 100% = 200 messages ← CORRECTIONS
         └─ Label: final_label from admin decision
         
TOTAL TRAINING DATA: 4,200 messages
    ├─ Anchors: 4,000 (95%)  → Prevents forgetting
    └─ Corrections: 200 (5%) → Improves accuracy

WHY THIS WORKS:
├─ Anchors keep model grounded in obvious examples
├─ Model doesn't forget "clearly toxic" vs "clearly safe"
├─ Corrections fix mistakes on edge cases
└─ Distribution matches production (prevents shift)
```

### Monthly Retraining Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│          AUTOMATED MLOPS ORCHESTRATOR (Prefect)                │
│                  Runs: Daily 3 AM                              │
└────────────────────────────────────────────────────────────────┘

01:00 AM - Pre-Flight Check
    │
    ├─ Query Supabase: COUNT(*) FROM logs WHERE timestamp > NOW() - 24h
    ├─ Minimum: 100 messages
    └─ Status: ✅ 1,247 messages available

03:00 AM - Drift Detection
    │
    ├─ Fetch last 7 days of data
    ├─ Compare to reference dataset (training data)
    ├─ Evidently Report:
    │   ├─ Data Drift: 0.15 (low)
    │   ├─ Model Drift: 0.08 (low)
    │   └─ Prediction Drift: 0.22 (medium)
    └─ Decision: No retraining needed (drift < 0.3)

03:10 AM - Data Ingestion (runs anyway for accumulation)
    │
    ├─ Fetch stratified sample from Supabase
    ├─ Calculate features (text + context)
    ├─ Quality checks:
    │   ├─ No nulls in critical columns ✅
    │   ├─ No spam/repetition attacks ✅
    │   ├─ Class balance acceptable ✅
    │   └─ Label sources verified ✅
    ├─ Merge with existing Parquet
    ├─ Deduplicate by text hash
    ├─ Push to S3 via DVC
    └─ Status: ✅ +1,247 messages added

Monthly (1st of month) - Retraining Trigger
    │
    ├─ Check: 30 days since last train
    ├─ Pull training data from S3 (now 45,000 messages)
    ├─ Train DistilBERT + XGBoost pipeline
    ├─ Cross-validate (5-fold)
    ├─ Metrics:
    │   ├─ Accuracy: 95.3% (was 94.8%) ↑
    │   ├─ Precision: 93.1%
    │   ├─ Recall: 91.7%
    │   ├─ F1: 92.4%
    │   └─ False Positive Rate: 6.2% (was 7.8%) ↓
    ├─ Register in MLflow
    ├─ Tag: production-2024-02
    └─ Status: ✅ Ready for deployment

Monthly (1st) - Deployment
    │
    ├─ Copy model to baked_model/
    ├─ Git commit + push
    ├─ Railway detects push
    ├─ Rebuilds API container (5 min)
    ├─ Health check: /health
    └─ Status: ✅ New model live
```

### Drift Detection Dashboard

```
┌──────────────────────────────────────────────────────────────┐
│              EVIDENTLY DRIFT REPORT                          │
│                  Week of Feb 12-18, 2026                     │
└──────────────────────────────────────────────────────────────┘

DATA DRIFT (Feature Distribution Changes)
├─ msg_len:               0.02 (stable)    ✅
├─ caps_ratio:            0.15 (low drift) ✅
├─ user_bad_ratio_7d:     0.31 (HIGH)      ⚠️
├─ slur_count:            0.05 (stable)    ✅
└─ channel_toxicity:      0.08 (stable)    ✅

MODEL DRIFT (Prediction Distribution Changes)
├─ P(toxic) mean:         0.15 → 0.18      ↑
├─ P(toxic) std:          0.22 → 0.24      ↑
└─ Confidence calibration: 0.92            ✅

PREDICTION DRIFT (Actual Label Distribution)
├─ Toxic rate:            8% → 11%         ↑↑
├─ Safe rate:             92% → 89%        ↓
└─ Uncertain rate:        3% → 4%          ↑

RECOMMENDATION:
⚠️ MEDIUM DRIFT DETECTED
   user_bad_ratio_7d shows significant drift (0.31)
   Toxic rate increased 3 percentage points
   
   Suggested Actions:
   1. Investigate: Why are users more toxic this week?
   2. Collect more labels: Review uncertain queue
   3. Consider retraining if drift persists 2+ weeks
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ (or Supabase account)
- Redis 7+
- Discord Bot Token
- S3-compatible storage (AWS S3, MinIO, or Supabase Storage)

### Quick Start (Local Development)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/antibully-bot.git
cd antibully-bot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# 5. Run database migrations
python scripts/run_migrations.py

# 6. Start services
# Terminal 1: API
uvicorn api_service.app:app --reload --port 8000

# Terminal 2: Bot
python -m bot_service.bot

# Terminal 3: Redis (if local)
redis-server
```

### Environment Variables

Create `.env` file:

```bash
# ═══════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════
DATABASE_URL=postgresql://user:pass@host:5432/db

# ═══════════════════════════════════════════════════════════════
# REDIS (Feature Store)
# ═══════════════════════════════════════════════════════════════
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=  # Optional

# ═══════════════════════════════════════════════════════════════
# DISCORD
# ═══════════════════════════════════════════════════════════════
DISCORD_TOKEN=your_bot_token_here
DISCORD_APPLICATION_ID=your_app_id

# ═══════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════
API_BASE_URL=http://localhost:8000

# ═══════════════════════════════════════════════════════════════
# MLOPS
# ═══════════════════════════════════════════════════════════════
MLFLOW_TRACKING_URI=https://dagshub.com/user/repo.mlflow
S3_BUCKET=your-s3-bucket
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════
MODEL_LOCAL_PATH=./baked_model
EXPERIMENT_NAME=toxicity-detector
STAGE=Production
```

---

## 📖 Usage

### For Server Admins

#### Initial Setup

```bash
# 1. Invite bot to server (requires Administrator permission)
https://discord.com/api/oauth2/authorize?client_id=YOUR_APP_ID&permissions=1099780063238&scope=bot%20applications.commands

# 2. Configure moderation settings
/config

# Interactive menu appears:
├─ ⚡ Strike System (strikes before timeout/kick/ban)
├─ ⏱️ Timeouts (duration in minutes)
├─ 🎯 Thresholds (confidence levels)
├─ ⚙️ Actions (warn/delete/timeout/kick/ban)
├─ 🔧 Features (auto_moderate, send_dm_warnings, etc)
├─ 📢 Channels (log_channel, alert_channel)
└─ 🔄 Refresh (reload current config)

# 3. Quick presets available
/quickset preset:balanced  # Recommended for most servers
```

#### Daily Admin Tasks

```bash
# Review disputed messages (5-10 min/day)
/review

# Check statistics
/review view:stats

# Handle bulk disputes
/review view:by_user
/review_user user:@someone

# Pardon users (false positives)
/pardon user:@someone reason:"Bot error"

# View user history
/strikes user:@someone
```

### For Users

```bash
# If you get flagged, understand why
!explain

# You'll receive a DM with:
# ├─ Word importance visualization
# ├─ Your violation history
# └─ Dispute buttons if you disagree
```

---

## 🔧 Configuration

### Server Config Schema

```python
{
    # Strike System
    "strikes_before_timeout": 3,
    "strikes_before_kick": 5,
    "strikes_before_ban": 7,
    "strike_decay_days": 30,  # Strikes expire after 30 days
    
    # Timeout Durations (minutes)
    "timeout_duration_low": 10,
    "timeout_duration_medium": 60,
    "timeout_duration_high": 1440,  # 24 hours
    
    # Detection Thresholds (0.0-1.0)
    "threshold_low": 0.3,
    "threshold_medium": 0.5,
    "threshold_high": 0.7,
    
    # Actions per Severity
    "low_severity_action": "warn",      # warn | delete | timeout
    "medium_severity_action": "delete",
    "high_severity_action": "timeout",
    
    # Features
    "auto_moderate": true,              # Auto-delete toxic messages
    "send_dm_warnings": true,           # DM users when flagged
    "require_human_review": false,      # Admin approval required
    "log_all_messages": false,          # Log safe messages too
    
    # Channels
    "log_channel_id": "123456789",      # Moderation log
    "alert_channel_id": "987654321",    # High-severity alerts
    
    # Misc
    "warning_message_template": "{mention}, your message violated community guidelines.",
    "warning_delete_delay": 30          # Auto-delete warnings after 30s
}
```

---

## 📡 API Reference

### POST /predict

Classify message toxicity.

**Request:**
```json
{
  "text": "you are trash",
  "user_id": "user_123",
  "channel_id": "channel_456"
}
```

**Response:**
```json
{
  "is_toxic": true,
  "confidence": 0.85,
  "severity": "HIGH",
  "features_used": {
    "text_derived": {
      "msg_len": 13,
      "caps_ratio": 0.0,
      "slur_count": 1
    },
    "user_context": {
      "user_bad_ratio_7d": 0.15,
      "violation_count_7d": 3
    }
  }
}
```

### POST /explain

Generate LIME explanation.

**Request:**
```json
{
  "text": "you are trash",
  "user_id": "user_123",
  "num_features": 6
}
```

**Response:**
```json
{
  "toxic_probability": 0.85,
  "trigger_words": [
    {"word": "trash", "score": 0.65, "category": "toxic"},
    {"word": "you", "score": 0.08, "category": "safe"},
    {"word": "are", "score": 0.05, "category": "safe"}
  ],
  "features_used": {...}
}
```

---

## 🚢 Deployment

### Railway.app (Recommended for MLH/Demo)

**Cost:** ~$19/month | **Setup Time:** 30 minutes

```bash
# 1. Create account at railway.app
# 2. Connect GitHub repository
# 3. Create 4 services:

Service 1: Discord Bot
├─ Dockerfile: Dockerfile.bot
├─ Start: python -m bot_service.bot
├─ RAM: 512MB
└─ Env vars: DISCORD_TOKEN, DATABASE_URL, API_BASE_URL

Service 2: Inference API
├─ Dockerfile: Dockerfile.api
├─ Start: uvicorn api_service.app:app
├─ RAM: 2GB (for model)
├─ Port: 8000 (public)
└─ Env vars: DATABASE_URL, REDIS_URL, MODEL_LOCAL_PATH

Service 3: Redis (Plugin)
├─ Type: Redis
└─ Auto-provision (1-click)

Service 4: MLOps Worker (Cron)
├─ Dockerfile: Dockerfile.mlops
├─ Schedule: 0 3 * * * (3 AM daily)
└─ Env vars: DATABASE_URL, S3 credentials

# 4. Deploy
git push origin main
# Railway auto-deploys on push
```

### AWS Fargate (Production Scale)

See [DEPLOYMENT.md](./docs/DEPLOYMENT.md) for full guide.

---

## 📊 MLOps

### Data Versioning (DVC)

```bash
# Track training data
dvc add data/training_data.parquet
git add data/training_data.parquet.dvc
git commit -m "Add training data v1"

# Push to remote storage
dvc push

# Pull on another machine
dvc pull
```

### Experiment Tracking (MLflow)

```python
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/user/repo.mlflow")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

### Model Registry

```bash
# Register model
mlflow models register -m runs:/<run_id>/model -n toxicity-detector

# Promote to production
mlflow models set-tag -n toxicity-detector -v 3 -t stage -v Production

# Deploy new version
# API auto-loads: models:/toxicity-detector/Production
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md).

**High-Priority Areas:**
- [ ] Slack bot implementation
- [ ] WhatsApp bot implementation  
- [ ] Streamlit admin dashboard
- [ ] More comprehensive tests
- [ ] Localization (i18n)

---

## 📄 License

MIT License - see [LICENSE](./LICENSE)

---

## 🙏 Acknowledgments

- **DistilBERT**: Hugging Face Transformers
- **LIME**: Marco Tulio Ribeiro et al.
- **Evidently AI**: Drift detection framework
- **Prefect**: Workflow orchestration
- **MLflow**: Experiment tracking

---

## 📞 Support

- **Documentation**: [docs.antibully.bot](https://docs.antibully.bot)
- **Discord**: [Join our server](https://discord.gg/antibully)
- **Issues**: [GitHub Issues](https://github.com/yourusername/antibully-bot/issues)

---

## 🎯 Project Status

**Current Version:** 1.0.0 (Production Ready)

**Roadmap:**
- [x] Discord bot with LIME explanations
- [x] Admin feedback loop
- [x] Automated MLOps pipeline
- [ ] Multi-language support
- [ ] Slack integration
- [ ] WhatsApp integration
- [ ] Mobile admin app

---

Built with ❤️ for safer online communities.

**Star this repo if you found it useful!** ⭐