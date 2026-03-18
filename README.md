# 🛡️ AntiBully Bot

**AI-powered content moderation with explainable decisions and continuous learning.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![Discord.py](https://img.shields.io/badge/discord.py-2.3-blue.svg)](https://discordpy.readthedocs.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://postgresql.org)

> Demo: https://youtu.be/H17_tdERf5g

---

## What It Does

Most moderation tools rely on keyword lists that are easy to bypass, or black-box AI that offers no transparency. AntiBully combines hybrid ML, explainable AI (LIME), and a human-in-the-loop feedback system that improves the model over time.

**Key capabilities:**
- **Hybrid ML** — DistilBERT embeddings + XGBoost, enriched with user context (violation history, channel toxicity) from a Redis feature store
- **Explainability** — LIME highlights which words triggered a flag; users see exactly why they were moderated
- **Feedback loop** — Users dispute false positives → admins review → corrected labels retrain the model monthly
- **Drift detection** — Evidently monitors weekly; automated retraining triggers on significant drift or monthly schedule
- **Multi-platform** — One ML backend serving Discord, Slack, and WhatsApp

---

<<<<<<< HEAD
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
  
### DEMO: https://youtu.be/H17_tdERf5g
---

## 🏗️ System Architecture

### High-Level Architecture
=======
## Architecture
>>>>>>> 33177d6 (Updated readme)

```
┌────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                             │
│   Discord Bot  │  Slack Bot  │  WhatsApp Bot  │  Telegram (WIP)   │
└────────────┬───────────────────────────────────────────────────────┘
             │ WebSocket/REST
             ▼
┌────────────────────────────────────────────────────────────────────┐
│                       BOT SERVICE LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  Moderation  │  │    Admin     │  │   Feedback   │            │
│  │  (on_message)│  │   Commands   │  │  (!explain)  │            │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘            │
└─────────┼─────────────────┼──────────────────┼────────────────────┘
          │ POST /predict   │ GET /config      │ POST /explain
          ▼                 ▼                  ▼
┌────────────────────────────────────────────────────────────────────┐
│                      INFERENCE API (FastAPI)                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │ Toxicity Detector│  │ LIME Explainer   │  │ Feature Enricher│ │
│  │ DistilBERT+XGB   │  │ (word importance)│  │ (Redis lookup)  │ │
│  └──────┬───────────┘  └──────────────────┘  └────────┬────────┘ │
└─────────┼───────────────────────────────────────────────┼──────────┘
          │ log_event()                                   │ get_features()
          ▼                                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                  │
│  ┌─────────────────────────────┐  ┌──────────────────────────────┐│
│  │   Supabase PostgreSQL       │  │      Redis (Feature Store)   ││
│  │  ├─ logs                    │  │  ├─ user_toxicity:prod:{id}  ││
│  │  ├─ server_configs          │  │  ├─ channel_stats:{id}       ││
│  │  ├─ server_user_violations  │  │  └─ (5ms lookups)            ││
│  │  ├─ feedback (disputes)     │  └──────────────────────────────┘│
│  │  └─ admin_review_queue      │                                  │
│  └─────────────────────────────┘                                  │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ Nightly Sync
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│                  MLOPS ORCHESTRATOR (Prefect)                      │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ STEP 1: Drift Detection                                     │  │
│  │  KS-test + PSI (statistical) + Evidently report            │  │
│  │  → severity: none / low / medium / high / critical         │  │
│  └───────────────────────────┬─────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ STEP 2: Data Ingestion                                      │  │
│  │  Stratified sample (5% anchors) + 100% admin feedback      │  │
│  │  Quality checks → merge → DVC push to S3 (Parquet)         │  │
│  │  → status: success / skipped / failed (aborts pipeline)    │  │
│  └───────────────────────────┬─────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ STEP 3: Retraining Decision                                 │  │
│  │  high/critical drift → trigger main_flow() immediately      │  │
│  │  medium drift        → flag for next scheduled run          │  │
│  │  low/none            → skip, model is stable               │  │
│  │  Experiments tracked in MLflow (DagsHub)                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### Model Pipeline

```
Input: "you are trash" + user_id + channel_id
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: TEXT FEATURES                                  │
├─────────────────────────────────────────────────────────┤
│ DistilBERT Embeddings (768 dims)                        │
│ + Static Features:                                      │
│   ├─ msg_len, caps_ratio, slur_count                    │
│   ├─ personal_pronoun_count, question_count             │
│   └─ char_repetition, exclamation_count                 │
└───────────────────┬─────────────────────────────────────┘
                    │ (768 + 15 features)
                    ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 2: USER CONTEXT ENRICHMENT (Redis)                │
├─────────────────────────────────────────────────────────┤
│   ├─ user_bad_ratio_7d (% toxic messages)               │
│   ├─ violation_count_7d                                 │
│   ├─ channel_toxicity_ratio                             │
│   ├─ hours_since_last_msg                               │
│   └─ is_new_to_channel                                  │
└───────────────────┬─────────────────────────────────────┘
                    │ (768 + 15 + 5 = 788 features)
                    ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 3: XGBoost Classifier                             │
├─────────────────────────────────────────────────────────┤
│ Output: P(toxic) ∈ [0, 1]                               │
│   ├─ < 0.3  → SAFE                                      │
│   ├─ 0.3–0.5 → LOW                                      │
│   ├─ 0.5–0.7 → MEDIUM                                   │
│   └─ > 0.7  → HIGH                                      │
└───────────────────┬─────────────────────────────────────┘
                    ▼
  { "is_toxic": true, "confidence": 0.85, "severity": "HIGH" }
```

---

## How the Feedback Loop Works

```
1. Model flags message → User gets DM with LIME explanation
2. User clicks ❌ Wrong → submits dispute reason
3. Admin runs /review → approves or overrides model decision
4. Month-end: corrected labels + high-confidence anchors retrain model
5. New model deployed → false positive rate drops
```

Admins can review disputes one-by-one or bulk-approve by user pattern — useful when the model is systematically misclassifying gaming slang, regional expressions, etc.

---

## Setup Guide

### Prerequisites

- Python 3.11+
- Node.js 18+ (for some ML tooling)
- PostgreSQL 15+ or a [Supabase](https://supabase.com) account
- Redis 7+
- Discord bot token
- S3-compatible storage (AWS S3, MinIO, or Supabase Storage)
- MLflow tracking server (free via [DagsHub](https://dagshub.com))

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/antibully-bot.git
cd antibully-bot

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/db

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=          # leave blank if none

# Discord
DISCORD_TOKEN=your_bot_token
DISCORD_APPLICATION_ID=your_app_id

# API
API_BASE_URL=http://localhost:8000

# MLOps
MLFLOW_TRACKING_URI=https://dagshub.com/user/repo.mlflow
S3_BUCKET=your-bucket
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Model
MODEL_LOCAL_PATH=./baked_model
EXPERIMENT_NAME=toxicity-detector
STAGE=Production
```

### 3. Initialize Database

```bash
python scripts/run_migrations.py
```

This creates the `logs`, `server_configs`, `server_user_violations`, and `feedback` tables, plus indexes.

### 4. Run Services

Open three terminals:

```bash
# Terminal 1 — Inference API
uvicorn api_service.app:app --reload --port 8000

# Terminal 2 — Discord Bot
python -m bot_service.bot

# Terminal 3 — Redis (if running locally)
redis-server
```

### 5. Invite the Bot to Your Server

```
https://discord.com/api/oauth2/authorize?client_id=YOUR_APP_ID&permissions=1099780063238&scope=bot%20applications.commands
```

### 6. Configure Moderation Settings

In your Discord server, run:

```
/config
```

This opens an interactive menu to set strike thresholds, timeout durations, severity actions, log channels, and more. To use safe defaults immediately:

```
/quickset preset:balanced
```

---

## Usage

### For Admins

```bash
/review                  # Work through dispute queue
/review view:stats       # See model accuracy and review counts
/review view:by_user     # Group disputes by user for bulk review
/review_user user:@name  # Bulk approve/override all disputes from one user
/pardon user:@name reason:"false positive"
/strikes user:@name      # View violation history
```

### For Users

```bash
!explain    # Get a DM showing which words triggered the flag, with a dispute button
```

---

## Deployment (Railway.app)

Estimated cost: ~$19/month. Setup time: ~30 minutes.

1. Create an account at [railway.app](https://railway.app) and connect your GitHub repo.
2. Create four services:

| Service | Dockerfile | Start Command | Notes |
|---|---|---|---|
| Discord Bot | `Dockerfile.bot` | `python -m bot_service.bot` | 512 MB RAM |
| Inference API | `Dockerfile.api` | `uvicorn api_service.app:app` | 2 GB RAM, expose port 8000 |
| Redis | Plugin | — | 1-click provision |
| MLOps Worker | `Dockerfile.mlops` | — | Cron: `0 3 * * *` |

3. Add environment variables to each service.
4. Push to `main` — Railway auto-deploys.

---

## API Reference

| Endpoint | Method | Description | Latency |
|---|---|---|---|
| `/predict` | POST | Classify message toxicity | 50–100ms |
| `/explain` | POST | Generate LIME word importance | 2–4s |
| `/feedback` | POST | Record user dispute | <50ms |
| `/health` | GET | Service status | <10ms |

**`POST /predict` example:**

```json
// Request
{ "text": "you are trash", "user_id": "u123", "channel_id": "c456" }

// Response
{ "is_toxic": true, "confidence": 0.85, "severity": "HIGH", "features_used": { ... } }
```

---

## Roadmap

- [x] Discord bot with LIME explanations
- [x] Admin feedback loop and review dashboard
- [x] Automated MLOps pipeline with drift detection
- [ ] Slack integration
- [ ] WhatsApp integration
- [ ] Multi-language support
- [ ] Mobile admin app

---

## Tech Stack

<<<<<<< HEAD
**Star this repo if you found it useful!** ⭐
=======
DistilBERT · XGBoost · LIME · FastAPI · discord.py · Supabase · Redis · MLflow · DVC · Evidently · Prefect · Railway.app

---

MIT License · Built for safer online communities ⭐
>>>>>>> 33177d6 (Updated readme)

