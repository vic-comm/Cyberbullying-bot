FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --default-timeout=1000 \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --default-timeout=1000 transformers==4.57.6 

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install awscli

COPY mlops/ /app/mlops
COPY scripts/ /app/scripts/

RUN mkdir -p /app/data/archive /app/data/logs

ENV PYTHONPATH=/app
CMD ["python", "mlops/orchestrator.py"]

# version: '3.8'

# services:
#   # 1. The Brain (API Service)
#   api:
#     build:
#       context: .
#       dockerfile: Dockerfile.api
#     container_name: toxicity-api
#     ports:
#       - "8000:8000"
#     environment:
#       - REDIS_HOST=redis
#       - REDIS_PORT=6379
#       - MLFLOW_TRACKING_URI=https://dagshub.com/obiezuevictor/Cyberbullying-bot.mlflow
#       - MODEL_STAGE=Production
#     depends_on:
#       - redis
#     networks:
#       - bot-net

#   # 2. The Mouth (Discord Bot)
#   bot:
#     build:
#       context: .
#       dockerfile: Dockerfile.bot
#     container_name: discord-bot
#     environment:
#       - DISCORD_BOT_TOKEN=${DISCORD_BOT_TOKEN}
#       # Notice we use the service name "api" as the URL hostname!
#       - BASE_URL=http://api:8000
#       - DB_PATH=/app/data/bot_memory.db
#     volumes:
#       # Persist the SQLite database file on your host machine
#       - ./data/bot_memory.db:/app/data/bot_memory.db
#     depends_on:
#       - api
#     networks:
#       - bot-net

#   # 3. The Memory (Redis Feature Store)
#   redis:
#     image: redis:alpine
#     container_name: feature-store
#     ports:
#       - "6379:6379"
#     volumes:
#       - redis_data:/data
#     networks:
#       - bot-net

#   # 4. The Trainer (MLOps Worker - Optional Local Run)
#   # This container runs once and exits, or runs on a schedule if you kept the orchestrator loop.
#   mlops-worker:
#     build:
#       context: .
#       dockerfile: Dockerfile.mlops
#     container_name: mlops-worker
#     environment:
#       - REDIS_HOST=redis
#       - MLFLOW_TRACKING_URI=https://dagshub.com/obiezuevictor/Cyberbullying-bot.mlflow
#       # Add your other secrets here
#     depends_on:
#       - redis
#     networks:
#       - bot-net
#     # We typically don't run this 24/7 locally, but it's here if you need to test it.
#     profiles: ["tools"] 

# volumes:
#   redis_data:

# networks:
#   bot-net:
#     driver: bridge