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

COPY .dvc/ /app/.dvc/
COPY shared/ /app/shared/
COPY mlops/ /app/mlops
COPY scripts/ /app/scripts/

RUN mkdir -p /app/data/archive /app/data/logs

ENV PYTHONPATH=/app
# CMD ["python", "mlops/orchestrator.py"]aan