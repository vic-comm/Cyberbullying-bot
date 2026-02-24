FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
    

RUN pip install --no-cache-dir --default-timeout=1000 \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --default-timeout=1000 transformers==4.57.6 

COPY api_service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir asyncpg==0.29.0 pyarrow==15.0.0
RUN pip install --no-cache-dir --upgrade "numpy>=2.0.0"
RUN pip install --no-cache-dir "scikit-learn==1.8.0" --upgrade xgboost

COPY shared/ /app/shared/
COPY api_service/ /app/api_service/
COPY mlops/ /app/mlops/
COPY scripts/ /app/scripts/

COPY baked_model/ /app/baked_model/
COPY baked_bert/ /app/baked_bert/

ENV PYTHONPATH=/app

ENV MODEL_LOCAL_PATH=/app/baked_model
ENV BAKED_BERT_PATH=/app/baked_bert
ENV TRANSFORMERS_OFFLINE=1

CMD ["uvicorn", "api_service.app:app", "--host", "0.0.0.0", "--port", "8000"]