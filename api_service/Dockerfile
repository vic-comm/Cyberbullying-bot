FROM python:3.10-slim

WORKDIR /app

COPY api_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_service/ /app/api_service/

COPY mlops/ /app/mlops/

ENV PYTHONPATH=/app

CMD ["uvicorn", "api_service.app:app", "--host", "0.0.0.0", "--port", "8000"]