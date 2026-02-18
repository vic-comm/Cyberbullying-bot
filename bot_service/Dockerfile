FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1

COPY bot_service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY bot_service/ /app/bot_service/

ENV PYTHONPATH=/app

RUN mkdir -p app/data
VOLUME /app/data

ENV DB_PATH=/app/data/bot_memory.db
CMD ["python", "bot_service/bot.py"]
