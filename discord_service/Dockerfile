FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1

COPY discord_service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir asyncpg==0.29.0 pyarrow==15.0.0

COPY shared/ /app/shared/
COPY discord_service/ /app/discord_service/

ENV PYTHONPATH=/app

CMD ["python", "discord_service/bot.py"]
