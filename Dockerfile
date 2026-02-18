# Use lightweight Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (gcc needed for some python packages)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ENTIRE project (api_service, bot_service, mlops, scripts)
COPY . .

# Make the startup script executable
RUN chmod +x run.sh

# Expose the API port so AWS can see it
EXPOSE 8000

# Run the unified script
CMD ["./run.sh"]