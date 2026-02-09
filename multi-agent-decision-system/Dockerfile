FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY index.html ./index.html

ENV PYTHONPATH=/app/src

# Cloud Run uses PORT env var
CMD ["sh", "-c", "uvicorn multi_agent_decision_system.main:app --host 0.0.0.0 --port ${PORT:-8080}"]