FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 🔴 THIS IS THE FIX
ENV PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY index.html ./index.html

EXPOSE 8000

CMD ["uvicorn", "multi_agent_decision_system.main:app", "--host", "0.0.0.0", "--port", "8000"]