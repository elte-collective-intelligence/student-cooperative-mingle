FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default: run tests
# Override with: docker run <image> python train.py algorithm=ppo
CMD ["pytest", "-v", "tests/"]

# Usage examples:
# Build:     docker build -t cooperative-mingle .
# Run tests: docker run cooperative-mingle
# Training:  docker run cooperative-mingle python train.py algorithm=ppo train.total_frames=50000
# MAPPO:     docker run cooperative-mingle python train.py algorithm=mappo fairness=gini
# Sweep:     docker run cooperative-mingle python train.py --multirun algorithm=ppo,ippo,mappo