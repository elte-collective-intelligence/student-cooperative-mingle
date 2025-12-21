# Cooperative Mingle - Multi-Agent Reinforcement Learning
# ========================================================
# Build:     docker build -t cooperative-mingle .
# Run tests: docker run cooperative-mingle
# Training:  docker run cooperative-mingle python train_hydra.py
# GPU:       docker run --gpus all cooperative-mingle python train_hydra.py

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create outputs directory
RUN mkdir -p /app/outputs

# Default: run tests
CMD ["pytest", "-v", "tests/"]

# ========================================================
# Usage Examples:
# ========================================================
#
# BUILD:
#   docker build -t cooperative-mingle .
#
# RUN TESTS:
#   docker run cooperative-mingle
#   docker run cooperative-mingle pytest tests/ -v --tb=short
#
# TRAINING (single experiment):
#   docker run cooperative-mingle python train_hydra.py
#   docker run cooperative-mingle python train_hydra.py algorithm=mappo
#   docker run cooperative-mingle python train_hydra.py algorithm=mappo fairness=gini
#   docker run cooperative-mingle python train_hydra.py communication=discrete
#
# TRAINING (sweeps - multiple experiments):
#   docker run cooperative-mingle python train_hydra.py --multirun algorithm=ppo,ippo,mappo
#   docker run cooperative-mingle python train_hydra.py --multirun communication=none,discrete,continuous
#   docker run cooperative-mingle python train_hydra.py --multirun fairness=none,gini,participation
#
# FULL STACK:
#   docker run cooperative-mingle python train_hydra.py algorithm=mappo communication=discrete fairness=gini curriculum=progressive
#
# EVALUATION:
#   docker run cooperative-mingle python evaluate.py model_path=outputs/model.pt
#
# WITH GPU (requires nvidia-docker):
#   docker run --gpus all cooperative-mingle python train_hydra.py
#
# MOUNT OUTPUT DIRECTORY:
#   docker run -v $(pwd)/outputs:/app/outputs cooperative-mingle python train_hydra.py
#
# ========================================================