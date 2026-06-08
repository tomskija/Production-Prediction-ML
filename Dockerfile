FROM python:3.11-slim

# System dependencies + Node.js 20
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    procps \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY productionPredictionCalculator/ ./productionPredictionCalculator/

# Keep container alive for devcontainer use — run script manually
CMD ["sleep", "infinity"]