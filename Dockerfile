# Multi-stage lightweight Docker image for SentinelGuard
FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy SentinelGuard application code
COPY . .

# Generate dataset, train models, and seed database inside container
RUN python ml/dataset_generator.py && \
    python ml/train_model.py && \
    python utils/db_seed.py

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Start script to run both FastAPI backend and Streamlit frontend
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
