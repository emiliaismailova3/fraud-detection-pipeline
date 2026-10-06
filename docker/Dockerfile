# Inference image for the fraud-scoring API.
# Build from the repo root:  docker build -f docker/Dockerfile -t fraud-api .

# Small official Python image based on Debian Linux
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime (parallel computing library) from the OS
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Don't write .pyc files; print logs immediately instead of buffering them
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first: Docker caches this layer, so code changes
# don't trigger a slow reinstall
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Then copy only what the service needs to run
COPY src/__init__.py src/config.py src/preprocess.py src/
COPY src/api/ src/api/
COPY models/preprocessor.joblib models/lightgbm.txt models/api_config.json models/

# Run as a non-root user: if the app is compromised, the attacker is not root
RUN useradd --create-home appuser
USER appuser

# Cloud Run tells the container which port to listen on via $PORT (default 8080)
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
