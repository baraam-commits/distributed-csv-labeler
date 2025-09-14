FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# System deps kept minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY app/requirements.txt /app/app/requirements.txt
RUN pip install -r /app/app/requirements.txt \
    && pip install ollama

# Copy your code + app_data (calibration files live under app/)
COPY app/ /app/app/

# Default: talk to host’s Ollama (works on Docker Desktop Win/Mac)
# For Linux host, override OLLAMA_URL in compose to http://<host-lan-ip>:11434
ENV OLLAMA_URL=http://host.docker.internal:11434

# Run the node
CMD ["python","-u","/app/app/node.py"]