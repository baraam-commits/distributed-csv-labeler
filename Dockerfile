FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/requirements.txt /app/
RUN pip install -r requirements.txt

COPY app/ /app/app/

# Default; compose overrides with arguments
CMD ["python","-u","/app/app/node.py"]
