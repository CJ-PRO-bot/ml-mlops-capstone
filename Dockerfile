FROM python:3.10-slim

WORKDIR /app

# Faster + smaller
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies first (so Docker cache works)
COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir -r /app/requirements-docker.txt

# Copy project (will be small because .dockerignore filters big stuff)
COPY . /app