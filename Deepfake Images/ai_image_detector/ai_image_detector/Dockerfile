# Use NVIDIA CUDA base image for PyTorch GPU support
# For CPU-only host, change to: FROM python:3.12-slim
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set non-interactive timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.12 and system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements & install Python dependencies
COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir --upgrade pip && \
    python3.12 -m pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port 5000
EXPOSE 5000

# Set Python Path
ENV PYTHONPATH=/app

# Start Gunicorn server (4 workers, 120s timeout for model loading)
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "--timeout=120", "run:app"]
