# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies (libsndfile for soundfile, ffmpeg for pydub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt /app/
# Optimize: 
# 1. Substitute 'tensorflow' with 'tensorflow-cpu' to prevent installing heavy GPU dependencies.
# 2. Pre-install PyTorch CPU-only version.
# 3. Install remaining requirements (pip will skip already-satisfied packages like torch/tensorflow).
RUN sed -i 's/tensorflow/tensorflow-cpu/g' requirements.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app/

# Expose port 5000
EXPOSE 5000

# Start server
CMD ["python", "server.py"]
