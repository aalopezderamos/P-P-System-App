# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (libgomp1 required for LightGBM)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install CPU-only PyTorch FIRST (from special index)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and static files
COPY main.py .
COPY init_db.py .
COPY static/ static/

# Expose port 8000
EXPOSE 8000

# Run initialization then start the app
CMD python init_db.py && uvicorn main:app --host 0.0.0.0 --port 8000