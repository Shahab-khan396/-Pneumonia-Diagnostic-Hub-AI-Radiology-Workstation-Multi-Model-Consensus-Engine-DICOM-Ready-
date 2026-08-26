# Multi-architecture production Dockerfile for Pneumonia Diagnostic Hub
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PORT=7860

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy entire application codebase
COPY ["Flask Application/", "./"]

# Ensure upload directory permissions
RUN mkdir -p static/uploads static/samples && chmod -R 777 static/uploads static/samples

# Expose port (7860 for Hugging Face Spaces / 5000 for standard)
EXPOSE 7860

# Start production WSGI Gunicorn server
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 180 wsgi:app"]
