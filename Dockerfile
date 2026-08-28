# Multi-architecture production Dockerfile for Pneumonia Diagnostic Hub AI Engine
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PORT=7860 \
    HOME=/home/user

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up user for Hugging Face Spaces security standards
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Copy requirements and install dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn[standard]

# Copy entire application codebase
COPY --chown=user:user . .

# Ensure directory permissions for static outputs
RUN mkdir -p static/uploads static/samples models && \
    chmod -R 777 static/uploads static/samples models

USER user

# Expose Hugging Face Space port
EXPOSE 7860

# Start production ASGI Uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
