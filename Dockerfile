# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POPPLER_PATH=/usr/bin

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create temp directory
RUN mkdir -p /app/temp_files

# Expose port
EXPOSE 1000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "1000", "--timeout-keep-alive", "300"]
