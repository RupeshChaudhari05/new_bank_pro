FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add source code
COPY . /app
WORKDIR /app

# Expose API port
EXPOSE 1000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "1000"]
