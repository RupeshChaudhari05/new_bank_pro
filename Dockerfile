FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 1000
CMD ["gunicorn", "-b", "0.0.0.0:1000", "app:app"]
