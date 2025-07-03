FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir python-doctr[torch] fastapi uvicorn

COPY app.py . 

# Expose port
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "1000"]
