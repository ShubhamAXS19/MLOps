# Dockerfile

FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Expose port 5000
EXPOSE 5000

# Run the FastAPI server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5000"]
