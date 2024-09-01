# Use a base image with Python 3.8
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire contents of the current directory to the container
COPY . .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install fastapi uvicorn boto3 zipfile36 joblib dvc[gs] dvc[s3]

# Run the DVC pipeline to download data, preprocess it, and train the model
RUN dvc repro

# Start the FastAPI application
CMD uvicorn src.app.fastapi_app:app --host 0.0.0.0 --port 8000 & \
    echo "FastAPI is running at http://localhost:8000" && \
    tail -f /dev/null
