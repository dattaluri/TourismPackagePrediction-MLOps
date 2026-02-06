# Use a Python 3.9 image as the base
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port that the application will run on (e.g., 8000 for FastAPI, 5000 for Flask)
EXPOSE 8000

# Define the command to run your application (e.g., a FastAPI app)
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# This will be replaced with the actual command once app.py is defined
# For now, it's a placeholder or can be omitted if app.py is not yet ready.
