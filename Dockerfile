# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for pandas/pyarrow, especially for various compression codecs
# (e.g., snappy, lz4, zstd). These are often pre-compiled but good to include for robustness.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsnappy-dev \
    liblz4-dev \
    libzstd-dev \
    libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./app /app/app

# Expose the port the app runs on
EXPOSE 8000

# Run the application using uvicorn
# --host 0.0.0.0 makes the server accessible from outside the container
# --port 8000 specifies the port
# app.main:app refers to the 'app' object in 'main.py' inside the 'app' directory
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
