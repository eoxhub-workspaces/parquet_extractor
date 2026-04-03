
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.9.2

# Set the working directory in the container
WORKDIR /app

# Step 1: Install system deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libblosc-dev \
    libz-dev \
    python3-pip

# Step 2: Copy only requirements and install packages
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt --break-system-packages

# Step 3: Remove build tools to reduce final image size
RUN apt-get purge -y \
    build-essential \
    python3-dev \
    libblosc-dev \
    libz-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /root/.cache

# Step 4: Copy source code into image
COPY . .

USER www-data

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
