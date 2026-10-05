# Stage 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools for compiling any C extensions and apply security updates
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Minimal runtime stage
FROM python:3.11-slim

WORKDIR /app

# Apply latest security patches to the base OS libraries (resolves CVE-2026-103111 / libpcre2)
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source code
COPY . .

# Create a dedicated system non-root user and group
RUN groupadd -r -g 10001 appgroup && \
    useradd -r -u 10001 -g appgroup -m -d /home/appuser -s /sbin/nologin appuser && \
    chown -R appuser:appgroup /app

# Switch to the non-root user by UID for security standard compliance
USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
