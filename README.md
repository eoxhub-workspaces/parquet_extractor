# Parquet Statistics Endpoint

This project provides a FastAPI-based endpoint to extract, dynamically filter, and inspect data from monthly Parquet files stored on S3.

## Features

*   **S3 Parquet Data Extraction**: Reads Parquet files directly from S3 using `pandas`, `duckdb`, and `geopandas` with optimized range requests.
*   **Datetime-based File Selection**: Automatically determines the correct monthly Parquet file based on an input ISO 8601 datetime string.
*   **Dynamic Server-Side Filtering**: Supports URL-query-based filtering on geospatial and storm-level metrics (lightning group counts, area size, duration, bounding boxes, categorical land/water types, etc.) with overflow thresholds.
*   **Flexible Output**: Returns filtered data in JSON, CSV, or GeoJSON formats.
*   **STAC/GeoParquet Catalog**: Generates SpatioTemporal Asset Catalogs dynamically pointing at filtered GeoJSON endpoints.
*   **Production Hardened (Security-First)**:
    *   **Minimalist Base Image**: Utilizes Python distroless-like multi-stage Debian slim images (`python:3.11-slim`) to minimize attack vectors.
    *   **Strict Non-Root Execution**: Runs strictly as non-root user `appuser` (UID: 10001) in compliance with modern restricted Kubernetes cluster policies.
    *   **CI Vulnerability Scanning**: Integrated Trivy image scanning in the GitHub action pipeline.
    *   **Keyless Image Signing**: Configured keyless cryptographic signing via Sigstore/Cosign in CI using GitHub OIDC.

---

## Dynamic Query Parameter Filtering

The `/data/geojson` endpoint supports dynamic, server-side filtering. Filters are applied to the Parquet dataset before generating the GeoJSON FeatureCollection.

### Supported Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lightning_min` | `float` / `int` | `0` | Minimum storm lightning group count |
| `lightning_max` | `float` / `int` | `5000` | Maximum storm lightning group count (value $\ge 5000$ acts as "all/overflow") |
| `area_min` | `float` / `int` | `0` | Minimum storm lightning area in km² |
| `area_max` | `float` / `int` | `5000` | Maximum storm lightning area in km² (value $\ge 5000$ acts as "all/overflow") |
| `duration_min` | `float` / `int` | `0` | Minimum recorded storm duration in minutes |
| `duration_max` | `float` / `int` | `121` | Maximum recorded storm duration in minutes |
| `lon_min` | `float` / `int` | `-180` | Bounding box minimum longitude |
| `lon_max` | `float` / `int` | `180` | Bounding box maximum longitude |
| `lat_min` | `float` / `int` | `-90` | Bounding box minimum latitude |
| `lat_max` | `float` / `int` | `90` | Bounding box maximum latitude |
| `surface_type` | `str` | `"all"` | Categorical filter (`"all"`, `"land"`, or `"water"`) |
| `earthcare_id`| `str` | `""` | Optional filter match on EarthCARE ID |

---

## Development & Test Setup

### Local Installation

```bash
# Set up a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Running Unit Tests

The test suite includes complete isolated verification for filters, threshold overflows, bounding boxes, categorical matches, and FastAPI router mapping:

```bash
python3 -m unittest tests/test_filtering.py
```

---

## Docker & Security Verification

### Building the Secure Image

The project uses a secure, multi-stage, non-root build:

```bash
docker build -t parquet-extractor:latest .
```

To run the container locally:

```bash
docker run -p 8000:8000 parquet-extractor:latest
```

---

### Trivy Security Scanning

To ensure the container is secure and free of vulnerabilities, you can scan the image locally using **Trivy**:

#### 1. Install Trivy
- **macOS (Homebrew)**: `brew install aquasecurity/trivy/trivy`
- **Linux (Debian/Ubuntu)**: 
  ```bash
  sudo apt-get install wget apt-transport-https gnupg lsb-release
  wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | gpg --dearmor | sudo tee /usr/share/keyrings/trivy.gpg > /dev/null
  echo "deb [signed-by=/usr/share/keyrings/trivy.gpg] https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/trivy.list
  sudo apt-get update
  sudo apt-get install trivy
  ```

#### 2. Run Scan
Scan the local image for High and Critical vulnerabilities, ignoring unfixable/upstream library vulnerabilities:

```bash
trivy image --severity HIGH,CRITICAL --ignore-unfixed parquet-extractor:latest
```

In the CI/CD pipeline (`.github/workflows/build-and-push.yaml`), this check is automated and will block any deployment if high or critical vulnerabilities exist.
