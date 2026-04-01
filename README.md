# Parquet Statistics Endpoint

This project provides a FastAPI-based endpoint to extract and filter data from monthly Parquet files stored on S3. It's designed to mimic the setup of the `cog-statistics-endpoint` but with a focus on Parquet data extraction and filtering.

## Features

*   **S3 Parquet Data Extraction**: Reads Parquet files directly from S3 using `pandas` and `s3fs`.
*   **Datetime-based File Selection**: Automatically determines the correct monthly Parquet file (e.g., `GLM_2024_10.parquet`) based on an input ISO 8601 datetime string.
*   **Data Filtering**: Filters data by `cluster_id` and `subcluster_id` columns.
*   **Flexible Output**: Returns filtered data in either JSON or CSV format.
*   **Containerized**: Ready for deployment using Docker.
*   **API Documentation**: Automatic interactive API documentation (Swagger UI) via FastAPI.

## Project Structure

## Development Setup

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Running with Docker
1. Build the Docker image
bash
docker build -t parquet-statistics-endpoint .
2. Run the Docker container
bash
docker run -p 8000:8000 parquet-statistics-endpoint
The API will be available at http://localhost:8000.
