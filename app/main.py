from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List
import json
import logging
import io

from .services.parquet_service import (
    get_filtered_parquet_data, 
    get_parquet_metadata,
    get_geojson_from_parquet_url,
    get_stac_geoparquet_catalog
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Parquet Statistics Endpoint",
    description="API to extract, filter, and inspect data from Parquet files.",
    version="1.0.0",
)

@app.get("/")
async def read_root():
    """
    Root endpoint providing a welcome message and link to API documentation.
    """
    return {"message": "Welcome to the Parquet Statistics Endpoint! Use /docs for API documentation."}

@app.get("/data/filtered_parquet", summary="Get filtered data from Parquet files")
async def get_parquet_data(
    datetime_str: str = Query(
        ...,
        description="Datetime string in ISO 8601 format (e.g., '2024-10-26T10:00:00Z') to identify the Parquet file. Only the year and month are used."
    ),
    parent_cluster_id: int = Query(..., description="The parent cluster ID to filter the data."),
    cluster_id: int = Query(..., description="The cluster ID to filter the data."),
    earthcare_id: str = Query(..., description="The EarthCARE ID to filter the data."),
    parquet_base_url: str = Query(
        "https://s3.waw4-1.cloudferro.com/EarthCODE/OSCAssets/storm-data/EC_lightning_GLM",
        description="Base S3 URL for the Parquet files (e.g., 'https://s3.waw4-1.cloudferro.com/EarthCODE/OSCAssets/storm-data/GLM'). The year and month will be appended to this."
    ),
    columns: Optional[List[str]] = Query(
        None, 
        alias="columns_to_extract",
        description="Optional list of column names to extract."
    ),
    output_format: str = Query(
        "json",
        description="Desired output format: 'json', 'csv', or 'geojson'.",
        regex="^(json|csv|geojson)$"
    )
):
    """
    Retrieves and filters data from a monthly Parquet file stored on S3.
    The Parquet file is determined by the `datetime_str` (e.g., `GLM_2024_10.parquet`).
    Data is then filtered by `parent_cluster_id`, `cluster_id`, and `earthcare_id`.
    """
    try:
        data = await get_filtered_parquet_data(
            datetime_str=datetime_str,
            parent_cluster_id=parent_cluster_id,
            cluster_id=cluster_id,
            earthcare_id=earthcare_id,
            parquet_base_url=parquet_base_url,
            output_format=output_format,
            columns_to_extract=columns
        )

        if output_format in ["json", "geojson"]:
            # The service returns a JSON string, so we parse it back to a dict/list
            # for FastAPI to handle correctly.
            json_content = json.loads(data)
            return JSONResponse(content=json_content, media_type="application/json")
        elif output_format == "csv":
            # StreamingResponse is suitable for returning large CSV data
            return StreamingResponse(io.StringIO(data), media_type="text/csv")
        else:
            # This case should ideally be caught by the regex in Query, but as a fallback
            raise HTTPException(status_code=400, detail="Invalid output_format specified.")

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error during data processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    except Exception as e:
        logger.exception("An unexpected error occurred.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.get("/inspect/", summary="Inspect Parquet file metadata")
async def inspect_parquet_url(
    url: str = Query(..., description="The full URL to the Parquet file (e.g., `s3://...`, `http://...`, `file://...`)."),
    columns: Optional[List[str]] = Query(None, description="A list of column names to inspect. If not provided, all columns are inspected.")
):
    """
    Inspects a Parquet file from a given URL and returns its metadata and
    column-level statistics for each row group.

    - **url**: The full URL to the Parquet file.
    - **columns**: Optional query parameter to specify which columns to inspect.
      Can be provided multiple times (e.g., `?columns=col1&columns=col2`).
    """
    try:
        metadata = get_parquet_metadata(url, columns)
        return metadata
    except ValueError as e:
        logger.error(f"Validation error for URL {url}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error during Parquet inspection for URL {url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"An unexpected error occurred during inspection of {url}.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during inspection.")

@app.get("/data/geojson", summary="Get GeoJSON data from a Parquet file by time range")
async def get_geojson_data(
    parquet_url: str = Query(..., description="The full URL to the Parquet file."),
    start_time: str = Query(..., description="Start of the time range in ISO 8601 format (e.g., '2024-10-26T10:00:00Z')."),
    end_time: str = Query(..., description="End of the time range in ISO 8601 format (e.g., '2024-10-26T12:00:00Z')."),
    columns: Optional[List[str]] = Query(None, alias="columns_to_extract", description="Optional list of column names to extract.")
):
    """
    Directly queries a single Parquet file, filters the data by a specified
    time range on the `peak_datetime` column, and returns the result as a GeoJSON FeatureCollection.
    """
    try:
        geojson_data = await get_geojson_from_parquet_url(
            parquet_url=parquet_url,
            start_time=start_time,
            end_time=end_time,
            columns_to_extract=columns
        )
        # The service returns a GeoJSON string, so we parse it for a proper JSON response
        json_content = json.loads(geojson_data)
        return JSONResponse(content=json_content, media_type="application/geo+json")
    except ValueError as e:
        logger.error(f"Validation error for GeoJSON query on {parquet_url}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error during GeoJSON processing for {parquet_url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"An unexpected error occurred during GeoJSON query of {parquet_url}.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.get("/stac/geoparquet", summary="Generate a STAC Catalog for a GeoParquet file")
async def get_stac_catalog(
    request: Request,
    parquet_url: str = Query(..., description="The full URL to the Parquet file to be cataloged.")
):
    """
    Generates a STAC (SpatioTemporal Asset Catalog) for a given Parquet file.
    It creates monthly STAC Items, where each item's asset points to the
    `/data/geojson` endpoint to dynamically fetch data for that month.
    """
    try:
        # Construct the base URL of this service from the incoming request
        # This is needed to build the asset links in the STAC items.
        service_base_url = str(request.base_url).rstrip('/')
        
        parquet_bytes = await get_stac_geoparquet_catalog(
            parquet_url=parquet_url,
            service_base_url=service_base_url
        )
        
        # Return the generated Parquet file as a streaming response
        return StreamingResponse(
            io.BytesIO(parquet_bytes),
            media_type="application/x-parquet",
            headers={"Content-Disposition": "attachment; filename=stac_catalog.parquet"}
        )
    except ValueError as e:
        logger.error(f"Validation error for STAC generation on {parquet_url}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error during STAC generation for {parquet_url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception(f"An unexpected error occurred during STAC generation for {parquet_url}.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")