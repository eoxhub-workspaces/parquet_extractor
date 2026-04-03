import pandas as pd
from datetime import datetime
import io
import logging
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from typing import Optional, List
import s3fs
import shapely.wkb
import shapely.geometry as shapely_geometry
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def _construct_parquet_url(base_url: str, datetime_str: str) -> str:
    """
    Constructs the full S3 URL for the Parquet file based on the datetime string.
    Example: base_url="...", datetime_str="2024-10-26T10:00:00Z" -> "..._2024_10.parquet"
    """
    try:
        # Handle ISO 8601 format, including 'Z' for UTC
        dt_obj = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        year = dt_obj.year
        month = dt_obj.month
        # Format: base_url_YYYY_MM.parquet
        parquet_filename = f"{base_url}_{year}_{month:02d}.parquet"
        return parquet_filename
    except ValueError as e:
        logger.error(f"Invalid datetime string format: {datetime_str}. Error: {e}")
        raise ValueError(f"Invalid datetime string format: {datetime_str}. Expected ISO 8601 format (e.g., '2024-10-26T10:00:00Z').")

async def get_filtered_parquet_data(
    datetime_str: str,
    parent_cluster_id: int,
    cluster_id: int,
    earthcare_id: str,
    parquet_base_url: str,
    output_format: str = "json"
) -> str:
    """
    Fetches, filters, and returns data from a Parquet file on S3.

    Args:
        datetime_str: A datetime string (e.g., "2024-10-26T10:00:00Z") to determine the Parquet file.
        parent_cluster_id: The parent cluster ID to filter by.
        cluster_id: The cluster ID to filter by.
        earthcare_id: The EarthCARE ID to filter by.
        parquet_base_url: The base S3 URL for the Parquet files (e.g., "https://s3.waw4-1.cloudferro.com/EarthCODE/OSCAssets/storm-data/EC_lightning_GLM").
        output_format: The desired output format ("json" or "csv").

    Returns:
        A string containing the filtered data in the specified format.
    """
    parquet_url = _construct_parquet_url(parquet_base_url, datetime_str)
    logger.info(f"Constructed Parquet URL: {parquet_url}")

    try:
        parsed_url = urlparse(parquet_url)
        
        # For S3-compatible storage, we must provide the endpoint URL.
        # We construct it from the scheme (http/https) and netloc (the domain).
        endpoint_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # The path for s3fs should not have a leading slash.
        s3_path = parsed_url.path.lstrip('/')
        
        # Create the filesystem object with the correct endpoint for this specific request.
        # anon=True is equivalent to the AWS CLI's --no-sign-request.
        s3 = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': endpoint_url})

        logger.info(f"Attempting to read S3 path: '{s3_path}' from endpoint: '{endpoint_url}'")

        # Use the pyarrow.dataset API for more robust predicate pushdown.
        # This API is designed to scan datasets and apply filters before loading data into memory,
        # which is more efficient for remote filesystems like S3.
        dataset = ds.dataset(
            s3_path,
            filesystem=s3,
            format="parquet"
        )
        
        # Define the filter expression. This will be pushed down to the file scan.
        filter_expression = (
            (ds.field('earthcare_id') == earthcare_id) &
            (ds.field('parent_cluster_id') == parent_cluster_id) & 
            (ds.field('cluster_id') == cluster_id)
        )
        
        table = dataset.to_table(filter=filter_expression)
        filtered_df = table.to_pandas()

        downloaded_mb = table.nbytes / (1024 * 1024)
        
        logger.info(f"Successfully read and filtered Parquet file. Rows returned: {len(filtered_df)}, Data loaded: {downloaded_mb:.2f} MB")
    except Exception as e:
        logger.error(f"Failed to read Parquet file from {parquet_url}. Error: {e}")
        raise RuntimeError(f"Could not read Parquet file: {parquet_url}. Please check the URL and S3 bucket permissions. Error: {e}")

    if filtered_df.empty:
        logger.warning(f"No data found for parent_cluster_id={parent_cluster_id}, cluster_id={cluster_id}, earthcare_id='{earthcare_id}' in {parquet_url}")
        if output_format == "json":
            return "[]"
        else: # csv
            # For CSV, an empty string or just headers might be returned.
            # Returning an empty string is simpler if no data.
            return ""

    def _sanitize_value(value, column_name):
        """Helper to decode bytes and clean strings for JSON serialization."""
        if isinstance(value, bytes):
            if column_name == 'geometry':
                try:
                    # Decode WKB to a shapely geometry, then to a GeoJSON-like dict
                    return shapely_geometry.mapping(shapely.wkb.loads(value))
                except Exception:
                    # If it's not valid WKB, return a placeholder
                    return "invalid_geometry_data"
            # For other byte columns, decode with replacement for bad characters
            return value.decode('utf-8', 'replace')
        # If it's not bytes (e.g., already a string, number), just convert to string
        return str(value) if pd.notna(value) else None

    if output_format == "json":
        # Sanitize object columns to prevent UTF-8 encoding errors with ujson
        for col in filtered_df.select_dtypes(include=['object']).columns:
            filtered_df[col] = filtered_df[col].apply(lambda x: _sanitize_value(x, col))
        
        return filtered_df.to_json(orient="records", date_format="iso", force_ascii=False)
    elif output_format == "csv":
        # Use io.StringIO to capture CSV output as a string
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
    else:
        raise ValueError(f"Unsupported output format: {output_format}. Must be 'json' or 'csv'.")


def _format_stat(value):
    """Helper to decode bytes for cleaner display."""
    if isinstance(value, bytes):
        try:
            # Attempt to decode as UTF-8 for a clean string representation
            return value.decode('utf-8')
        except UnicodeDecodeError:
            # If it fails, show the raw byte representation
            return repr(value)
    return value

def get_parquet_metadata(parquet_url: str, columns_to_inspect: Optional[List[str]] = None):
    """
    Reads a Parquet file from a URL (local or remote S3/HTTP) and extracts
    its metadata and row group statistics.
    """
    try:
        parsed_url = urlparse(parquet_url)
        
        # Determine if it's an S3-like URL and construct the filesystem object
        if parsed_url.scheme in ['s3', 'http', 'https'] and 's3' in parsed_url.netloc:
            endpoint_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            s3_path = parsed_url.path.lstrip('/')
            fs = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': endpoint_url})
            parquet_file = pq.ParquetFile(s3_path, filesystem=fs)
        else:
            # Fallback for local files or other schemes pyarrow might handle natively
            parquet_file = pq.ParquetFile(parquet_url)
        # (e.g., s3fs for s3://).
        # The FileMetaData object contains basic file-level metadata.
        metadata = parquet_file.metadata
        # The ParquetFile.schema_arrow property gives us the full pyarrow.Schema object we need.
        arrow_schema = parquet_file.schema_arrow

        # --- Prepare File Metadata ---
        file_metadata = {
            "total_rows": metadata.num_rows,
            "num_row_groups": metadata.num_row_groups,
            "schema": str(arrow_schema),
        }

        # --- Prepare Row Group Statistics ---
        row_groups_stats = []

        # If no columns are specified, inspect all columns.
        if not columns_to_inspect:
            columns_to_inspect = arrow_schema.names

        for i in range(metadata.num_row_groups):
            row_group_meta = metadata.row_group(i)
            group_info = {
                "row_group_id": i,
                "total_rows": row_group_meta.num_rows,
                "columns": []
            }

            for col_name in columns_to_inspect:
                col_info = {"column_name": col_name}
                try:
                    col_index = arrow_schema.get_field_index(col_name)
                    if col_index == -1:
                        col_info["status"] = "Not found in schema"
                    else:
                        column_meta = row_group_meta.column(col_index)
                        if column_meta.statistics:
                            stats = column_meta.statistics
                            col_info["statistics"] = {
                                "min": _format_stat(stats.min),
                                "max": _format_stat(stats.max),
                                "has_nulls": stats.has_null_count and stats.null_count > 0,
                                "null_count": stats.null_count,
                                "num_values": stats.num_values,
                            }
                        else:
                            col_info["status"] = "No statistics available"
                except Exception as col_e:
                    col_info["status"] = f"Error reading stats: {col_e}"

                group_info["columns"].append(col_info)

            row_groups_stats.append(group_info)

        return {
            "file_metadata": file_metadata,
            "row_group_statistics": row_groups_stats
        }

    except Exception as e:
        # Let the caller handle the HTTPException
        raise RuntimeError(f"An error occurred while inspecting Parquet metadata: {e}")
