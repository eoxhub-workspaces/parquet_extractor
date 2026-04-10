import pandas as pd
from datetime import datetime, timezone
import io
import logging
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from typing import Optional, List
import json
import geopandas as gpd
from shapely.geometry import box
import pystac
import s3fs
import shapely.wkb
import shapely.geometry as shapely_geometry
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def _construct_parquet_url(base_url: str, datetime_str: str) -> str:
    """
    Constructs the full S3 URL for the Parquet file based on the datetime string.
    Example: base_url="...", datetime_str="2025-11-01 00:46:18.481660160+00:00" -> "..._2025_11.parquet"
    """
    try:
        dt = pd.to_datetime(datetime_str)
        year = dt.strftime("%Y")
        month = str(dt.month)
        day = dt.strftime("%d")
        return f"{base_url}_{year}_{month}.parquet"
    except ValueError as e:
        logger.error(f"Invalid datetime string format: {datetime_str}. Error: {e}")
        raise ValueError(f"Invalid datetime string format: {datetime_str}. Expected ISO 8601 format (e.g., '2024-10-26T10:00:00Z').")

async def get_filtered_parquet_data(
    datetime_str: str,
    parent_cluster_id: int,
    cluster_id: int,
    earthcare_id: str,
    parquet_base_url: str,
    output_format: str = "json",
    columns_to_extract: Optional[List[str]] = None,
    exact_parquet_url: Optional[str] = None
) -> str:
    """
    Fetches, filters, and returns data from a Parquet file on S3.

    Args:
        datetime_str: A datetime string (e.g., "2025-11-01 00:46:18.481660160+00:00") to determine the Parquet file.
        parent_cluster_id: The parent cluster ID to filter by.
        cluster_id: The cluster ID to filter by.
        earthcare_id: The EarthCARE ID to filter by.
        parquet_base_url: The base URL for the Parquet files.
        output_format: The desired output format ("json", "csv", or "geojson").
        columns_to_extract: An optional list of column names to extract.
        exact_parquet_url: If provided, bypasses URL construction and directly uses this URL.

    Returns:
        A string containing the filtered data in the specified format.
    """
    # Check if the absolute URL was provided first
    if exact_parquet_url:
        parquet_url = exact_parquet_url
        logger.info(f"Using explicitly provided Parquet URL: {parquet_url}")
    else:
        # Fall back to constructing it if no exact URL was given
        parquet_url = _construct_parquet_url(parquet_base_url, datetime_str)
        logger.info(f"Constructed Parquet URL: {parquet_url}")

    try:
        parsed_url = urlparse(parquet_url)
        filesystem = None
        source = parquet_url
        
        # Use s3fs for s3:// URLs or for http(s) URLs pointing to S3-compatible storage.
        if parsed_url.scheme == 's3' or 's3' in parsed_url.netloc:
            endpoint_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            source = parsed_url.path.lstrip('/')
            filesystem = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': endpoint_url})
            logger.info(f"Using s3fs for path: '{source}' at endpoint: '{endpoint_url}'")
        else:
            logger.info(f"Using pyarrow's native handler for URL: '{source}'")
       
        # FIX: Pass 'source' as the first positional argument and define format
        dataset = ds.dataset(
            source,
            filesystem=filesystem,
            format="parquet"
        )
        
        # Define the filter expression. This will be pushed down to the file scan.
        if parent_cluster_id == -1:
            filter_expression = (
            (ds.field('earthcare_id') == earthcare_id) &
            (ds.field('cluster_id') == cluster_id)
        )
        else:
            filter_expression = (
                (ds.field('earthcare_id') == earthcare_id) &
                (ds.field('parent_cluster_id') == parent_cluster_id) & 
                (ds.field('cluster_id') == cluster_id)
            )
        # Safeguard: Expand any comma-separated strings in the columns list
        if columns_to_extract:
            expanded_columns = []
            for item in columns_to_extract:
                # Split by comma and strip any accidental whitespace
                expanded_columns.extend([col.strip() for col in item.split(',')])
            columns_to_extract = expanded_columns
        
        table = dataset.to_table(filter=filter_expression, columns=columns_to_extract)
        filtered_df = table.to_pandas()

        downloaded_mb = table.nbytes / (1024 * 1024)
        
        logger.info(f"Successfully read and filtered Parquet file. Rows returned: {len(filtered_df)}, Data loaded: {downloaded_mb:.2f} MB")
    
    except Exception as e:
        logger.error(f"Failed to read Parquet file from {parquet_url}. Error: {e}")
        raise RuntimeError(f"Could not read Parquet file: {parquet_url}. Please check the URL and permissions. Error: {e}")

    if filtered_df.empty:
        logger.warning(f"No data found for parent_cluster_id={parent_cluster_id}, cluster_id={cluster_id}, earthcare_id='{earthcare_id}' in {parquet_url}")
        if output_format in ["json", "geojson"]:
            return "[]"
        else: 
            return ""

    def _sanitize_value(value, column_name):
        """Helper to decode bytes and clean strings for JSON serialization."""
        if isinstance(value, bytes):
            if column_name == 'geometry':
                try:
                    return shapely_geometry.mapping(shapely.wkb.loads(value))
                except Exception:
                    return "invalid_geometry_data"
            return value.decode('utf-8', 'replace')
        return str(value) if pd.notna(value) else None

    if output_format == "json":
        # Sanitize object columns to prevent UTF-8 encoding errors
        for col in filtered_df.select_dtypes(include=['object']).columns:
            filtered_df[col] = filtered_df[col].apply(lambda x: _sanitize_value(x, col))
        
        return filtered_df.to_json(orient="records", date_format="iso", force_ascii=False)
    
    elif output_format == "csv":
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
    
    elif output_format == "geojson":
        features = []
        for _, row in filtered_df.iterrows():
            properties = row.to_dict()
            geometry_wkb = properties.pop('geometry', None)
            
            if geometry_wkb:
                try:
                    geom = shapely.wkb.loads(geometry_wkb)
                    geometry = shapely_geometry.mapping(geom)
                except Exception:
                    geometry = None
            else:
                geometry = None

            sanitized_properties = {k: _sanitize_value(v, k) for k, v in properties.items()}

            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": sanitized_properties
            })
        
        feature_collection = {"type": "FeatureCollection", "features": features}
        return json.dumps(feature_collection)
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}. Must be 'json', 'csv', or 'geojson'.")

async def get_geojson_from_parquet_url(
    parquet_url: str,
    start_time: str,
    end_time: str,
    columns_to_extract: Optional[List[str]] = None
) -> str:
    """
    Fetches and filters a single Parquet file by a time range and returns GeoJSON.
    """
    logger.info(f"Reading single Parquet URL: {parquet_url}")

    try:
        parsed_url = urlparse(parquet_url)
        filesystem = None
        source = parquet_url

        if parsed_url.scheme == 's3' or 's3' in parsed_url.netloc:
            endpoint_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            source = parsed_url.path.lstrip('/')
            filesystem = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': endpoint_url})
        
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))

        dataset = ds.dataset(source, filesystem=filesystem, format="parquet")
        
        filter_expression = (
            (ds.field('peak_datetime') >= start_dt) &
            (ds.field('peak_datetime') < end_dt)
        )

        # Ensure geometry is always included for GeoJSON
        final_columns = columns_to_extract
        if columns_to_extract:
            if 'geometry' not in columns_to_extract:
                final_columns = columns_to_extract + ['geometry']
        
        table = dataset.to_table(filter=filter_expression, columns=final_columns)
        df = table.to_pandas()

    except Exception as e:
        logger.error(f"Failed to read or filter Parquet file from {parquet_url}. Error: {e}")
        raise RuntimeError(f"Could not process Parquet file: {parquet_url}. Error: {e}")

    if df.empty:
        return json.dumps({"type": "FeatureCollection", "features": []})

    features = []
    for _, row in df.iterrows():
        properties = row.to_dict()
        geometry_wkb = properties.pop('geometry', None)
        
        if geometry_wkb:
            try:
                geom = shapely.wkb.loads(geometry_wkb)
                geometry = shapely_geometry.mapping(geom)
            except Exception:
                geometry = None
        else:
            geometry = None

        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {k: str(v) if pd.notna(v) else None for k, v in properties.items()}
        })

    return json.dumps({"type": "FeatureCollection", "features": features})

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
        filesystem = None
        source = parquet_url
        
        # Determine if it's an S3-like URL and construct the filesystem object
        if parsed_url.scheme == 's3' or 's3' in parsed_url.netloc:
            endpoint_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            source = parsed_url.path.lstrip('/')
            filesystem = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': endpoint_url})
            parquet_file = pq.ParquetFile(source, filesystem=filesystem)
        else:
            # Fallback for local files or other schemes pyarrow might handle natively
            parquet_file = pq.ParquetFile(parquet_url)
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

async def get_stac_geoparquet_catalog(
    parquet_url: str,
    service_base_url: str,
    ) -> bytes:
    """
    Generates an items GeoParquet for monthly items based on provided parquet.
    """
    try:
        # Use existing metadata function to get time range
        metadata = get_parquet_metadata(parquet_url, columns_to_inspect=['peak_datetime'])
        style_url = "https://workspace-ui-public.gtif-austria.hub-otc.eox.at/api/public/share/public-4wazei3y-02/assets/stormtracker_style.json"
        # Find min/max from all row groups
        all_min = []
        all_max = []
        for group in metadata['row_group_statistics']:
            for col in group['columns']:
                if col['column_name'] == 'peak_datetime' and 'statistics' in col:
                    all_min.append(col['statistics']['min'])
                    all_max.append(col['statistics']['max'])

        if not all_min or not all_max:
            raise ValueError("Could not determine temporal extent from 'peak_datetime' column.")

        # The dates from metadata can be strings, so they need to be parsed.
        # We also need to handle cases where they might already be datetime objects.
        parsed_min_dates = [datetime.fromisoformat(d.replace('Z', '+00:00')) if isinstance(d, str) else d for d in all_min]
        parsed_max_dates = [datetime.fromisoformat(d.replace('Z', '+00:00')) if isinstance(d, str) else d for d in all_max]

        start_dt = min(parsed_min_dates)
        end_dt = max(parsed_max_dates)

        items = []
        # Generate monthly STAC Items
        for dt in pd.date_range(start_dt.replace(day=1), end_dt, freq='MS'):
            # Set time to midnight UTC for start of month
            item_start_time = dt.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            # Set time to midnight UTC for end of month (start of next month)
            item_end_time = (dt + pd.offsets.MonthEnd(1)).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
            geom = box(-180, -90, 180, 90)
            item = pystac.Item(
                id=f"month-{dt.year}-{dt.month:02d}",
                geometry=geom,
                bbox=list(geom.bounds),
                datetime=item_start_time,
                properties={
                    # Format to ISO 8601 with milliseconds and 'Z' for UTC
                    "start_datetime": item_start_time.isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
                    "end_datetime": item_end_time.isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
                },
            )

            # Add asset pointing to the new GeoJSON endpoint
            asset_href = f"{service_base_url}/data/geojson?parquet_url={parquet_url}&start_time={item_start_time.isoformat(timespec='milliseconds').replace('+00:00', 'Z')}&end_time={item_end_time.isoformat(timespec='milliseconds').replace('+00:00', 'Z')}"
            item.add_asset(
                key="geojson_data",
                asset=pystac.Asset(
                    href=asset_href,
                    media_type="application/geo+json",
                    roles=["data"]
                )
            )
            # The links property is part of the item itself, not a separate list to append.
            item.add_link(
                pystac.Link(
                    rel="style",
                    target=style_url,
                    media_type="application/json",
                    extra_fields={'asset:keys': ['geojson_data']}
                )
            )
            items.append(item)

        # Convert STAC items to dictionaries
        item_dicts = [item.to_dict() for item in items]

        # Manually process dictionaries to ensure all top-level STAC fields become columns
        for item_dict in item_dicts:
            # Promote 'properties' to the top level
            properties = item_dict.pop('properties', {})
            item_dict.update(properties)

            # Convert 'bbox' array to a dictionary so Parquet writes it as a Struct
            if 'bbox' in item_dict and isinstance(item_dict['bbox'], list):
                b = item_dict['bbox']
                if len(b) == 4:
                    item_dict['bbox'] = {
                        'xmin': b[0], 
                        'ymin': b[1], 
                        'xmax': b[2], 
                        'ymax': b[3]
                    }
                elif len(b) == 6: # Fallback just in case you ever have 3D bounding boxes
                    item_dict['bbox'] = {
                        'xmin': b[0], 'ymin': b[1], 'zmin': b[2], 
                        'xmax': b[3], 'ymax': b[4], 'zmax': b[5]
                    }

        # Manually create the GeoDataFrame to ensure all STAC fields are included as columns.
        geometries = [
            shapely_geometry.shape(item.get('geometry')) if item.get('geometry') else None
            for item in item_dicts
        ]
        
        gdf = gpd.GeoDataFrame(item_dicts, geometry=geometries, crs="EPSG:4326")

        # Drop the original GeoJSON dict geometry from STAC to avoid serialization conflicts, 
        # GeoPandas is now managing the active Shapely geometries.
        if 'geometry' in gdf.columns:
            gdf['geometry'] = geometries

        # Convert datetime column to a proper timestamp type for Parquet
        gdf['datetime'] = pd.to_datetime(gdf['datetime'])

        # --- WKB GEOMETRY HANDLING ---
        # GeoPandas automatically serializes the active geometry to WKB when calling to_parquet().
        # If you need strict GeoParquet compliance, do nothing else here.
        #
        # ONLY uncomment the line below if you want to bypass standard GeoParquet metadata 
        # and force the column into raw bytes for a standard PyArrow Parquet file:
        #
        # gdf['geometry'] = gdf.geometry.to_wkb()

        # Write to an in-memory Parquet buffer
        buffer = io.BytesIO()
        gdf.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Failed to generate STAC catalog for {parquet_url}. Error: {e}")
        raise RuntimeError(f"Could not generate STAC catalog. Error: {e}")