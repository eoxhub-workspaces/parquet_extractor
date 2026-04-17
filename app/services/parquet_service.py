import pandas as pd
from datetime import datetime, timezone
import io
import logging
import duckdb
from dateutil.parser import parse as dateutil_parse
from typing import Optional, List
import json
import geopandas as gpd
from shapely.geometry import box
from geopandas.array import to_wkb
import pystac
import shapely.wkb
import shapely.geometry as shapely_geometry
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _construct_parquet_url(base_url: str, datetime_str: str) -> str:
    """
    Constructs the full S3 URL for the Parquet file based on the datetime string.
    Example: base_url="...", datetime_str="2025-11-01 00:46:18.481660160+00:00" -> "..._2025_11.parquet"
    """
    try:
        dt = pd.to_datetime(datetime_str)
        year = dt.strftime("%Y")
        month = str(dt.month)
        return f"{base_url}_{year}_{month}.parquet"
    except ValueError as e:
        logger.error(f"Invalid datetime string format: {datetime_str}. Error: {e}")
        raise ValueError(
            f"Invalid datetime string format: {datetime_str}. "
            f"Expected ISO 8601 format (e.g., '2024-10-26T10:00:00Z')."
        )


def _configure_duckdb_for_url(con: duckdb.DuckDBPyConnection, parquet_url: str) -> None:
    """
    Installs/loads the httpfs extension and configures S3 credentials
    based on the URL scheme.
    """
    con.execute("INSTALL httpfs; LOAD httpfs;")

    parsed = urlparse(parquet_url)

    if parsed.scheme == "s3" or "s3" in parsed.netloc:
        con.execute("SET s3_use_ssl = true;")
        con.execute("SET s3_url_style = 'path';")

        if parsed.scheme in ("http", "https") and "s3" in parsed.netloc:
            con.execute(f"SET s3_endpoint = '{parsed.netloc}';")

        # Anonymous access
        con.execute("SET s3_access_key_id = '';")
        con.execute("SET s3_secret_access_key = '';")


def _open_duckdb(parquet_url: str) -> duckdb.DuckDBPyConnection:
    """Opens a fresh DuckDB connection configured for the given URL."""
    con = duckdb.connect()
    _configure_duckdb_for_url(con, parquet_url)
    return con


def _build_filter_clause(
    earthcare_id: str,
    parent_cluster_id: int,
    cluster_id: int,
) -> str:
    """Builds the SQL WHERE clause string."""
    clauses = [
        f"earthcare_id = '{earthcare_id}'",
        f"cluster_id = {cluster_id}",
    ]
    if parent_cluster_id != -1:
        clauses.append(f"parent_cluster_id = {parent_cluster_id}")
    return " AND ".join(clauses)


def _resolve_columns(
    con: duckdb.DuckDBPyConnection,
    parquet_url: str,
    columns_to_extract: Optional[List[str]],
) -> Optional[List[str]]:
    """
    Expands comma-separated column strings and only appends coordinate
    columns that actually exist in the remote Parquet schema.
    """
    available_columns = {
        row[0]
        for row in con.execute(
            f"SELECT name FROM parquet_schema('{parquet_url}')"
        ).fetchall()
    }
    logger.info(f"Remote schema columns: {available_columns}")

    if not columns_to_extract:
        return None

    expanded = []
    for item in columns_to_extract:
        expanded.extend([col.strip() for col in item.split(",")])

    required_coords = [
        "parallax_corrected_lon",
        "parallax_corrected_lat",
        "longitude",
        "latitude",
    ]
    for coord in required_coords:
        if coord not in expanded and coord in available_columns:
            expanded.append(coord)

    missing = [col for col in expanded if col not in available_columns]
    if missing:
        logger.warning(f"Requested columns not found in schema, dropping: {missing}")
        expanded = [col for col in expanded if col in available_columns]

    return expanded


def _normalise_wkb(series: pd.Series) -> pd.Series:
    """Cast bytearray → bytes so shapely/geopandas can consume the column."""
    return series.apply(lambda g: bytes(g) if isinstance(g, bytearray) else g)


def _sanitize_value(value, column_name: str):
    """Decode bytes and clean values for JSON serialisation."""
    if isinstance(value, (bytes, bytearray)):
        if column_name == "geometry":
            try:
                return shapely_geometry.mapping(shapely.wkb.loads(bytes(value)))
            except Exception:
                return "invalid_geometry_data"
        return bytes(value).decode("utf-8", "replace")
    return str(value) if pd.notna(value) else None


def _build_geometry(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a WKB geometry column from coordinate columns when present,
    overriding any existing geometry.  Falls back to normalising an
    existing geometry column that DuckDB returned as bytearray.
    """
    if (
        "parallax_corrected_lon" in filtered_df.columns
        and "parallax_corrected_lat" in filtered_df.columns
    ):
        logger.info("Creating geometry from parallax-corrected coordinates.")
        geometries = gpd.points_from_xy(
            filtered_df["parallax_corrected_lon"],
            filtered_df["parallax_corrected_lat"],
        )
        filtered_df["geometry"] = to_wkb(geometries)

    elif "longitude" in filtered_df.columns and "latitude" in filtered_df.columns:
        logger.info("Creating geometry from standard lat/lon.")
        geometries = gpd.points_from_xy(
            filtered_df["longitude"], filtered_df["latitude"]
        )
        filtered_df["geometry"] = to_wkb(geometries)

    # Normalise whatever geometry column exists (bytearray → bytes)
    if "geometry" in filtered_df.columns:
        filtered_df["geometry"] = _normalise_wkb(filtered_df["geometry"])

    return filtered_df


def _serialise(filtered_df: pd.DataFrame, output_format: str) -> str:
    """Serialises a DataFrame to the requested output format."""
    if output_format == "json":
        for col in filtered_df.select_dtypes(include=["object"]).columns:
            filtered_df[col] = filtered_df[col].apply(
                lambda x: _sanitize_value(x, col)
            )
        return filtered_df.to_json(orient="records", date_format="iso", force_ascii=False)

    elif output_format == "csv":
        return filtered_df.to_csv(index=False)

    elif output_format == "geojson":
        if "geometry" not in filtered_df.columns:
            logger.warning("GeoJSON requested but no geometry column present.")
            return filtered_df.to_json(orient="records")

        features = []
        for _, row in filtered_df.iterrows():
            props = row.to_dict()
            geom_raw = props.pop("geometry", None)
            geometry = None
            if geom_raw is not None:
                try:
                    geometry = shapely_geometry.mapping(
                        shapely.wkb.loads(bytes(geom_raw) if isinstance(geom_raw, bytearray) else geom_raw)
                    )
                except Exception:
                    pass
            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": {k: _sanitize_value(v, k) for k, v in props.items()},
            })
        return json.dumps({"type": "FeatureCollection", "features": features})

    else:
        raise ValueError(
            f"Unsupported output_format: '{output_format}'. Use 'json', 'csv', or 'geojson'."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_filtered_parquet_data(
    datetime_str: str,
    parent_cluster_id: int,
    cluster_id: int,
    earthcare_id: str,
    parquet_base_url: str,
    output_format: str = "json",
    columns_to_extract: Optional[List[str]] = None,
    exact_parquet_url: Optional[str] = None,
) -> str:
    """
    Fetches, filters, and returns data from a Parquet file using DuckDB.
    DuckDB issues HTTP range requests so only matching row groups are
    downloaded, rather than the entire file.
    """
    parquet_url = (
        exact_parquet_url
        if exact_parquet_url
        else _construct_parquet_url(parquet_base_url, datetime_str)
    )
    logger.info(f"Parquet URL: {parquet_url}")

    try:
        con = _open_duckdb(parquet_url)
        resolved_columns = _resolve_columns(con, parquet_url, columns_to_extract)
        select_clause = ", ".join(resolved_columns) if resolved_columns else "*"
        where_clause = _build_filter_clause(earthcare_id, parent_cluster_id, cluster_id)

        query = f"""
            SELECT {select_clause}
            FROM read_parquet('{parquet_url}')
            WHERE {where_clause}
        """
        logger.info(f"Executing query: {query}")
        filtered_df = con.execute(query).df()
        con.close()

        logger.info(
            f"Query complete. Rows: {len(filtered_df)}, "
            f"~{filtered_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        )
    except Exception as e:
        logger.error(f"DuckDB failed on {parquet_url}. Error: {e}")
        raise RuntimeError(
            f"Could not read Parquet file: {parquet_url}. "
            f"Please check the URL and permissions. Error: {e}"
        )

    if filtered_df.empty:
        logger.warning(
            f"No data for parent_cluster_id={parent_cluster_id}, "
            f"cluster_id={cluster_id}, earthcare_id='{earthcare_id}'"
        )
        return "[]" if output_format in ("json", "geojson") else ""

    filtered_df = _build_geometry(filtered_df)
    return _serialise(filtered_df, output_format)


async def get_geojson_from_parquet_url(
    parquet_url: str,
    start_time: str,
    end_time: str,
    columns_to_extract: Optional[List[str]] = None,
) -> str:
    """
    Fetches and filters a single Parquet file by a time range and returns GeoJSON.
    """
    logger.info(f"Reading Parquet URL: {parquet_url}")

    # Ensure geometry is always fetched if a column list is given
    if columns_to_extract and "geometry" not in columns_to_extract:
        columns_to_extract = columns_to_extract + ["geometry"]

    select_clause = ", ".join(columns_to_extract) if columns_to_extract else "*"

    try:
        con = _open_duckdb(parquet_url)

        # Validate that peak_datetime actually exists before filtering on it
        available = {
            row[0]
            for row in con.execute(
                f"SELECT name FROM parquet_schema('{parquet_url}')"
            ).fetchall()
        }
        if "peak_datetime" not in available:
            raise ValueError(
                f"Column 'peak_datetime' not found in {parquet_url}. "
                f"Available: {available}"
            )

        query = f"""
            SELECT {select_clause}
            FROM read_parquet('{parquet_url}')
            WHERE peak_datetime >= TIMESTAMPTZ '{start_time}'
              AND peak_datetime <  TIMESTAMPTZ '{end_time}'
        """
        logger.info(f"Executing query: {query}")
        df = con.execute(query).df()
        con.close()

    except Exception as e:
        logger.error(f"Failed to read/filter {parquet_url}. Error: {e}")
        raise RuntimeError(f"Could not process Parquet file: {parquet_url}. Error: {e}")

    if df.empty:
        return json.dumps({"type": "FeatureCollection", "features": []})

    if "geometry" in df.columns:
        df["geometry"] = _normalise_wkb(df["geometry"])

    features = []
    for _, row in df.iterrows():
        props = row.to_dict()
        geom_raw = props.pop("geometry", None)
        geometry = None
        if geom_raw is not None:
            try:
                geometry = shapely_geometry.mapping(shapely.wkb.loads(geom_raw))
            except Exception:
                pass
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {k: str(v) if pd.notna(v) else None for k, v in props.items()},
        })

    return json.dumps({"type": "FeatureCollection", "features": features})


def get_parquet_metadata(
    parquet_url: str,
    columns_to_inspect: Optional[List[str]] = None,
) -> dict:
    """
    Reads schema and row-group statistics from a Parquet file using DuckDB.
    Uses parquet_schema() for column info and parquet_metadata() for row-group stats.
    """
    try:
        con = _open_duckdb(parquet_url)

        # --- Schema ---
        schema_rows = con.execute(
            f"SELECT name, logical_type FROM parquet_schema('{parquet_url}')"
        ).fetchall()
        all_columns = [row[0] for row in schema_rows]
        schema_str = "\n".join(f"{row[0]}: {row[1]}" for row in schema_rows)

        inspect = columns_to_inspect if columns_to_inspect else all_columns

        # --- Row-group metadata ---
        meta_df = con.execute(
            f"SELECT * FROM parquet_metadata('{parquet_url}')"
        ).df()
        con.close()

        # Basic file-level info
        num_row_groups = int(meta_df["row_group_id"].max()) + 1 if not meta_df.empty else 0
        total_rows = (
            meta_df[meta_df["row_group_id"] == 0]["row_group_num_rows"].sum()
            if not meta_df.empty else 0
        )

        file_metadata = {
            "total_rows": int(total_rows),
            "num_row_groups": num_row_groups,
            "schema": schema_str,
        }
        
        # HELPER: Converts NumPy/Pandas scalars to native Python types for JSON serialization
        def to_native(val):
            if pd.isna(val):
                return None
            return val.item() if hasattr(val, "item") else val

        # Per-row-group stats
        row_groups_stats = []
        for rg_id in range(num_row_groups):
            rg_df = meta_df[meta_df["row_group_id"] == rg_id]
            num_rows = int(rg_df["row_group_num_rows"].iloc[0]) if not rg_df.empty else 0
            group_info = {
                "row_group_id": rg_id,
                "total_rows": num_rows,
                "columns": [],
            }

            for col_name in inspect:
                col_rows = rg_df[rg_df["path_in_schema"] == col_name]
                col_info: dict = {"column_name": col_name}

                if col_rows.empty:
                    col_info["status"] = "Not found in row group metadata"
                else:
                    r = col_rows.iloc[0]
                    
                    # Apply the helper here to strip away numpy.int64, pd.NA, etc.
                    stats_min = to_native(r.get("stats_min_value") or r.get("stats_min"))
                    stats_max = to_native(r.get("stats_max_value") or r.get("stats_max"))
                    null_count = to_native(r.get("null_count"))
                    num_values = to_native(r.get("num_values"))

                    if stats_min is not None or stats_max is not None:
                        col_info["statistics"] = {
                            "min": stats_min,
                            "max": stats_max,
                            "has_nulls": bool(null_count) if null_count is not None else None,
                            "null_count": null_count,
                            "num_values": num_values,
                        }
                    else:
                        col_info["status"] = "No statistics available"

                group_info["columns"].append(col_info)

            row_groups_stats.append(group_info)

        return {
            "file_metadata": file_metadata,
            "row_group_statistics": row_groups_stats,
        }

    except Exception as e:
        raise RuntimeError(
            f"An error occurred while inspecting Parquet metadata: {e}"
        )

async def get_stac_geoparquet_catalog(
    parquet_url: str,
    service_base_url: str,
) -> bytes:
    """
    Generates an items GeoParquet for monthly items based on the provided parquet.
    """
    try:
        metadata = get_parquet_metadata(parquet_url, columns_to_inspect=["peak_datetime"])
        style_url = "https://workspace-ui-public.gtif-austria.hub-otc.eox.at/api/public/share/public-4wazei3y-02/assets/stormtracker_style.json"

        all_min, all_max = [], []
        for group in metadata["row_group_statistics"]:
            for col in group["columns"]:
                if col["column_name"] == "peak_datetime" and "statistics" in col:
                    all_min.append(col["statistics"]["min"])
                    all_max.append(col["statistics"]["max"])

        if not all_min or not all_max:
            raise ValueError("Could not determine temporal extent from 'peak_datetime'.")

        parsed_min = [dateutil_parse(d) if isinstance(d, str) else d for d in all_min]
        parsed_max = [dateutil_parse(d) if isinstance(d, str) else d for d in all_max]
        start_dt = min(parsed_min)
        end_dt = max(parsed_max)

        items = []
        current_end = end_dt.replace(
            hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc
        )

        while current_end > start_dt.replace(tzinfo=timezone.utc):
            current_start = current_end - pd.DateOffset(months=6)
            item_start_time = max(
                current_start,
                start_dt.replace(
                    hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
                ),
            )
            item_end_time = current_end

            geom = box(-180, -90, 180, 90)
            item = pystac.Item(
                id=f"{item_end_time.year}-{item_end_time.month}",
                geometry=geom,
                bbox=list(geom.bounds),
                datetime=item_start_time,
                properties={
                    "start_datetime": item_start_time.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                    "end_datetime": item_end_time.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                },
            )

            asset_href = (
                f"{service_base_url}/data/geojson"
                f"?parquet_url={parquet_url}"
                f"&start_time={item_start_time.isoformat(timespec='milliseconds').replace('+00:00', 'Z')}"
                f"&end_time={item_end_time.isoformat(timespec='milliseconds').replace('+00:00', 'Z')}"
            )
            item.add_asset(
                key="geojson_data",
                asset=pystac.Asset(
                    href=asset_href,
                    media_type="application/geo+json",
                    roles=["data"],
                ),
            )
            item.add_link(
                pystac.Link(
                    rel="style",
                    target=style_url,
                    media_type="application/json",
                    extra_fields={"asset:keys": ["geojson_data"]},
                )
            )
            items.append(item)
            current_end = current_start

        item_dicts = [item.to_dict() for item in items]
        for item_dict in item_dicts:
            properties = item_dict.pop("properties", {})
            item_dict.update(properties)
            if "bbox" in item_dict and isinstance(item_dict["bbox"], list):
                b = item_dict["bbox"]
                item_dict["bbox"] = (
                    {"xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3]}
                    if len(b) == 4
                    else {"xmin": b[0], "ymin": b[1], "zmin": b[2], "xmax": b[3], "ymax": b[4], "zmax": b[5]}
                )

        geometries = [
            shapely_geometry.shape(d.get("geometry")) if d.get("geometry") else None
            for d in item_dicts
        ]
        gdf = gpd.GeoDataFrame(item_dicts, geometry=geometries, crs="EPSG:4326")
        if "geometry" in gdf.columns:
            gdf["geometry"] = geometries
        gdf["datetime"] = pd.to_datetime(gdf["datetime"], format="ISO8601")

        buffer = io.BytesIO()
        gdf.to_parquet(buffer, index=False)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        logger.error(f"Failed to generate STAC catalog for {parquet_url}. Error: {e}")
        raise RuntimeError(f"Could not generate STAC catalog. Error: {e}")