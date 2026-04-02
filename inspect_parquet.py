# inspect_parquet.py
import sys
import pyarrow.parquet as pq
import s3fs
from urllib.parse import urlparse

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

def inspect_parquet_metadata(parquet_url: str):
    """
    Connects to an S3-compatible service to read and print the metadata
    of a Parquet file, including row group statistics.
    """
    print(f"Inspecting: {parquet_url}\n")

    try:
        parsed_url = urlparse(parquet_url)
        endpoint_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        s3_path = parsed_url.path.lstrip('/')

        s3 = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': endpoint_url})

        # Open the parquet file and get the metadata
        parquet_file = pq.ParquetFile(s3_path, filesystem=s3)
        metadata = parquet_file.metadata

        print("--- File Metadata ---")
        print(f"Total Rows: {metadata.num_rows}")
        print(f"Number of Row Groups: {metadata.num_row_groups}")
        print(f"Schema:\n{metadata.schema}\n")

        print("--- Row Group Statistics ---")
        # Iterate through each row group
        for i in range(metadata.num_row_groups):
            row_group_meta = metadata.row_group(i)
            print(f"\n[Row Group {i}]")
            print(f"  - Total Rows: {row_group_meta.num_rows}")
            
            # Check for statistics on the columns we care about
            for col_name in ['parent_cluster_id', 'cluster_id', 'earthcare_id', 'latitude', 'longitude']:
                try:
                    # The pyarrow._parquet.ParquetSchema object might not have get_field_index directly.
                    # Iterate through the fields to find the positional index.
                    col_index = -1
                    for j, field in enumerate(metadata.schema):
                        if field.name == col_name:
                            col_index = j
                            break
                    column_meta = row_group_meta.column(col_index)
                    if column_meta.statistics:
                        stats = column_meta.statistics
                        min_val, max_val = _format_stat(stats.min), _format_stat(stats.max)
                        print(f"  - Column '{col_name}':")
                        print(f"    - Min: {min_val}")
                        print(f"    - Max: {max_val}")
                        print(f"    - Has Nulls: {stats.has_null_count and stats.null_count > 0}")
                    else:
                        print(f"  - Column '{col_name}': No statistics available.")
                except KeyError:
                    print(f"  - Column '{col_name}': Not found in this row group.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_parquet.py <full_parquet_s3_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    inspect_parquet_metadata(url)
