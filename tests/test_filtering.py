import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Add app directory to path to allow import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.parquet_service import get_geojson_from_parquet_url
from app.main import get_geojson_data, get_stac_catalog

class TestDynamicFiltering(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create a sample DataFrame to simulate DuckDB query output
        self.sample_data = pd.DataFrame({
            "peak_datetime": pd.to_datetime(["2024-10-26T10:30:00Z", "2024-10-26T11:00:00Z", "2024-10-26T11:30:00Z"]),
            "cluster_lightning": [10.0, 100.0, 1000.0],
            "cluster_area_km2": [50.0, 500.0, 5000.0],
            "duration_min": [10, 60, 120],
            "peak_lon": [10.0, 20.0, 30.0],
            "peak_lat": [40.0, 50.0, 60.0],
            "surface_type": ["land", "water", "land"],
            "earthcare_id": ["EC1", "EC2", "EC3"],
            "geometry": [bytearray(b'dummy_wkb1'), bytearray(b'dummy_wkb2'), bytearray(b'dummy_wkb3')]
        })

    @patch("app.services.parquet_service._open_duckdb_sync")
    @patch("app.services.parquet_service._normalise_wkb")
    @patch("shapely.wkb.loads")
    @patch("shapely.geometry.mapping")
    async def test_all_filters_default(self, mock_mapping, mock_loads, mock_normalise, mock_open_duckdb):
        # Mock DuckDB connection
        mock_con = MagicMock()
        mock_open_duckdb.return_value = mock_con
        
        # Mock schemas and query execution
        mock_con.execute.return_value.fetchall.return_value = [
            ("peak_datetime", "TIMESTAMP"),
            ("cluster_lightning", "DOUBLE"),
            ("cluster_area_km2", "DOUBLE"),
            ("duration_min", "INTEGER"),
            ("peak_lon", "DOUBLE"),
            ("peak_lat", "DOUBLE"),
            ("surface_type", "VARCHAR"),
            ("earthcare_id", "VARCHAR"),
            ("geometry", "BLOB")
        ]
        
        # We patch lambda or execute to return a DataFrame
        # Mock execution returning our dataframe
        with patch("asyncio.to_thread") as mock_to_thread:
            # We want asyncio.to_thread to return mock_con for _open_duckdb_sync,
            # then the list of schema rows, then the dataframe, then close
            mock_to_thread.side_effect = [
                mock_con,              # _open_duckdb_sync
                [                      # available columns fetch
                    ("peak_datetime",),
                    ("cluster_lightning",),
                    ("cluster_area_km2",),
                    ("duration_min",),
                    ("peak_lon",),
                    ("peak_lat",),
                    ("surface_type",),
                    ("earthcare_id",),
                    ("geometry",)
                ],
                self.sample_data.copy(), # query df() execution
                None                   # connection close
            ]
            
            mock_normalise.side_effect = lambda x: x
            mock_loads.return_value = "mock_geometry"
            mock_mapping.return_value = {"type": "Point", "coordinates": [0, 0]}

            # Call function under test with default filter values
            result_str = await get_geojson_from_parquet_url(
                parquet_url="s3://dummy/file.parquet",
                start_time="2024-10-26T10:00:00Z",
                end_time="2024-10-26T12:00:00Z"
            )
            
            result = json.loads(result_str)
            self.assertEqual(result["type"], "FeatureCollection")
            # Default filters should return all 3 features since they are within limits
            self.assertEqual(len(result["features"]), 3)

    @patch("app.services.parquet_service._open_duckdb_sync")
    @patch("app.services.parquet_service._normalise_wkb")
    @patch("shapely.wkb.loads")
    @patch("shapely.geometry.mapping")
    async def test_lightning_filters(self, mock_mapping, mock_loads, mock_normalise, mock_open_duckdb):
        mock_con = MagicMock()
        mock_open_duckdb.return_value = mock_con
        
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = [
                mock_con,
                [(c,) for c in self.sample_data.columns],
                self.sample_data.copy(),
                None
            ]
            mock_normalise.side_effect = lambda x: x
            mock_loads.return_value = "mock_geom"
            mock_mapping.return_value = {"type": "Point"}

            # Try lightning_min=50 and lightning_max=500
            result_str = await get_geojson_from_parquet_url(
                parquet_url="s3://dummy/file.parquet",
                start_time="2024-10-26T10:00:00Z",
                end_time="2024-10-26T12:00:00Z",
                lightning_min=50,
                lightning_max=500
            )
            result = json.loads(result_str)
            # Only index 1 (cluster_lightning = 100) satisfies 50 <= lightning <= 500
            self.assertEqual(len(result["features"]), 1)
            self.assertEqual(result["features"][0]["properties"]["cluster_lightning"], "100.0")

    @patch("app.services.parquet_service._open_duckdb_sync")
    @patch("app.services.parquet_service._normalise_wkb")
    @patch("shapely.wkb.loads")
    @patch("shapely.geometry.mapping")
    async def test_lightning_overflow_threshold(self, mock_mapping, mock_loads, mock_normalise, mock_open_duckdb):
        mock_con = MagicMock()
        mock_open_duckdb.return_value = mock_con
        
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = [
                mock_con,
                [(c,) for c in self.sample_data.columns],
                self.sample_data.copy(),
                None
            ]
            mock_normalise.side_effect = lambda x: x
            mock_loads.return_value = "mock_geom"
            mock_mapping.return_value = {"type": "Point"}

            # Try lightning_min=500, lightning_max=5000 (threshold/overflow check should bypass max check)
            result_str = await get_geojson_from_parquet_url(
                parquet_url="s3://dummy/file.parquet",
                start_time="2024-10-26T10:00:00Z",
                end_time="2024-10-26T12:00:00Z",
                lightning_min=500,
                lightning_max=5000  # Exactly 5000, upper bound check skipped, value 1000 and anything higher remains visible
            )
            result = json.loads(result_str)
            # Index 2 has lightning = 1000.0, which is >= 500. Since max is 5000, it is allowed (even if there was a value higher than 5000).
            self.assertEqual(len(result["features"]), 1)
            self.assertEqual(result["features"][0]["properties"]["cluster_lightning"], "1000.0")

    @patch("app.services.parquet_service._open_duckdb_sync")
    @patch("app.services.parquet_service._normalise_wkb")
    @patch("shapely.wkb.loads")
    @patch("shapely.geometry.mapping")
    async def test_area_filters(self, mock_mapping, mock_loads, mock_normalise, mock_open_duckdb):
        mock_con = MagicMock()
        mock_open_duckdb.return_value = mock_con
        
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = [
                mock_con,
                [(c,) for c in self.sample_data.columns],
                self.sample_data.copy(),
                None
            ]
            mock_normalise.side_effect = lambda x: x
            mock_loads.return_value = "mock_geom"
            mock_mapping.return_value = {"type": "Point"}

            # Try area_min=100 and area_max=1000
            result_str = await get_geojson_from_parquet_url(
                parquet_url="s3://dummy/file.parquet",
                start_time="2024-10-26T10:00:00Z",
                end_time="2024-10-26T12:00:00Z",
                area_min=100,
                area_max=1000
            )
            result = json.loads(result_str)
            # Only index 1 (area = 500) satisfies 100 <= area <= 1000
            self.assertEqual(len(result["features"]), 1)
            self.assertEqual(result["features"][0]["properties"]["cluster_area_km2"], "500.0")

    @patch("app.services.parquet_service._open_duckdb_sync")
    @patch("app.services.parquet_service._normalise_wkb")
    @patch("shapely.wkb.loads")
    @patch("shapely.geometry.mapping")
    async def test_duration_filters(self, mock_mapping, mock_loads, mock_normalise, mock_open_duckdb):
        mock_con = MagicMock()
        mock_open_duckdb.return_value = mock_con
        
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = [
                mock_con,
                [(c,) for c in self.sample_data.columns],
                self.sample_data.copy(),
                None
            ]
            mock_normalise.side_effect = lambda x: x
            mock_loads.return_value = "mock_geom"
            mock_mapping.return_value = {"type": "Point"}

            # Try duration_min=30 and duration_max=90
            result_str = await get_geojson_from_parquet_url(
                parquet_url="s3://dummy/file.parquet",
                start_time="2024-10-26T10:00:00Z",
                end_time="2024-10-26T12:00:00Z",
                duration_min=30,
                duration_max=90
            )
            result = json.loads(result_str)
            # Only index 1 (duration_min = 60) satisfies 30 <= duration_min <= 90
            self.assertEqual(len(result["features"]), 1)
            self.assertEqual(result["features"][0]["properties"]["duration_min"], "60")

    @patch("app.services.parquet_service._open_duckdb_sync")
    @patch("app.services.parquet_service._normalise_wkb")
    @patch("shapely.wkb.loads")
    @patch("shapely.geometry.mapping")
    async def test_coordinate_bounding_box_filters(self, mock_mapping, mock_loads, mock_normalise, mock_open_duckdb):
        mock_con = MagicMock()
        mock_open_duckdb.return_value = mock_con
        
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = [
                mock_con,
                [(c,) for c in self.sample_data.columns],
                self.sample_data.copy(),
                None
            ]
            mock_normalise.side_effect = lambda x: x
            mock_loads.return_value = "mock_geom"
            mock_mapping.return_value = {"type": "Point"}

            # lon: [10, 20, 30], lat: [40, 50, 60]
            # Bounding box filter: lon between 15 and 25, lat between 45 and 55
            result_str = await get_geojson_from_parquet_url(
                parquet_url="s3://dummy/file.parquet",
                start_time="2024-10-26T10:00:00Z",
                end_time="2024-10-26T12:00:00Z",
                lon_min=15,
                lon_max=25,
                lat_min=45,
                lat_max=55
            )
            result = json.loads(result_str)
            # Only index 1 satisfies both bounding box checks
            self.assertEqual(len(result["features"]), 1)
            self.assertEqual(result["features"][0]["properties"]["peak_lon"], "20.0")
            self.assertEqual(result["features"][0]["properties"]["peak_lat"], "50.0")

    @patch("app.services.parquet_service._open_duckdb_sync")
    @patch("app.services.parquet_service._normalise_wkb")
    @patch("shapely.wkb.loads")
    @patch("shapely.geometry.mapping")
    async def test_categorical_surface_and_earthcare_filters(self, mock_mapping, mock_loads, mock_normalise, mock_open_duckdb):
        mock_con = MagicMock()
        mock_open_duckdb.return_value = mock_con
        
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = [
                mock_con,
                [(c,) for c in self.sample_data.columns],
                self.sample_data.copy(),
                None
            ]
            mock_normalise.side_effect = lambda x: x
            mock_loads.return_value = "mock_geom"
            mock_mapping.return_value = {"type": "Point"}

            # Filter by surface_type = land and earthcare_id = EC3
            result_str = await get_geojson_from_parquet_url(
                parquet_url="s3://dummy/file.parquet",
                start_time="2024-10-26T10:00:00Z",
                end_time="2024-10-26T12:00:00Z",
                surface_type="land",
                earthcare_id="EC3"
            )
            result = json.loads(result_str)
            # Only index 2 (land, EC3) satisfies both
            self.assertEqual(len(result["features"]), 1)
            self.assertEqual(result["features"][0]["properties"]["earthcare_id"], "EC3")
            self.assertEqual(result["features"][0]["properties"]["surface_type"], "land")

class TestFastAPIRoutes(unittest.IsolatedAsyncioTestCase):
    @patch("app.main.get_geojson_from_parquet_url")
    async def test_geojson_endpoint_passes_filters(self, mock_service_func):
        # Configure mocked service response
        mock_service_func.return_value = json.dumps({"type": "FeatureCollection", "features": []})

        # Call the get_geojson_data endpoint directly (as route handlers are standard async functions)
        response = await get_geojson_data(
            parquet_url="s3://dummy/file.parquet",
            start_time="2024-10-26T10:00:00Z",
            end_time="2024-10-26T12:00:00Z",
            columns=["cluster_lightning"],
            lightning_min=10,
            lightning_max=100,
            area_min=5,
            area_max=200,
            duration_min=2,
            duration_max=30,
            lon_min=-10,
            lon_max=10,
            lat_min=35,
            lat_max=45,
            surface_type="land",
            earthcare_id="EC-123"
        )

        # Since it returns a JSONResponse, we parse its body
        body = json.loads(response.body.decode("utf-8"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, {"type": "FeatureCollection", "features": []})

        # Verify that the service function was called with all of those parsed query parameters!
        mock_service_func.assert_called_once_with(
            parquet_url="s3://dummy/file.parquet",
            start_time="2024-10-26T10:00:00Z",
            end_time="2024-10-26T12:00:00Z",
            columns_to_extract=["cluster_lightning"],
            lightning_min=10.0,
            lightning_max=100.0,
            area_min=5.0,
            area_max=200.0,
            duration_min=2.0,
            duration_max=30.0,
            lon_min=-10.0,
            lon_max=10.0,
            lat_min=35.0,
            lat_max=45.0,
            surface_type="land",
            earthcare_id="EC-123"
        )

    @patch("app.main.get_stac_geoparquet_catalog")
    async def test_stac_catalog_endpoint_passes_parameters(self, mock_get_stac_geoparquet_catalog):
        mock_get_stac_geoparquet_catalog.return_value = b"mock_parquet_bytes"

        # Mock the FastAPI request object
        mock_request = MagicMock()
        mock_request.base_url = "http://localhost:8000"

        # Call get_stac_catalog directly with custom parameters
        response = await get_stac_catalog(
            request=mock_request,
            parquet_url="s3://dummy/file.parquet",
            style_url="https://example.com/custom_style.json",
            split=False
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "application/x-parquet")

        # Verify that get_stac_geoparquet_catalog was called with the custom parameters!
        mock_get_stac_geoparquet_catalog.assert_called_once_with(
            parquet_url="s3://dummy/file.parquet",
            service_base_url="http://localhost:8000",
            style_url="https://example.com/custom_style.json",
            split=False
        )

if __name__ == '__main__':
    unittest.main()
