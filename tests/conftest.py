"""
Common test fixtures for TileBank tests
"""

import os
import pytest
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return str(data_dir)

@pytest.fixture
def sample_tile_data():
    """Create a sample tile array"""
    return np.random.rand(3, 100, 100)  # 3 bands, 100x100 pixels

@pytest.fixture
def sample_tif_file(test_data_dir, sample_tile_data):
    """Create a sample GeoTIFF file"""
    filepath = os.path.join(test_data_dir, "sample.tif")
    
    # Sample geographical bounds
    bounds = (-122.5, 37.5, -122.0, 38.0)
    transform = from_bounds(*bounds, sample_tile_data.shape[2], sample_tile_data.shape[1])
    
    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=sample_tile_data.shape[1],
        width=sample_tile_data.shape[2],
        count=sample_tile_data.shape[0],
        dtype=sample_tile_data.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(sample_tile_data)
    
    return filepath

@pytest.fixture
def sample_bounds():
    """Sample geographical bounds"""
    return (-122.5, 37.5, -122.0, 38.0)  # (min_lon, min_lat, max_lon, max_lat)

@pytest.fixture
def sample_dimensions():
    """Sample tile dimensions"""
    return (100, 100)  # (width, height) 