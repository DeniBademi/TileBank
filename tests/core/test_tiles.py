"""
Tests for the core tiles module
"""

import os
import numpy as np
import pytest
from rasterio.crs import CRS

from tilebank.core.tiles import (
    save_tile_as_npy,
    extract_patch,
    create_tile_record,
    load_tile_from_tif
)

def test_save_tile_as_npy(test_data_dir, sample_tile_data):
    """Test saving a tile as NPY file"""
    # Save the tile
    saved_path = save_tile_as_npy(sample_tile_data, test_data_dir)
    
    # Check that file exists and has .npy extension
    assert os.path.exists(saved_path)
    assert saved_path.endswith('.npy')
    
    # Load and verify data
    loaded_data = np.load(saved_path)
    np.testing.assert_array_equal(loaded_data, sample_tile_data)

def test_extract_patch():
    """Test extracting patches from arrays"""
    # Test 3D array (bands, height, width)
    data_3d = np.random.rand(3, 100, 100)
    patch_3d = extract_patch(data_3d, 10, 20, 32)
    assert patch_3d.shape == (3, 32, 32)
    np.testing.assert_array_equal(patch_3d, data_3d[:, 10:42, 20:52])
    
    # Test 4D array (time, bands, height, width)
    data_4d = np.random.rand(5, 3, 100, 100)
    patch_4d = extract_patch(data_4d, 10, 20, 32)
    assert patch_4d.shape == (5, 3, 32, 32)
    np.testing.assert_array_equal(patch_4d, data_4d[..., 10:42, 20:52])

def test_create_tile_record(sample_bounds, sample_dimensions):
    """Test creating a tile record dictionary"""
    path = "/path/to/tile.npy"
    satellite_id = 1
    date_origin = "2024-03-15"
    srid = 4326
    
    record = create_tile_record(
        path=path,
        satellite_id=satellite_id,
        date_origin=date_origin,
        bounds=sample_bounds,
        dimensions=sample_dimensions,
        srid=srid
    )
    
    assert record["path"] == path
    assert record["satellite_id"] == satellite_id
    assert record["date_origin"] == date_origin
    assert record["min_lon"] == sample_bounds[0]
    assert record["min_lat"] == sample_bounds[1]
    assert record["max_lon"] == sample_bounds[2]
    assert record["max_lat"] == sample_bounds[3]
    assert record["width"] == sample_dimensions[0]
    assert record["height"] == sample_dimensions[1]
    assert record["srid"] == srid

def test_load_tile_from_tif(sample_tif_file):
    """Test loading and optionally reprojecting a tile from TIF"""
    # Test loading without reprojection
    data, metadata = load_tile_from_tif(sample_tif_file)
    assert isinstance(data, np.ndarray)
    assert data.ndim == 3  # (bands, height, width)
    assert len(metadata["bounds"]) == 4
    assert isinstance(metadata["crs"], CRS)
    assert metadata["width"] > 0
    assert metadata["height"] > 0
    
    # Test loading with reprojection
    target_crs = CRS.from_epsg(3857)  # Web Mercator
    data_proj, metadata_proj = load_tile_from_tif(sample_tif_file, target_crs=target_crs)
    assert isinstance(data_proj, np.ndarray)
    assert metadata_proj["crs"] == target_crs
    # Bounds should be different after reprojection
    assert metadata_proj["bounds"] != metadata["bounds"] 