"""
Tests for the core masks module
"""

import os
import numpy as np
import pytest
from rasterio.crs import CRS

from tilebank.core.masks import (
    create_mask_record,
    extract_mask_for_tile
)

def test_create_mask_record(sample_bounds):
    """Test creating a mask record dictionary"""
    path = "/path/to/mask.npy"
    tile_id = 1
    task = "land_cover"
    mask_type = "binary"
    date_origin = "2024-03-15"
    srid = 4326
    
    record = create_mask_record(
        path=path,
        tile_id=tile_id,
        task=task,
        mask_type=mask_type,
        bounds=sample_bounds,
        date_origin=date_origin,
        srid=srid
    )
    
    assert record["path"] == path
    assert record["tile_id"] == tile_id
    assert record["task"] == task
    assert record["mask_type"] == mask_type
    assert record["date_origin"] == date_origin
    assert record["min_lon"] == sample_bounds[0]
    assert record["min_lat"] == sample_bounds[1]
    assert record["max_lon"] == sample_bounds[2]
    assert record["max_lat"] == sample_bounds[3]
    assert record["srid"] == srid

def test_extract_mask_for_tile(sample_tif_file, sample_bounds):
    """Test extracting mask data for a tile"""
    # Test with same CRS
    mask_data = extract_mask_for_tile(
        mask_raster_path=sample_tif_file,
        tile_bounds=sample_bounds,
        tile_crs=CRS.from_epsg(4326)
    )
    
    assert isinstance(mask_data, np.ndarray)
    assert mask_data.ndim == 3  # (bands, height, width)
    
    # Test with different CRS
    mask_data_proj = extract_mask_for_tile(
        mask_raster_path=sample_tif_file,
        tile_bounds=sample_bounds,
        tile_crs=CRS.from_epsg(3857)  # Web Mercator
    )
    
    assert isinstance(mask_data_proj, np.ndarray)
    assert mask_data_proj.shape == mask_data.shape  # Should maintain shape after reprojection