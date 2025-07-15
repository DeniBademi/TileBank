"""
Tests for the spatial bounds module
"""

import pytest
from rasterio.crs import CRS

from tilebank.spatial.bounds import get_tile_bounds, calculate_patch_bounds

def test_get_tile_bounds(sample_tif_file):
    """Test getting bounds from a raster file"""
    # Test without CRS transformation
    bounds = get_tile_bounds(sample_tif_file)
    assert len(bounds) == 4
    min_lon, min_lat, max_lon, max_lat = bounds
    assert min_lon < max_lon
    assert min_lat < max_lat
    
    # Test with CRS transformation
    target_crs = CRS.from_epsg(3857)  # Web Mercator
    bounds_proj = get_tile_bounds(sample_tif_file, target_crs=target_crs)
    assert len(bounds_proj) == 4
    assert bounds_proj != bounds  # Bounds should be different after reprojection
    
    # Test with string CRS
    bounds_str = get_tile_bounds(sample_tif_file, target_crs="EPSG:3857")
    assert bounds_str == bounds_proj

def test_calculate_patch_bounds(sample_bounds):
    """Test calculating bounds for a patch within a larger array"""
    full_shape = (100, 100)  # (height, width)
    patch_size = 32
    
    # Test patch at origin (0, 0)
    origin_bounds = calculate_patch_bounds(
        full_bounds=sample_bounds,
        full_shape=full_shape,
        patch_coords=(0, 0),
        patch_size=patch_size
    )
    assert len(origin_bounds) == 4
    assert origin_bounds[0] == sample_bounds[0]  # min_lon should match
    assert origin_bounds[3] == sample_bounds[3]  # max_lat should match
    
    # Test patch in middle
    mid_i, mid_j = 34, 34
    mid_bounds = calculate_patch_bounds(
        full_bounds=sample_bounds,
        full_shape=full_shape,
        patch_coords=(mid_i, mid_j),
        patch_size=patch_size
    )
    assert len(mid_bounds) == 4
    assert sample_bounds[0] < mid_bounds[0] < sample_bounds[2]  # min_lon between full bounds
    assert sample_bounds[1] < mid_bounds[1] < sample_bounds[3]  # min_lat between full bounds
    
    # Test patch at bottom-right corner
    br_i, br_j = 68, 68
    br_bounds = calculate_patch_bounds(
        full_bounds=sample_bounds,
        full_shape=full_shape,
        patch_coords=(br_i, br_j),
        patch_size=patch_size
    )
    assert len(br_bounds) == 4
    assert br_bounds[2] <= sample_bounds[2]  # max_lon should not exceed full bounds
    assert br_bounds[1] >= sample_bounds[1]  # min_lat should not exceed full bounds 