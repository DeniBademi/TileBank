"""
Tests for the TileBankRepository class
"""

import os
import numpy as np
import pytest
from datetime import datetime
from rasterio.crs import CRS

from tilebank.repository.tile_bank import TileBankRepository

@pytest.fixture
def repo(test_data_dir):
    """Create a test repository"""
    db_path = os.path.join(test_data_dir, "test.db")
    return TileBankRepository(db_path=db_path, save_dir=test_data_dir)

@pytest.fixture
def sample_array():
    """Create a sample array for testing"""
    return np.random.rand(3, 512, 512)  # 3 bands, 512x512 pixels

def test_init(test_data_dir):
    """Test repository initialization"""
    db_path = os.path.join(test_data_dir, "test.db")
    repo = TileBankRepository(db_path=db_path, save_dir=test_data_dir)
    
    assert os.path.exists(db_path)
    assert os.path.exists(test_data_dir)
    assert repo.default_crs == CRS.from_epsg(4326)

def test_add_tiles_from_array(repo, sample_array, sample_bounds):
    """Test adding tiles from a large array"""
    date_origin = datetime.now()
    patch_size = 256
    
    # Test with default parameters
    tiles = repo.add_tiles_from_array(
        data=sample_array,
        satellite_name="Sentinel-2",
        date_origin=date_origin,
        bounds=sample_bounds,
        patch_size=patch_size
    )
    
    assert len(tiles) == 4  # 512x512 should create 4 256x256 tiles
    
    # Verify first tile
    first_tile = tiles[0]
    assert os.path.exists(first_tile['path'].values[0])
    assert first_tile['satellite_id'].values[0] > 0
    assert first_tile['date_origin'].values[0] == date_origin.strftime("%Y-%m-%d")
    assert first_tile['width'].values[0] == patch_size
    assert first_tile['height'].values[0] == patch_size
    
    # Test with validator
    def validator(patch):
        return np.mean(patch) > 0.4
    
    tiles_with_validator = repo.add_tiles_from_array(
        data=sample_array,
        satellite_name="Sentinel-2",
        date_origin=date_origin,
        bounds=sample_bounds,
        patch_size=patch_size,
        validator=validator
    )
    
    assert len(tiles_with_validator) <= 4  # Some tiles may be filtered out

def test_add_single_tile_from_path(repo, sample_tif_file):
    """Test adding a single tile from a file"""
    date_origin = "2024-03-15"
    
    # Test with TIF file
    tile = repo.add_single_tile_from_path(
        path=sample_tif_file,
        satellite_name="Sentinel-2",
        date_origin=date_origin
    )
    
    assert os.path.exists(tile['path'].values[0])
    assert tile['path'].values[0].endswith('.npy')
    assert tile['satellite_id'].values[0] > 0
    assert tile['date_origin'].values[0] == date_origin
    
    # Test exists_ok parameter
    with pytest.raises(Exception):
        repo.add_single_tile_from_path(
            path=sample_tif_file,
            satellite_name="Sentinel-2",
            date_origin=date_origin,
            exists_ok=False
        )
    
    # Should not raise with exists_ok=True
    tile2 = repo.add_single_tile_from_path(
        path=sample_tif_file,
        satellite_name="Sentinel-2",
        date_origin=date_origin,
        exists_ok=True
    )
    assert tile2 is not None

def test_create_mask_for_tile(repo, sample_tif_file):
    """Test creating a mask for a specific tile"""
    # First create a tile
    tile = repo.add_single_tile_from_path(
        path=sample_tif_file,
        satellite_name="Sentinel-2",
        date_origin="2024-03-15"
    )
    
    # Create a mask
    mask = repo.create_mask_for_tile(
        tile_id=tile['id'].values[0],
        mask_raster_path=sample_tif_file,  # Using same file as dummy mask
        task="land_cover",
        mask_type="binary",
        date_origin="2024-03-15"
    )
    
    assert os.path.exists(mask['path'].values[0])
    assert mask['path'].values[0].endswith('.npy')
    assert mask['tile_id'].values[0] == tile['id'].values[0]
    assert mask['task'].values[0] == "land_cover"
    assert mask['mask_type'].values[0] == "binary"
    assert mask['date_origin'].values[0] == "2024-03-15"

def test_create_masks_from_raster(repo, sample_tif_file):
    """Test creating masks for all overlapping tiles"""
    # First create some tiles
    date_origin = "2024-03-15"
    tile1 = repo.add_single_tile_from_path(
        path=sample_tif_file,
        satellite_name="Sentinel-2",
        date_origin=date_origin
    )
    
    # Create masks
    masks = repo.create_masks_from_raster(
        mask_raster_path=sample_tif_file,  # Using same file as dummy mask
        task="land_cover",
        mask_type="binary",
        date_origin=date_origin
    )
    
    assert len(masks) >= 1  # Should at least create mask for tile1
    
    # Verify first mask
    first_mask = masks[0]
    assert os.path.exists(first_mask['path'].values[0])
    assert first_mask['task'].values[0] == "land_cover"
    assert first_mask['mask_type'].values[0] == "binary"
    assert first_mask['date_origin'].values[0] == date_origin 