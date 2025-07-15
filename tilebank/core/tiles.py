"""
Core functionality for handling tiles.
"""

import os
import uuid
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple, Dict
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds, reproject, calculate_default_transform

from ..spatial.bounds import get_tile_bounds, calculate_patch_bounds

def save_tile_as_npy(data: np.ndarray, save_dir: str) -> str:
    """Save a tile as an NPY file with a unique name.
    
    Args:
        data (np.ndarray): The tile data
        save_dir (str): Directory to save in
        
    Returns:
        str: Path to the saved file
    """
    npy_path = os.path.join(save_dir, f"{uuid.uuid4()}.npy")
    np.save(npy_path, data)
    return npy_path

def extract_patch(data: np.ndarray, i: int, j: int, patch_size: int) -> np.ndarray:
    """Extract a patch from a larger array.
    
    Args:
        data (np.ndarray): Array of shape (bands, height, width) or (time, bands, height, width)
        i (int): Starting row
        j (int): Starting column
        patch_size (int): Size of the patch
        
    Returns:
        np.ndarray: The extracted patch
    """
    if len(data.shape) == 4:  # (time, bands, height, width)
        return data[..., i:i+patch_size, j:j+patch_size]
    else:  # (bands, height, width)
        return data[:, i:i+patch_size, j:j+patch_size]

def create_tile_record(
    path: str,
    satellite_id: int,
    date_origin: str,
    bounds: Tuple[float, float, float, float],
    dimensions: Tuple[int, int],
    srid: Optional[int] = None
) -> Dict:
    """Create a dictionary with tile record data.
    
    Args:
        path (str): Path to the tile file
        satellite_id (int): ID of the satellite
        date_origin (str): Date origin in YYYY-MM-DD format
        bounds (Tuple[float, float, float, float]): (min_lon, min_lat, max_lon, max_lat)
        dimensions (Tuple[int, int]): (width, height)
        srid (Optional[int]): SRID of the CRS
        
    Returns:
        Dict: Dictionary ready for database insertion
    """
    width, height = dimensions
    min_lon, min_lat, max_lon, max_lat = bounds
    
    return {
        "path": path,
        "satellite_id": satellite_id,
        "date_origin": date_origin,
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
        "width": width,
        "height": height,
        "srid": srid
    }

def load_tile_from_tif(path: str, target_crs: Optional[CRS] = None) -> Tuple[np.ndarray, Dict]:
    """Load a tile from a TIF file and optionally reproject it.
    
    Args:
        path (str): Path to the TIF file
        target_crs (Optional[CRS]): Target CRS to reproject to
        
    Returns:
        Tuple[np.ndarray, Dict]: The tile data and metadata
    """
    with rasterio.open(path) as src:
        data = src.read()
        bounds = src.bounds
        width = src.width
        height = src.height
        
        if target_crs and src.crs != target_crs:
            # Calculate the transform for the output
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, target_crs, width, height, *bounds)
            
            # Initialize the destination array
            dst_data = np.zeros((data.shape[0], dst_height, dst_width), dtype=data.dtype)
            
            # Reproject
            for band in range(data.shape[0]):
                reproject(
                    source=data[band],
                    destination=dst_data[band],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=rasterio.enums.Resampling.nearest
                )
            data = dst_data
            bounds = transform_bounds(src.crs, target_crs, *bounds)
            width = dst_width
            height = dst_height
        
        metadata = {
            "bounds": bounds,
            "width": width,
            "height": height,
            "crs": target_crs if target_crs else src.crs
        }
        
        return data, metadata 