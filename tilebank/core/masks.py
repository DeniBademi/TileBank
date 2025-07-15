"""
Core functionality for handling masks.
"""

import os
import uuid
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds, reproject, calculate_default_transform
from shapely.geometry import box
from rasterio.crs import CRS

from .tiles import save_tile_as_npy
from ..spatial.bounds import get_tile_bounds

def create_mask_record(
    path: str,
    tile_id: int,
    task: str,
    mask_type: str,
    bounds: Tuple[float, float, float, float],
    date_origin: Optional[str] = None,
    srid: Optional[int] = None
) -> Dict:
    """Create a dictionary with mask record data.
    
    Args:
        path (str): Path to the mask file
        tile_id (int): ID of the parent tile
        task (str): Type of mask task
        mask_type (str): Type of mask
        bounds (Tuple[float, float, float, float]): (min_lon, min_lat, max_lon, max_lat)
        date_origin (Optional[str]): Date origin in YYYY-MM-DD format
        srid (Optional[int]): SRID of the CRS
        
    Returns:
        Dict: Dictionary ready for database insertion
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    return {
        "path": path,
        "tile_id": tile_id,
        "task": task,
        "mask_type": mask_type,
        "date_origin": date_origin,
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
        "srid": srid
    }

def extract_mask_for_tile(
    mask_raster_path: str,
    tile_bounds: Tuple[float, float, float, float],
    tile_crs: CRS
) -> np.ndarray:
    """Extract a mask region matching a tile's extent.
    
    Args:
        mask_raster_path (str): Path to the mask raster
        tile_bounds (Tuple[float, float, float, float]): Bounds of the tile
        tile_crs (CRS): CRS of the tile
        
    Returns:
        np.ndarray: The extracted and reprojected mask data
    """
    if mask_raster_path.lower().endswith('.npy'):
        # Load NPY file
        mask_data = np.load(mask_raster_path)
        # TODO: Add metadata handling for NPY files (CRS, bounds, etc.)
        # For now, assume same CRS as tile
        return mask_data
    
    with rasterio.open(mask_raster_path) as src:
        # Transform tile bounds to mask CRS if needed
        if src.crs != tile_crs:
            mask_bounds = transform_bounds(tile_crs, src.crs, *tile_bounds)
        else:
            mask_bounds = tile_bounds
        
        # Convert bounds to pixel coordinates
        tile_box = box(*mask_bounds)
        mask_box = box(*src.bounds)
        
        if not tile_box.intersects(mask_box):
            raise ValueError(f"Tile does not overlap with mask raster")
        
        # Get the intersection
        intersection = tile_box.intersection(mask_box)
        bounds = intersection.bounds
        
        # Get pixel coordinates for the intersection
        window = Window.from_bounds(*bounds, src.transform)
        
        # Read data
        data = src.read(window=window)
        
        # Reproject if needed
        if src.crs != tile_crs:
            # Calculate the transform for the output
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, tile_crs, data.shape[1], data.shape[2], *bounds)
            
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
                    dst_crs=tile_crs,
                    resampling=rasterio.enums.Resampling.nearest
                )
            data = dst_data
        
        return data 