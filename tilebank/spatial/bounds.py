"""
Functions for handling geographical bounds and coordinate transformations.
"""

from typing import Tuple, Optional, Union
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.transform import Affine

def get_tile_bounds(raster_path: str, target_crs: Optional[Union[str, CRS]] = None) -> Tuple[float, float, float, float]:
    """Get the geographical bounds of a raster file.
    
    Args:
        raster_path (str): Path to the raster file
        target_crs (Optional[Union[str, CRS]]): Target CRS to transform bounds to. If None, uses source CRS.
        
    Returns:
        Tuple[float, float, float, float]: (min_lon, min_lat, max_lon, max_lat)
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        
        if target_crs is not None:
            if isinstance(target_crs, str):
                target_crs = CRS.from_string(target_crs)
            bounds = transform_bounds(src.crs, target_crs, *bounds)
        
        return (bounds.left, bounds.bottom, bounds.right, bounds.top)

def calculate_patch_bounds(
    full_bounds: Tuple[float, float, float, float],
    full_shape: Tuple[int, int],
    patch_coords: Tuple[int, int],
    patch_size: int
) -> Tuple[float, float, float, float]:
    """Calculate the geographical bounds of a patch within a larger array.
    
    Args:
        full_bounds (Tuple[float, float, float, float]): Bounds of the full array (min_lon, min_lat, max_lon, max_lat)
        full_shape (Tuple[int, int]): Shape of the full array (height, width)
        patch_coords (Tuple[int, int]): Coordinates of the patch (i, j)
        patch_size (int): Size of the patch
        
    Returns:
        Tuple[float, float, float, float]: Bounds of the patch
    """
    min_lon, min_lat, max_lon, max_lat = full_bounds
    full_height, full_width = full_shape
    i, j = patch_coords
    
    # Calculate transform from pixel to geographical coordinates
    transform = Affine.from_gdal(
        min_lon, (max_lon - min_lon) / full_width, 0,
        max_lat, 0, -(max_lat - min_lat) / full_height
    )
    
    # Apply offset for this patch
    patch_transform = transform * Affine.translation(j, i)
    
    # Calculate bounds
    return rasterio.transform.array_bounds(patch_size, patch_size, patch_transform) 