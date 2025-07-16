import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from shapely.geometry import box

def get_raster_spatial_info(raster_path: str) -> dict:
    """Extract spatial information from a raster file.

    Args:
        raster_path (str): Path to the raster file

    Returns:
        dict: Dictionary containing spatial information
    """
    with rasterio.open(raster_path) as src:
        bounds = box(*src.bounds)
        # Convert to WKT format for DuckDB
        bounds_wkt = bounds.wkt
        return {
            "crs": src.crs.to_string(),
            "bounds": bounds_wkt,  # Store as WKT for DuckDB
            "pixel_size_x": src.transform[0],
            "pixel_size_y": abs(src.transform[4]),  # Usually negative, we store absolute
            "width": src.width,
            "height": src.height
        }

def get_array_spatial_info(array: np.ndarray, crs: str, transform: tuple) -> dict:
    """Extract spatial information from a numpy array and its geospatial metadata.

    Args:
        array (np.ndarray): The array data
        crs (str): Coordinate reference system (e.g. 'EPSG:4326')
        transform (tuple): Affine transform (a, b, c, d, e, f)

    Returns:
        dict: Dictionary containing spatial information
    """
    height, width = array.shape[-2:]
    
    # Convert 6-element tuple to Affine transform
    transform = Affine(
        transform[0],  # a: pixel width
        transform[1],  # b: row rotation
        transform[2],  # c: x-coordinate of upper-left corner
        transform[3],  # d: column rotation
        transform[4],  # e: pixel height
        transform[5]   # f: y-coordinate of upper-left corner
    )
    
    # Get bounds using the Affine transform
    bounds = rasterio.transform.array_bounds(height, width, transform)
    bounds_geom = box(*bounds)
    
    return {
        "crs": crs,
        "bounds": bounds_geom.wkt,  # Store as WKT for DuckDB
        "pixel_size_x": transform.a,
        "pixel_size_y": abs(transform.e),
        "width": width,
        "height": height
    }

def create_patch_transform(base_transform: Affine, i: int, j: int) -> Affine:
    """Create an affine transform for a patch at position (i,j).

    Args:
        base_transform (Affine): Base transform of the original raster
        i (int): Row index of the patch
        j (int): Column index of the patch

    Returns:
        Affine: New transform for the patch
    """
    return Affine(
        base_transform.a,
        base_transform.b,
        base_transform.c + j * base_transform.a,  # shift x
        base_transform.d,
        base_transform.e,
        base_transform.f + i * base_transform.e   # shift y
    ) 