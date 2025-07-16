import os
import numpy as np
import rasterio
from rasterio.transform import Affine

def save_array(array: np.ndarray, 
              output_path: str,
              transform: Affine = None,
              crs: str = None) -> None:
    """Save an array to disk in either .npy or .tif format.

    Args:
        array (np.ndarray): Array to save with shape (bands, height, width)
        output_path (str): Path to save the file
        transform (Affine, optional): Affine transform for GeoTIFF. Required for .tif format.
        crs (str, optional): CRS string for GeoTIFF. Required for .tif format.
    """
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext == '.npy':
        np.save(output_path, array)
    elif ext == '.tif':
        if transform is None or crs is None:
            raise ValueError("transform and crs are required for saving GeoTIFF files")
            
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=array.shape[1],
            width=array.shape[2],
            count=array.shape[0],
            dtype=array.dtype,
            crs=crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(array)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use either .npy or .tif") 