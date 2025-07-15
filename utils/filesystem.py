import numpy as np
from PIL import Image
import os
import rasterio

def save_raster_data(data: np.ndarray, path: str):
    extension = os.path.splitext(path)[1]
    directory = os.path.dirname(path)
    
    os.makedirs(directory, exist_ok=True)
    
    if extension == '.png':
        # convert from channels, height, width to height, width, channels
        data = np.moveaxis(data, 0, -1)
        assert data.shape[2] == 3, "RGB image expected"
                
        # Save the tile as a PNG image, dont compress
        Image.fromarray(
            data
        ).save(path, compress_level=0)

    elif extension == '.tif':
        with rasterio.open(path, 'w', driver='GTiff', width=data.shape[1], height=data.shape[2], count=data.shape[0], dtype=data.dtype) as dst:
            dst.write(data)

    elif extension == '.npy':
        np.save(path, data)
        
def load_raster_data(path: str):
    extension = os.path.splitext(path)[1]
    
    if extension == '.png':
        return np.array(Image.open(path))
    
    elif extension == '.tif':
        return rasterio.open(path).read()
    
    elif extension == '.npy':
        return np.load(path)
    
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
