import os
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling
import os
import numpy as np

from typing import Tuple

def rasterize_shp(shp_path: str, out_shape: Tuple[int,int], transform, dtype=np.int16) -> np.ndarray:
    """
    Rasterize shapefile using the same template as the raster file

    Args:
    shp_path (str): path to the shapefile
    out_shape (tuple): shape of the output raster
    transform (Affine): affine transformation of the output raster
    dtype (numpy.dtype): data type of the output raster

    Returns:
    numpy.ndarray: rasterized shapefile
    
    """
    import fiona
    gdf = fiona.open(shp_path)
    geom = [feature["geometry"] for feature in gdf]

    rasterized = features.rasterize(geom,
                                    out_shape = out_shape,
                                    fill = 0,
                                    out = None,
                                    transform = transform,
                                    all_touched = False,
                                    default_value = 1,
                                    dtype = dtype)
    return rasterized

def align_rasters(input_rasters: list[str], reference_raster: str, output_dir: str = None, verbose: bool = False) -> list[np.ndarray]:
    """
    Aligns a list of raster files to match the spatial extent, resolution, and CRS of a reference raster.
    
    Parameters:
        input_rasters (list of str): Paths to input raster files.
        reference_raster (str): Path to the reference raster file.
        output_dir (str, optional): Directory to save aligned rasters. If None, returns the aligned raster data.
    """
    assert os.path.exists(reference_raster), f"Reference raster {reference_raster} does not exist"
    
    # Open reference raster
    with rasterio.open(reference_raster) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
        ref_res = ref.res
        ref_bounds = ref.bounds
    
    aligned_rasters = []
    
    for raster_path in tqdm(input_rasters, desc=f"Aligning rasters to {reference_raster}"):
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise ValueError(f"Source raster {raster_path} has no CRS defined.")
            
            if verbose:
                print(f"Aligning raster {raster_path} to reference raster {reference_raster}")
                print(f"Reference CRS: {ref_crs}")
                print(f"Source CRS: {src.crs}")
            
            # transform, width, height = calculate_default_transform(
            #     src.crs, ref_crs, src.width, src.height, *src.bounds
            # )
            
            kwargs = src.meta.copy()
            kwargs.update({
                "crs": ref_crs,
                "transform": ref_transform,
                "width": ref_width,
                "height": ref_height
            })
            
            resampled_bands = []
            for i in range(1, src.count + 1):
                data = src.read(i)
                resampled_data = np.empty((ref_height, ref_width), dtype=src.dtypes[i - 1])
                
                reproject(
                    source=data,
                    destination=resampled_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear
                )
                
                # Clip to reference bounds
                row_start, col_start = 0, 0
                row_end, col_end = ref_height, ref_width
                resampled_data = resampled_data[row_start:row_end, col_start:col_end]
                resampled_bands.append(resampled_data)
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, os.path.basename(raster_path))
                with rasterio.open(output_path, "w", **kwargs) as dst:
                    for i, band in enumerate(resampled_bands, start=1):
                        dst.write(band, i)
                aligned_rasters.append(output_path)
            else:
                aligned_rasters.append(np.stack(resampled_bands))
    
    return aligned_rasters