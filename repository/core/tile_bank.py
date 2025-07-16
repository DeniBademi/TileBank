import os
import uuid
from datetime import datetime
from typing import Callable, List, Tuple, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from duckdb import ConstraintException
import json
from shapely.geometry import box, mapping, shape


from ..spatial.utils import get_raster_spatial_info, get_array_spatial_info, create_patch_transform
from ..io.array_writer import save_array
from .base import BaseRepository
from ..db.init import create_database, seed_data


class TileBankRepository(BaseRepository):
    def __init__(self, db_path="tile_bank.db", save_dir="."):
        if not os.path.exists(db_path):
            create_database(db_path)
            seed_data(db_path)
        super().__init__(db_path)
        
        # Install and load spatial extension
        self.sql("INSTALL spatial;")
        self.sql("LOAD spatial;")
        
        self.save_dir = save_dir
        self._created_files = set()  # Track files created in current transaction
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    def _track_file(self, filepath: str):
        """Track a file that was created during the current transaction."""
        self._created_files.add(filepath)
        


    def add_single_tile_from_array(self, 
                                array: np.ndarray, 
                                satellite_name: str, 
                                date_origin: datetime | str,
                                crs: str,
                                transform: tuple,
                                exists_ok: bool = False,
                                file_format: str = 'npy') -> pd.DataFrame:
        """Create a new tile record in the database from a numpy array.

        Args:
            array (np.ndarray): The numpy array to add
            satellite_name (str): The name of the satellite
            date_origin (datetime): The date of the origin of the tile (YYYY-MM-DD)
            crs (str): Coordinate reference system (e.g. 'EPSG:4326')
            transform (tuple): Affine transform (a, b, c, d, e, f)
            exists_ok (bool): If True, the function will not raise an error if the tile already exists
            file_format (str): Format to save the tile in ('npy' or 'tif')
            
        Returns:
            pd.DataFrame: The tile record
        """
        assert len(array.shape) == 3, "Array must be a 3D numpy array"
        assert array.shape[-1] == array.shape[-2], "Tiles must be square"
        assert file_format in ['npy', 'tif'], "file_format must be either 'npy' or 'tif'"
        
        # Extract spatial information
        spatial_info = get_array_spatial_info(array, crs, transform)
        
        # Create affine transform for GeoTIFF if needed
        affine_transform = Affine(*transform)
        
        # Save the array
        file_path = os.path.join(self.save_dir, f"{uuid.uuid4()}.{file_format}")
        save_array(
            array=array,
            output_path=file_path,
            transform=affine_transform if file_format == 'tif' else None,
            crs=crs if file_format == 'tif' else None
        )
        self._track_file(file_path)
        
        # Add the tile with spatial information
        record = self.add_record("tile", {
            "path": file_path,
            "satellite_id": self.find("satellite", name=satellite_name)['id'].values[0],
            "date_origin": self.parse_date_origin(date_origin),
            **spatial_info
        })
        
        return record

    def add_single_tile_from_path(self, 
                               path: str, 
                               satellite_name: str, 
                               date_origin: datetime | str,
                               exists_ok: bool = False) -> pd.DataFrame:
        """Create a new tile record in the database from a file path.

        Args:
            path (str): The path to the tile
            satellite_name (str): The name of the satellite
            date_origin (datetime): The date of the origin of the tile (YYYY-MM-DD)
            exists_ok (bool): If True, the function will not raise an error if the tile already exists
            
        Returns:
            pd.DataFrame: The tile record
        """
        # Get the satellite id
        satellite = self.find("satellite", name=satellite_name)
        
        date_origin = self.parse_date_origin(date_origin)
        
        # Verify that the file exists
        if not os.path.exists(path):
            raise ValueError(f"File {path} does not exist")

        # Extract spatial information
        spatial_info = get_raster_spatial_info(path)
        
        try:
            tile = self.add_record("tile", {
                "path": path,
                "satellite_id": satellite['id'].values[0],
                "date_origin": date_origin,
                **spatial_info
            })
            
        except ConstraintException as e:
            if not exists_ok:
                raise ValueError(f"Tile with path {path} already exists")
        return tile

    def save_raster_patches(self,
                         raster_path: str,
                         satellite_name: str,
                         date_origin: datetime | str,
                         patch_size: int,
                         validator: Optional[Callable] = None,
                         stride: Optional[int] = None,
                         file_format: str = 'npy') -> List[pd.DataFrame]:
        """Split a raster into patches and save them to the database.

        Args:
            raster_path (str): Path to the raster file
            satellite_name (str): Name of the satellite
            date_origin (datetime | str): Date of the raster acquisition
            patch_size (int): Size of patches in pixels
            validator (callable, optional): Function that takes a patch and returns bool if patch is valid
            stride (int, optional): Stride between patches. If None, uses patch_size
            file_format (str): Format to save the patches in ('npy' or 'tif')

        Returns:
            List[pd.DataFrame]: List of created tile records
        """
        assert file_format in ['npy', 'tif'], "file_format must be either 'npy' or 'tif'"
        
        if validator is None:
            validator = lambda x: True
            
        if stride is None:
            stride = patch_size

        created_tiles = []
        
        with rasterio.open(raster_path) as src:
            # Get the total valid area that can be split into patches
            max_height = ((src.height - patch_size) // stride + 1) * stride
            max_width = ((src.width - patch_size) // stride + 1) * stride
            
            # Calculate base transform
            base_transform = src.transform
            
            # Iterate through patches
            from tqdm import tqdm
            for i in tqdm(range(0, max_height, stride), desc="Creating patches"):
                for j in range(0, max_width, stride):
                    # Read the patch
                    window = Window(j, i, patch_size, patch_size)
                    patch = src.read(window=window)
                    
                    # Skip if patch is invalid
                    if not validator(patch):
                        continue
                    
                    # Calculate new transform for this patch
                    patch_transform = create_patch_transform(base_transform, i, j)
                    
                    try:
                        # Save the patch as a tile
                        tile_record = self.add_single_tile_from_array(
                            array=patch,
                            satellite_name=satellite_name,
                            date_origin=date_origin,
                            crs=src.crs.to_string(),
                            transform=tuple(patch_transform)[:6],
                            exists_ok=True,
                            file_format=file_format
                        )
                        created_tiles.append(tile_record)
                    except Exception as e:
                        print(f"Failed to save patch at ({i}, {j}): {str(e)}")
                        continue
        
        return created_tiles

    def save_raster_patches_with_masks(self,
                                    raster_path: str,
                                    mask_path: str,
                                    satellite_name: str,
                                    date_origin: datetime | str,
                                    patch_size: int,
                                    task: str,
                                    mask_type: str,
                                    validator: Optional[Callable] = None,
                                    stride: Optional[int] = None,
                                    file_format: str = 'npy') -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """Split both a raster and its mask into patches and save them to the database.

        Args:
            raster_path (str): Path to the raster file
            mask_path (str): Path to the mask file
            satellite_name (str): Name of the satellite
            date_origin (datetime | str): Date of the raster acquisition
            patch_size (int): Size of patches in pixels
            task (str): Task type for the mask (must match mask_task_type enum)
            mask_type (str): Type of the mask
            validator (callable, optional): Function that takes a patch and returns bool if patch is valid
            stride (int, optional): Stride between patches. If None, uses patch_size
            file_format (str): Format to save the patches in ('npy' or 'tif')

        Returns:
            Tuple[List[pd.DataFrame], List[pd.DataFrame]]: Lists of created tile and mask records
        """
        assert file_format in ['npy', 'tif'], "file_format must be either 'npy' or 'tif'"
        
        if validator is None:
            validator = lambda x: True
            
        if stride is None:
            stride = patch_size

        created_tiles = []
        created_masks = []
        
        with rasterio.open(raster_path) as src, rasterio.open(mask_path) as mask_src:
            # Verify raster and mask have same dimensions and projection
            assert src.height == mask_src.height and src.width == mask_src.width, \
                "Raster and mask must have same dimensions"
            assert src.crs == mask_src.crs, \
                "Raster and mask must have same coordinate reference system"
            
            # Get the total valid area that can be split into patches
            max_height = ((src.height - patch_size) // stride + 1) * stride
            max_width = ((src.width - patch_size) // stride + 1) * stride
            
            # Calculate base transform
            base_transform = src.transform
            
            # Iterate through patches
            from tqdm import tqdm
            for i in tqdm(range(0, max_height, stride), desc="Creating patches"):
                for j in range(0, max_width, stride):
                    # Read the patches
                    window = Window(j, i, patch_size, patch_size)
                    patch = src.read(window=window)
                    mask_patch = mask_src.read(window=window)
                    
                    # Skip if either patch is invalid
                    if not validator(patch) or not validator(mask_patch):
                        continue
                    
                    # Calculate new transform for this patch
                    patch_transform = create_patch_transform(base_transform, i, j)
                    
                    try:
                        # Save the image patch as a tile
                        tile_record = self.add_single_tile_from_array(
                            array=patch,
                            satellite_name=satellite_name,
                            date_origin=date_origin,
                            crs=src.crs.to_string(),
                            transform=tuple(patch_transform)[:6],
                            exists_ok=True,
                            file_format=file_format
                        )
                        created_tiles.append(tile_record)
                        
                        # Save the mask patch
                        mask_path = os.path.join(self.save_dir, f"mask_{tile_record['id'].values[0]}_{uuid.uuid4()}.{file_format}")
                        save_array(
                            array=mask_patch,
                            output_path=mask_path,
                            transform=patch_transform if file_format == 'tif' else None,
                            crs=src.crs.to_string() if file_format == 'tif' else None
                        )
                        
                        mask_record = self.add_record("mask", {
                            "task": task,
                            "path": mask_path,
                            "tile_id": tile_record['id'].values[0],
                            "date_origin": self.parse_date_origin(date_origin),
                            "mask_type": mask_type,
                            "bounds": tile_record['bounds'],  # Use same bounds as tile
                            "raster_transform": json.dumps(tuple(patch_transform)[:6])  # Store transform for future reference
                        })
                        created_masks.append(mask_record)
                        
                    except Exception as e:
                        print(f"Failed to save patch at ({i}, {j}): {str(e)}")
                        continue
        
        return created_tiles, created_masks

    def create_tile_masks_from_raster(self,
                                    mask_path: str,
                                    task: str,
                                    mask_type: str,
                                    file_format: str = 'npy') -> List[pd.DataFrame]:
        """Create tile masks for all tiles that overlap with a given raster mask.

        Args:
            mask_path (str): Path to the raster mask file
            task (str): Task type for the mask (must match mask_task_type enum)
            mask_type (str): Type of the mask
            file_format (str): Format to save the masks in ('npy' or 'tif')

        Returns:
            List[pd.DataFrame]: List of created mask records
        """
        assert file_format in ['npy', 'tif'], "file_format must be either 'npy' or 'tif'"
        
        created_masks = []
        
        # Read the mask raster to get its spatial information
        with rasterio.open(mask_path) as mask_src:
            # Get the bounds of the mask
            mask_bounds = box(*mask_src.bounds)
            mask_crs = mask_src.crs.to_string()
            

            # Find all tiles that intersect with these bounds using raw SQL
            # since DuckDB doesn't support PostGIS-style spatial operations directly
            query = f"""
            SELECT * FROM tile 
            WHERE CAST(json_extract_string(ST_AsGeoJSON(bounds), '$.coordinates[0][0][0]') AS DOUBLE) <= {mask_bounds.bounds[2]}  -- mask xmax
            AND CAST(json_extract_string(ST_AsGeoJSON(bounds), '$.coordinates[0][2][0]') AS DOUBLE) >= {mask_bounds.bounds[0]}    -- mask xmin
            AND CAST(json_extract_string(ST_AsGeoJSON(bounds), '$.coordinates[0][0][1]') AS DOUBLE) <= {mask_bounds.bounds[3]}    -- mask ymax
            AND CAST(json_extract_string(ST_AsGeoJSON(bounds), '$.coordinates[0][2][1]') AS DOUBLE) >= {mask_bounds.bounds[1]}    -- mask ymin
            """
            tiles = self.sql(query).fetchdf()
            
            if len(tiles) == 0:
                print("No tiles found that intersect with the mask")
                return created_masks
            
            # For each intersecting tile
            from tqdm import tqdm
            for _, tile in tqdm(tiles.iterrows(), desc="Creating tile masks", total=len(tiles)):
                try:
                    # Read the tile to get its dimensions and transform
                    with rasterio.open(tile['path']) as tile_src:
                        
                        # Get bounds from the source file instead of the database
                        tile_bounds = box(*tile_src.bounds)
                        # Get transform directly from the source file
                        tile_transform = tile_src.transform
                        transform_str = f"'{','.join(str(round(x, 10)) for x in tuple(tile_transform)[:6])}'"
                        
                        # Calculate the window in the mask that corresponds to this tile
                        window = rasterio.windows.from_bounds(
                            *tile_bounds.bounds,
                            transform=mask_src.transform
                        )
                        
                        # Read and reproject the mask data for this tile
                        mask_data = mask_src.read(
                            window=window,
                            out_shape=(mask_src.count, tile_src.height, tile_src.width),
                            resampling=rasterio.enums.Resampling.nearest
                        )
                        
                        # Save the mask patch
                        mask_path = os.path.join(self.save_dir, f"mask_{tile['id']}_{uuid.uuid4()}.{file_format}")
                        save_array(
                            array=mask_data,
                            output_path=mask_path,
                            transform=tile_transform if file_format == 'tif' else None,
                            crs=mask_crs if file_format == 'tif' else None
                        )
                        self._track_file(mask_path)
                        
                        # Create transform string as a list of numbers
                        transform_list = [round(x, 10) for x in tuple(tile_transform)[:6]]
                        
                        # Use raw SQL to insert the mask record to properly handle the transform data
                        bounds_geojson = json.dumps(mapping(tile_bounds))
                        mask_record = self.sql(f"""
                            INSERT INTO mask (task, path, tile_id, date_origin, mask_type, bounds, raster_transform)
                            VALUES (
                                '{task}',
                                '{mask_path}',
                                {tile['id']},
                                '{tile['date_origin']}',
                                '{mask_type}',
                                ST_GeomFromGeoJSON('{bounds_geojson}'),
                                '{",".join(map(str, transform_list))}'
                            )
                            RETURNING *;
                        """).fetchdf()
                        
                        created_masks.append(mask_record)
                        
                except Exception as e:
                    print(f"Failed to create mask for tile {tile['id']}: {str(e)}")
                    continue
        
        return created_masks

    def parse_date_origin(self, date_origin: datetime | str) -> str:
        """Parse date origin to standard format.

        Args:
            date_origin (datetime | str): Input date

        Returns:
            str: Date in YYYY-MM-DD format
        """
        if isinstance(date_origin, datetime):
            return date_origin.strftime("%Y-%m-%d")
        elif isinstance(date_origin, str):
            # assert date_origin is a string in the format YYYY-MM-DD
            import re
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_origin):
                raise ValueError(f"Date origin must be in the format YYYY-MM-DD, got {date_origin}")
            return date_origin 
        
    def _cleanup_files(self):
        """Delete all files created during the current transaction."""
        for filepath in self._created_files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                print(f"Warning: Failed to delete file {filepath}: {str(e)}")
        self._created_files.clear()
        
    def abort_changes(self):
        """Override parent's abort_changes to also cleanup files."""
        self._cleanup_files()
        super().abort_changes()