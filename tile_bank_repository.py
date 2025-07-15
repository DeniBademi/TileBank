import os
from base_repository import BaseRepository
from init_db import create_database, seed_data
from datetime import datetime
from duckdb import ConstraintException
import uuid
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union, Dict
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds, calculate_default_transform, reproject
from shapely.geometry import box
from rasterio.crs import CRS
from rasterio.transform import Affine

class TileBankRepository(BaseRepository):
    def __init__(self, db_path="tile_bank.db", save_dir=".", default_crs='EPSG:4326'):
        if not os.path.exists(db_path):
            create_database(db_path)
            seed_data(db_path)
        super().__init__(db_path)
        self.save_dir = save_dir
        self.default_crs = CRS.from_string(default_crs)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def get_tile_bounds(self, raster_path: str, target_crs: Optional[Union[str, CRS]] = None) -> Tuple[float, float, float, float]:
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

    def find_overlapping_tiles(self, bounds: Tuple[float, float, float, float], crs: Optional[Union[str, CRS]] = None) -> pd.DataFrame:
        """Find all tiles that overlap with the given bounds.
        
        Args:
            bounds (Tuple[float, float, float, float]): (min_lon, min_lat, max_lon, max_lat)
            crs (Optional[Union[str, CRS]]): CRS of the input bounds. If None, assumes same as default_crs.
            
        Returns:
            pd.DataFrame: DataFrame containing the overlapping tiles
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Transform bounds to default CRS if needed
        if crs is not None:
            if isinstance(crs, str):
                crs = CRS.from_string(crs)
            if crs != self.default_crs:
                min_lon, min_lat, max_lon, max_lat = transform_bounds(crs, self.default_crs, min_lon, min_lat, max_lon, max_lat)
        
        return self.conn.sql(f"""
            SELECT * FROM tile 
            WHERE min_lon IS NOT NULL 
            AND max_lon >= {min_lon}
            AND min_lon <= {max_lon}
            AND max_lat >= {min_lat}
            AND min_lat <= {max_lat}
        """).fetchdf()

    def create_mask_for_tile(self, 
                           tile_id: int,
                           mask_raster_path: str,
                           task: str,
                           mask_type: str,
                           date_origin: Optional[datetime | str] = None) -> pd.DataFrame:
        """Create a mask for a specific tile from a larger raster.
        
        Args:
            tile_id (int): ID of the tile to create mask for
            mask_raster_path (str): Path to the raster containing mask data (can be TIF or NPY)
            task (str): Type of mask task (e.g., 'ntp', 'field_delineation')
            mask_type (str): Type of mask (e.g., 'binary', 'multiclass')
            date_origin (Optional[datetime | str]): Date origin for the mask
            
        Returns:
            pd.DataFrame: The created mask record
        """
        # Get tile information
        tile = self.get_by_id("tile", tile_id)
        
        # Get tile bounds and CRS
        tile_bounds = (
            tile['min_lon'].values[0],
            tile['min_lat'].values[0],
            tile['max_lon'].values[0],
            tile['max_lat'].values[0]
        )
        
        # Get tile CRS from SRID or use default
        tile_crs = CRS.from_epsg(tile['srid'].values[0]) if tile['srid'].values[0] else self.default_crs
        
        # Create output path for the mask
        mask_save_path = os.path.join(self.save_dir, f"mask_{uuid.uuid4()}.npy")
        
        # Handle different input formats
        if mask_raster_path.lower().endswith('.npy'):
            # Load NPY file
            mask_data = np.load(mask_raster_path)
            # TODO: Add metadata handling for NPY files (CRS, bounds, etc.)
            # For now, assume same CRS as tile
            data = mask_data
        else:
            # Read and crop the mask raster to tile bounds
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
                    raise ValueError(f"Tile {tile_id} does not overlap with mask raster")
                
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
        
        # Save the mask as NPY
        np.save(mask_save_path, data)
        
        # Create mask record
        if date_origin:
            date_origin = self.parse_date_origin(date_origin)
        
        try:
            mask_record = self.add_record("mask", {
                "task": task,
                "path": mask_save_path,
                "tile_id": tile_id,
                "date_origin": date_origin.strftime("%Y-%m-%d") if isinstance(date_origin, datetime) else date_origin,
                "mask_type": mask_type,
                "min_lon": tile_bounds[0],
                "min_lat": tile_bounds[1],
                "max_lon": tile_bounds[2],
                "max_lat": tile_bounds[3],
                "srid": tile['srid'].values[0] if tile['srid'].values[0] else None
            })
        except Exception as e:
            # Clean up NPY file if record creation fails
            if os.path.exists(mask_save_path):
                os.remove(mask_save_path)
            raise e
        
        return mask_record

    def create_masks_from_raster(self,
                               mask_raster_path: str,
                               task: str,
                               mask_type: str,
                               date_origin: Optional[datetime | str] = None) -> List[pd.DataFrame]:
        """Create masks for all tiles that overlap with a given raster.
        
        Args:
            mask_raster_path (str): Path to the raster containing mask data
            task (str): Type of mask task (e.g., 'ntp', 'field_delineation')
            mask_type (str): Type of mask (e.g., 'binary', 'multiclass')
            date_origin (Optional[datetime | str]): Date origin for the masks
            
        Returns:
            List[pd.DataFrame]: List of created mask records
        """
        # Get raster bounds
        raster_bounds = self.get_tile_bounds(mask_raster_path)
        
        # Find overlapping tiles
        overlapping_tiles = self.find_overlapping_tiles(raster_bounds)
        
        # Create masks for each overlapping tile
        mask_records = []
        for _, tile in overlapping_tiles.iterrows():
            try:
                mask_record = self.create_mask_for_tile(
                    tile_id=tile['id'],
                    mask_raster_path=mask_raster_path,
                    task=task,
                    mask_type=mask_type,
                    date_origin=date_origin
                )
                mask_records.append(mask_record)
            except Exception as e:
                print(f"Failed to create mask for tile {tile['id']}: {str(e)}")
                continue
        
        return mask_records

    def seed_data(self):
        
        # Insert sample satellite data
        satellites_data = [
            ('Sentinel-2', 100, 'optic'),
            ('Sentinel-1', 100, 'radar'),
            ('Pleiades-50', 50, 'optic'),
            ('PleiadesNEO', 30, 'optic'),
            ('ortophoto25', 25, 'optic'),
        ]
        
        for row in satellites_data:
            self.add_record("satellite", {"name": row[0], "resolution_cm": row[1], "type": row[2]})
        
    def parse_date_origin(self, date_origin: datetime | str) -> str:
        if isinstance(date_origin, datetime):
            return date_origin.strftime("%Y-%m-%d")
        else:
            # assert date_origin is a string in the format YYYY-MM-DD
            import re
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_origin):
                raise ValueError(f"Date origin must be in the format YYYY-MM-DD, got {date_origin}")
            return datetime.strptime(date_origin, "%Y-%m-%d")
        
            
    def add_single_tile_from_path(self, 
                              path: str, 
                              satellite_name: str, 
                              date_origin: datetime | str,
                              exists_ok: bool = False) -> pd.DataFrame:
        """Create a new tile record in the database

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
        
        # Get geographical bounds, dimensions and data
        with rasterio.open(path) as src:
            bounds = src.bounds
            width = src.width
            height = src.height
            
            # Store original CRS if available, otherwise use default
            if src.crs:
                srid = src.crs.to_epsg() if src.crs.is_epsg_code else None
                crs_wkt = src.crs.to_wkt() if not src.crs.is_epsg_code else None
            else:
                srid = self.default_crs.to_epsg()
                crs_wkt = None
            
            # Transform bounds to default CRS if different
            if src.crs and src.crs != self.default_crs:
                bounds = transform_bounds(src.crs, self.default_crs, *bounds)
                
            # Read the data
            data = src.read()
        
        # Save as NPY file
        npy_path = os.path.join(self.save_dir, f"{uuid.uuid4()}.npy")
        np.save(npy_path, data)
        
        try:
            tile = self.add_record("tile", {
                "path": npy_path,  # Save NPY path instead of original
                "satellite_id": satellite['id'].values[0],
                "date_origin": date_origin.strftime("%Y-%m-%d") if isinstance(date_origin, datetime) else date_origin,
                "min_lon": bounds.left,
                "min_lat": bounds.bottom,
                "max_lon": bounds.right,
                "max_lat": bounds.top,
                "width": width,
                "height": height,
                "srid": srid
            })
            
            # If CRS is not EPSG, store WKT in a separate metadata table
            if crs_wkt:
                self.add_record("tile_crs", {
                    "tile_id": tile['id'].values[0],
                    "crs_wkt": crs_wkt
                })
                
        except ConstraintException as e:
            # Clean up NPY file if record creation fails
            if os.path.exists(npy_path):
                os.remove(npy_path)
            if not exists_ok:
                raise ValueError(f"Tile with path {path} already exists")
        return tile
    
    def add_single_tile_from_array(self, 
                                   array: np.ndarray, 
                                   satellite_name: str, 
                                   date_origin: datetime | str,
                                   exists_ok: bool = False) -> pd.DataFrame:
        """Create a new tile record in the database from a numpy array

        Args:
            array (np.ndarray): The numpy array to add
            satellite_name (str): The name of the satellite
            date_origin (datetime): The date of the origin of the tile (YYYY-MM-DD)
            exists_ok (bool): If True, the function will not raise an error if the tile already exists
            
        Returns:
            pd.DataFrame: The tile record
        """
        
        # Save the array to a file
        file_path = os.path.join(self.save_dir, f"{uuid.uuid4()}.npy")
        np.save(file_path, array)
        
        # Add the tile
        record = self.add_single_tile_from_path(file_path, satellite_name, date_origin, exists_ok=exists_ok)
        
        return record
    
    def add_timeseries_from_path(self,
                                  paths: list[str], 
                                  satellite_name: str, 
                                  date_origins: list[datetime | str]):
        """Create a new time series record in the database from a list of paths to the individual tiles

        Args:
            path (list[str]): The list of paths to the individual tiles
            satellite_name (str): The name of the satellite
            date_origins (list[datetime | str]): The list of dates of the origin of the tiles (YYYY-MM-DD)

        Returns:
            pd.DataFrame: The timeseries record
        """
        
        # assert that the number of paths is equal to the number of date origins
        if len(paths) != len(date_origins):
            raise ValueError(f"The number of paths ({len(paths)}) is not equal to the number of date origins ({len(date_origins)})")
        
        # assert that the date origins are in the correct format
        for i, date_origin in enumerate(date_origins):
            date_origins[i] = self.parse_date_origin(date_origin)

        date_origins.sort()
        
        # add all the tiles to the database
        tiles = []
        for i, path in enumerate(paths):
            new_tile = self.add_single_tile_from_path(path, satellite_name, date_origins[i], exists_ok=True)
            tiles.append(new_tile)
            
        # create a new timeseries record
        timeseries = self.add_record("timeseries", {
            "start_date": date_origins[0].strftime("%Y-%m-%d"),
            "end_date": date_origins[-1].strftime("%Y-%m-%d")
        })
        
        # link the tiles to the timeseries
        for tile in tiles:
            self.add_record("timeseries_tile_link", {
                "timeseries_id": timeseries['id'].values[0],
                "tile_id": tile['id'].values[0]
            })
        
        return timeseries
    
    def add_timeseries_from_array(self, 
                                  data: np.ndarray,
                                  satellite_name: str, 
                                  date_origins: list[datetime | str]):
        """This function accepts a 4d numpy array of shape (time, bands, height, width).
        First, it will split the array into a list of 3d arrays, each representing a single time step.
        Then, it will add each time step to the database as a separate tile.
        Finally, it will create a new timeseries record and link the tiles to it.

        Args:
            data (np.ndarray): The 4d numpy array of shape (time, bands, height, width)
            satellite_name (str): The name of the satellite
            date_origins (list[datetime  |  str]): The list of dates of the origin of the tiles (YYYY-MM-DD)
            exists_ok (bool): If True, the function will not raise an error if a tile in the timeseries already exists
            
        Raises:
            ValueError: If the data is not a 4d numpy array
            ValueError: If the data is not a 4d numpy array of shape (time, bands, height, width)
            ValueError: If the number of date origins is not equal to the number of time steps
            ValueError: If the date origins are not in the correct format
            ValueError: If the satellite name is not found
        """
        
        assert len(data.shape) == 4, "Data must be a 4d numpy array"
        assert data.shape[-1] == data.shape[-2], "Data must be a square array"
        
        for date_origin in date_origins:
            date_origin = self.parse_date_origin(date_origin)
            
        satellite_record = self.find("satellite", name=satellite_name)
        
        date_origins.sort()
        
        timeseries_record = self.add_record("timeseries", {
            "start_date": date_origins[0].strftime("%Y-%m-%d"),
            "end_date": date_origins[-1].strftime("%Y-%m-%d")
        })
        
        timestep_paths = []
        for i in range(data.shape[0]):            
            tile_record = self.add_single_tile_from_array(data[i], satellite_name, date_origins[i])
            timestep_paths.append(tile_record['path'].values[0])
        
        self.add_timeseries_from_path(timestep_paths, satellite_name, date_origins)
        
        return timeseries_record
        
        
    
    def add_multimodal_from_path(self, high_res_path: str, timeseries_paths: list[str], satellite_name: str, date_origin: datetime | str):
        """This function accepts a path to a high resolution image and a list of paths to the individual tiles of a timeseries.
        It will add the high resolution image to the database as a separate tile.
        Then, it will add each tile of the timeseries to the database as a separate tile.
        Next, it will create a new timeseries record and link the tiles to it.
        Finally, it will create a new multimodal record and link the high resolution tile and the timeseries to it.
        
        Args:
            high_res_path (str): The path to the high resolution image
            timeseries_paths (list[str]): The list of paths to the individual tiles of the timeseries
            satellite_name (str): The name of the satellite
            date_origin (datetime | str): The date of the origin of the tiles (YYYY-MM-DD)
            
        Raises:
            ValueError: If the high resolution image does not exist
            ValueError: If the timeseries paths do not exist
            ValueError: If the satellite name is not found
            ValueError: If the timeseries already exists and exists_ok is False
        """
        raise NotImplementedError("Not implemented yet")
    
    def add_multimodal_from_array(self, high_res_data: np.ndarray, timeseries_data: np.ndarray, satellite_name: str, date_origin: datetime | str):
        """This function accepts a 3D numpy array of shape (bands, height, width) for the high resolution image and a 4d numpy array of shape (time, bands, height, width) for the timeseries.
        It will add the high resolution image to the database as a separate tile and save it to a file.
        Then, it will split the timeseries array into a list of 3D arrays, each representing a single time step.
        Then, it will add each time step to the database as a separate tile and save it to a file.
        Next, it will create a new timeseries record and link the tiles to it.
        Finally, it will create a new multimodal record and link the high resolution tile and the timeseries to it.
        
        Args:
            high_res_data (np.ndarray): The 3D numpy array of shape (bands, height, width) for the high resolution image
            timeseries_data (np.ndarray): The 4d numpy array of shape (time, bands, height, width) for the timeseries
            satellite_name (str): The name of the satellite
            date_origin (datetime | str): The date of the origin of the tiles (YYYY-MM-DD)
            
        Raises:
            ValueError: If the high resolution data is not a 3d numpy array
            ValueError: If the timeseries data is not a 4d numpy array
            ValueError: If the number of date origins is not equal to the number of time steps
            ValueError: If the date origins are not in the correct format
            ValueError: If the satellite name is not found
            ValueError: If the timeseries already exists and exists_ok is False
            """
        raise NotImplementedError("Not implemented yet")
    
    def add_tiles_from_array(self,
                           data: np.ndarray,
                           satellite_name: str,
                           date_origin: datetime | str,
                           bounds: Tuple[float, float, float, float],
                           source_crs: Optional[Union[str, CRS]] = None,
                           patch_size: int = 256,
                           validator: Optional[callable] = None) -> List[pd.DataFrame]:
        """Add multiple tiles from a large numpy array by splitting it into patches.
        
        Args:
            data (np.ndarray): Array of shape (bands, height, width) or (time, bands, height, width)
            satellite_name (str): Name of the satellite
            date_origin (datetime | str): Date origin for the tiles
            bounds (Tuple[float, float, float, float]): (min_lon, min_lat, max_lon, max_lat) of the full array
            source_crs (Optional[Union[str, CRS]]): CRS of the bounds. If None, uses default_crs
            patch_size (int): Size of patches to split into
            validator (Optional[callable]): Function that takes a patch and returns bool if valid
            
        Returns:
            List[pd.DataFrame]: List of created tile records
        """
        # Convert CRS if needed
        if source_crs:
            if isinstance(source_crs, str):
                source_crs = CRS.from_string(source_crs)
            if source_crs != self.default_crs:
                bounds = transform_bounds(source_crs, self.default_crs, *bounds)
        
        # Calculate geographical size of each patch
        full_width = data.shape[-1]
        full_height = data.shape[-2]
        
        # Calculate transform from pixel to geographical coordinates
        min_lon, min_lat, max_lon, max_lat = bounds
        transform = Affine.from_gdal(
            min_lon, (max_lon - min_lon) / full_width, 0,
            max_lat, 0, -(max_lat - min_lat) / full_height
        )
        
        # Find the largest height and width that are divisible by patch_size
        max_height = full_height // patch_size * patch_size
        max_width = full_width // patch_size * patch_size
        
        tile_records = []
        
        for i in range(0, max_height, patch_size):
            for j in range(0, max_width, patch_size):
                # Extract patch
                if len(data.shape) == 4:  # (time, bands, height, width)
                    patch = data[..., i:i+patch_size, j:j+patch_size]
                else:  # (bands, height, width)
                    patch = data[:, i:i+patch_size, j:j+patch_size]
                
                # Skip if validator returns False
                if validator and not validator(patch):
                    continue
                
                # Calculate patch bounds
                patch_bounds = rasterio.transform.array_bounds(
                    patch_size, patch_size,
                    transform * Affine.translation(j, i)
                )
                
                # Save patch as NPY
                npy_path = os.path.join(self.save_dir, f"{uuid.uuid4()}.npy")
                np.save(npy_path, patch)
                
                try:
                    # Add record
                    tile = self.add_record("tile", {
                        "path": npy_path,
                        "satellite_id": self.find("satellite", name=satellite_name)['id'].values[0],
                        "date_origin": self.parse_date_origin(date_origin).strftime("%Y-%m-%d"),
                        "min_lon": patch_bounds.left,
                        "min_lat": patch_bounds.bottom,
                        "max_lon": patch_bounds.right,
                        "max_lat": patch_bounds.top,
                        "width": patch_size,
                        "height": patch_size,
                        "srid": self.default_crs.to_epsg()
                    })
                    tile_records.append(tile)
                except Exception as e:
                    # Clean up NPY file if record creation fails
                    if os.path.exists(npy_path):
                        os.remove(npy_path)
                    raise e
        
        return tile_records
    