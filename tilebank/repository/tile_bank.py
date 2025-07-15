"""
Main repository class for TileBank.
"""

import os
from datetime import datetime
from typing import List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from rasterio.crs import CRS

from .base import BaseRepository
from ..db.schema import create_database, seed_data
from ..core.tiles import (
    save_tile_as_npy, extract_patch, create_tile_record,
    load_tile_from_tif
)
from ..core.masks import create_mask_record, extract_mask_for_tile
from ..spatial.bounds import calculate_patch_bounds

class TileBankRepository(BaseRepository):
    def __init__(self, db_path="tile_bank.db", save_dir=".", default_crs='EPSG:4326'):
        """Initialize TileBank repository.
        
        Args:
            db_path (str): Path to the database file
            save_dir (str): Directory to save tile and mask files
            default_crs (str): Default CRS for storing coordinates
        """
        if not os.path.exists(db_path):
            create_database(db_path)
            seed_data(db_path)
        super().__init__(db_path)
        
        self.save_dir = save_dir
        self.default_crs = CRS.from_string(default_crs)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
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
            bounds (Tuple[float, float, float, float]): (min_lon, min_lat, max_lon, max_lat)
            source_crs (Optional[Union[str, CRS]]): CRS of the bounds. If None, uses default_crs
            patch_size (int): Size of patches to split into
            validator (Optional[callable]): Function that takes a patch and returns bool if valid
            
        Returns:
            List[pd.DataFrame]: List of created tile records
        """
        # Get satellite ID
        satellite = self.find("satellite", name=satellite_name)
        satellite_id = satellite['id'].values[0]
        
        # Parse date
        if isinstance(date_origin, datetime):
            date_str = date_origin.strftime("%Y-%m-%d")
        else:
            date_str = date_origin
            
        # Convert CRS if needed
        if source_crs:
            if isinstance(source_crs, str):
                source_crs = CRS.from_string(source_crs)
            if source_crs != self.default_crs:
                bounds = transform_bounds(source_crs, self.default_crs, *bounds)
        
        # Calculate dimensions
        full_height = data.shape[-2]
        full_width = data.shape[-1]
        max_height = full_height // patch_size * patch_size
        max_width = full_width // patch_size * patch_size
        
        tile_records = []
        
        for i in range(0, max_height, patch_size):
            for j in range(0, max_width, patch_size):
                # Extract patch
                patch = extract_patch(data, i, j, patch_size)
                
                # Skip if validator returns False
                if validator and not validator(patch):
                    continue
                
                # Calculate patch bounds
                patch_bounds = calculate_patch_bounds(
                    bounds, (full_height, full_width),
                    (i, j), patch_size
                )
                
                # Save patch as NPY
                npy_path = save_tile_as_npy(patch, self.save_dir)
                
                try:
                    # Create record
                    record_data = create_tile_record(
                        path=npy_path,
                        satellite_id=satellite_id,
                        date_origin=date_str,
                        bounds=patch_bounds,
                        dimensions=(patch_size, patch_size),
                        srid=self.default_crs.to_epsg()
                    )
                    
                    tile = self.add_record("tile", record_data)
                    tile_records.append(tile)
                    
                except Exception as e:
                    # Clean up NPY file if record creation fails
                    if os.path.exists(npy_path):
                        os.remove(npy_path)
                    raise e
        
        return tile_records
    
    def add_single_tile_from_path(self, 
                              path: str, 
                              satellite_name: str, 
                              date_origin: datetime | str,
                              exists_ok: bool = False) -> pd.DataFrame:
        """Create a new tile record in the database from a file.

        Args:
            path (str): Path to the tile file
            satellite_name (str): Name of the satellite
            date_origin (datetime | str): Date origin for the tile
            exists_ok (bool): If True, won't raise error if tile exists
            
        Returns:
            pd.DataFrame: The tile record
        """
        # Get satellite ID
        satellite = self.find("satellite", name=satellite_name)
        satellite_id = satellite['id'].values[0]
        
        # Parse date
        if isinstance(date_origin, datetime):
            date_str = date_origin.strftime("%Y-%m-%d")
        else:
            date_str = date_origin
        
        # Load and convert to NPY if needed
        if path.lower().endswith('.npy'):
            data = np.load(path)
            npy_path = path
        else:
            data, metadata = load_tile_from_tif(path, self.default_crs)
            npy_path = save_tile_as_npy(data, self.save_dir)
            
            try:
                # Create record
                record_data = create_tile_record(
                    path=npy_path,
                    satellite_id=satellite_id,
                    date_origin=date_str,
                    bounds=metadata['bounds'],
                    dimensions=(metadata['width'], metadata['height']),
                    srid=metadata['crs'].to_epsg() if metadata['crs'].is_epsg_code else None
                )
                
                tile = self.add_record("tile", record_data)
                
                # Store CRS WKT if not EPSG
                if metadata['crs'] and not metadata['crs'].is_epsg_code:
                    self.add_record("tile_crs", {
                        "tile_id": tile['id'].values[0],
                        "crs_wkt": metadata['crs'].to_wkt()
                    })
                    
            except Exception as e:
                # Clean up NPY file if record creation fails
                if os.path.exists(npy_path):
                    os.remove(npy_path)
                if not exists_ok:
                    raise e
                
        return tile
    
    def create_mask_for_tile(self, 
                           tile_id: int,
                           mask_raster_path: str,
                           task: str,
                           mask_type: str,
                           date_origin: Optional[datetime | str] = None) -> pd.DataFrame:
        """Create a mask for a specific tile.
        
        Args:
            tile_id (int): ID of the tile to create mask for
            mask_raster_path (str): Path to the mask raster
            task (str): Type of mask task
            mask_type (str): Type of mask
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
        
        tile_crs = (CRS.from_epsg(tile['srid'].values[0]) 
                   if tile['srid'].values[0] 
                   else self.default_crs)
        
        # Extract mask data
        mask_data = extract_mask_for_tile(
            mask_raster_path, tile_bounds, tile_crs)
        
        # Save as NPY
        mask_save_path = save_tile_as_npy(mask_data, self.save_dir)
        
        try:
            # Parse date if provided
            if date_origin:
                if isinstance(date_origin, datetime):
                    date_str = date_origin.strftime("%Y-%m-%d")
                else:
                    date_str = date_origin
            else:
                date_str = None
            
            # Create record
            record_data = create_mask_record(
                path=mask_save_path,
                tile_id=tile_id,
                task=task,
                mask_type=mask_type,
                bounds=tile_bounds,
                date_origin=date_str,
                srid=tile['srid'].values[0]
            )
            
            mask_record = self.add_record("mask", record_data)
            
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
            mask_raster_path (str): Path to the mask raster
            task (str): Type of mask task
            mask_type (str): Type of mask
            date_origin (Optional[datetime | str]): Date origin for the masks
            
        Returns:
            List[pd.DataFrame]: List of created mask records
        """
        # Get raster bounds
        bounds = get_tile_bounds(mask_raster_path)
        
        # Find overlapping tiles
        overlapping_tiles = self.find_overlapping_tiles(bounds)
        
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