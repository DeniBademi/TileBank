import os
from base_repository import BaseRepository
from init_db import create_database, seed_data
from datetime import datetime
from duckdb import ConstraintException
import uuid
import numpy as np
import pandas as pd

class TileBankRepository(BaseRepository):
    def __init__(self, db_path="tile_bank.db", save_dir="."):
        if not os.path.exists(db_path):
            create_database(db_path)
            seed_data(db_path)
        super().__init__(db_path)
        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
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
        
        try:
            tile = self.add_record("tile", {
                "path": path,
                "satellite_id": satellite['id'].values[0],
                "date_origin": date_origin.strftime("%Y-%m-%d") if isinstance(date_origin, datetime) else date_origin
            })
        except ConstraintException as e:
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
            "start_date": date_origins[0],
            "end_date": date_origins[-1]
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
            "start_date": date_origins[0],
            "end_date": date_origins[-1]
        })
        
        timestep_paths = []
        for i in range(data.shape[0]):            
            tile_record = self.add_single_tile_from_array(data[i], satellite_name, date_origins[i])
            timestep_paths.append(tile_record['path'].values[0])
        
        self.add_timeseries_from_path(timestep_paths, satellite_name, date_origins)
        
        return timeseries_record
        
        
    
    def add_multimodal_from_path(self, high_res_path: str, timeseries_paths: list[str], satellite_name: str, date_origin: datetime | str, date_origin_timeseries:list[ datetime | str]):
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

        tile_record = self.add_single_tile_from_path(high_res_path, satellite_name, date_origin, False)

        timeseries_record = self.add_timeseries_from_path(timeseries_paths, satellite_name, date_origin_timeseries)


        multimodal_record = self.add_record("multimodal", {

            "timeseries_id": timeseries_record['id'].values[0],
            "high_resolution_id": tile_record['id'].values[0]
            
        })

        return multimodal_record
    
    def add_multimodal_from_array(self, high_res_data: np.ndarray, timeseries_data: np.ndarray, satellite_name: str, date_origin: datetime | str, date_origin_timeseries:list[ datetime | str]):
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
        
        tile_record = self.add_single_tile_from_array(high_res_data, satellite_name, date_origin)
        

        timeseries_record = self.add_timeseries_from_array(timeseries_data, satellite_name, date_origin_timeseries)


        multimodal_record = self.add_record("multimodal", {

            "timeseries_id": timeseries_record['id'].values[0],
            "high_resolution_id": tile_record['id'].values[0]
            
        })

        return multimodal_record
    