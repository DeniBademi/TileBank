import sys
sys.path.append('C:/data/')
from typing import Optional
import os
import re
from repository.core import TileBankRepository
import rasterio
from datetime import datetime
import argparse


class MZ_orthophoto_client:
    def __init__(self, root_dir: str):
        """
        Args:
            root_dir (str): The root directory where the raw data from MZ is stored
        """
        # keep only the first letter of the disk
        self.root_dir = root_dir
        self.repository = TileBankRepository()
        
    def get_years(self) -> list[int]:
        """
        Get all years in the data directory

        Returns:
            list[int]: List of years
        """
        path = self.root_dir
        dir_contents = os.listdir(path)
        years = [int(year) for year in dir_contents if re.match(r'^\d{4}$', year) and os.path.isdir(os.path.join(path, year))]
        years.sort()
        return years
    
    def get_zones_for_year(self, year: int) -> list[str]:
        """
        Get all zones for a given year

        Args:
            year (int): Year

        Returns:
            list[str]: List of tiles
        """
        path = os.path.join(self.root_dir, str(year))
        dir_contents = os.listdir(path)
        zones = [zone for zone in dir_contents if re.match(r'^[A-Z]\d$', zone) and os.path.isdir(os.path.join(path, zone))]
        zones.sort()
        return zones
    
    def get_orthophoto_paths(self, year: Optional[int] = None, zone: Optional[str] = None, verbose: bool = False) -> list[str]:
        """
        Get all orthophoto paths for a given year and zone

        Args:
            year (Optional[int]): Year
            zone (Optional[str]): Zone
            verbose (bool): Verbose

        Returns:
            list[str]: List of orthophoto paths
        """
        if year is None and zone is None:
            # get all orthophotos for all years
            years = self.get_years()
            output = []
            for year in years:
                output += self.__get_orthophotos_for_year(year, verbose=verbose)
            return output
        elif year is not None and zone is None:
            # get all orthophotos for the year
            return self.__get_orthophotos_for_year(year, verbose=verbose)
        elif year is None and zone is not None:
            # get all orthophotos for the zone for all years
            return self.__get_orthophotos_for_zone(zone, verbose=verbose)
        else:
            # get all orthophotos for the year and zone
            return self.__get_orthophotos_for_year_and_zone(year, zone, verbose=verbose)

    def __get_orthophotos_for_year_and_zone(self, year: int, zone: str, verbose: bool = False):
            assert year is not None and zone is not None, "Year and zone must be provided"
            assert year in self.get_years(), f"Year {year} not found"
            assert zone in self.get_zones_for_year(year), f"Zone {zone} not found for year {year}"
        
            path = os.path.join(self.root_dir, str(year), zone)
            dir_contents = os.listdir(path)
            rasters = [os.path.join(path, raster) for raster in dir_contents if raster.endswith(".tif")]
            return rasters
        
    def __get_orthophotos_for_year(self, year: int, verbose: bool = False):
        # get all zones for the year
        zones = self.get_zones_for_year(year)
        
        output = []
        for zone in zones:
            try:
                output += self.__get_orthophotos_for_year_and_zone(year, zone)
            except Exception as e:
                if verbose:
                    print(f"Error getting orthophotos for year {year} and zone {zone}: {e}")
                continue
        return output
    
    def __get_orthophotos_for_zone(self, zone: str, verbose: bool = False):
        # get all years for the zone
        years = self.get_years()
        # get all orthophotos for the zone
        output = []
        for year in years:
            try:
                output += self.__get_orthophotos_for_year_and_zone(year, zone)
            except Exception as e:
                if verbose:
                    print(f"Error getting orthophotos for year {year} and zone {zone}: {e}")
                continue
        return output
    
    def get_orthophoto_tiles(self, year: int=None, zone: str=None, verbose: bool = False):
        """
        Get all orthophoto tiles for a given year and zone
        """
        
        if zone is None:
            zones = self.get_zones_for_year(year)
        else:
            zones = [zone]

        output = []
        for zone in zones:
            paths = self.get_orthophoto_paths(year=year, zone=zone, verbose=verbose)
            paths = [path for path in paths if path.endswith(".tif")]
            basenames = [os.path.basename(path) for path in paths]
            # names are with nomenclature <zone>-<tile>-<year>AA.tif
            # we want to get the tile and year from the basename
            tiles = [basename.split("-")[1] for basename in basenames]
            output += [(year, zone, tile) for tile in tiles]
        return output
    
    def get_orthophoto_path_from_tile(self, tile: str, year: int, zone: str):
        """
        Get the orthophoto path for a given tile, year and zone
        """
        path = os.path.join(self.root_dir, str(year), zone, f"{zone}-{tile}-{year}AA.tif")
        return path
    
    def get_meta_of_tile(self, tile: str, year: int, zone: str):
        """
        Get the extent of a given tile, year and zone
        """
        path = self.get_orthophoto_path_from_tile(tile, year, zone)
        with rasterio.open(path) as src:
            return src.meta
    
    def load_ortophotos(self, year: int, zone: str, verbose: bool = False):
        """
        Load input data for a given year and zone

        Args:
            year (int): Year
            zone (str): Zone
            verbose (bool): Verbose 

        Returns:
            list[dict]: List of input data
        """
        paths = self.get_orthophoto_paths(year=year, zone=zone, verbose=verbose)
        for path in paths:
            self.repository.save_raster_patches(
                raster_path=path,
                satellite_name="orthophoto",
                date_origin=datetime(year, 1, 1),
                patch_size=256,
                validator=None,
                stride=256,
                file_format='npy')