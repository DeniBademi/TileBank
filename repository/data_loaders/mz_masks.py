import sys
sys.path.append('C:/data/')
from typing import Optional, Tuple
import os
import re
from repository.core import TileBankRepository
import rasterio
from datetime import datetime
import argparse
from enum import Enum
import geopandas as gpd
from shapely.geometry import box
import numpy as np
from rasterio.features import rasterize

class MZ_masks_client:
    def __init__(self, root_dir: str = "D:/raw masks/"):
        """Initialize the MZ_masks_client

        Args:
            root_dir (str): Root directory where the raw MZ data is stored
        """
        self.repository = TileBankRepository()
        
        self.root_dir = root_dir
        self.aoi_shapefile_path = os.path.join(f"{self.root_dir}", "AOI_Tiles2.shp")
        
        # Caches to avoid re-reading large shapefiles repeatedly
        self._aoi_gdf = None
        self._year_to_features: dict[int, gpd.GeoDataFrame] = {}
        self._bbox_cache: dict[tuple[str, str | None], object] = {}
        # Default label mapping for NTP -> integer class ids
        self.ntp_label_mapper = {
            '010': 1,
            '020': 2,
            '040': 3,
            '101': 4,
            '200': 5,
        }
        
    def get_years(self) -> list[int]:
        """
        Get all years in the data directory

        Returns:
            list[int]: List of years
        """
        dir_contents = os.listdir(self.root_dir)
        years = [int(year) for year in dir_contents if re.match(r'^\d{4}$', year) and os.path.isdir(os.path.join(self.root_dir, year))]
        return years
    
    def __get_shapefile_path_for_year(self, year: int) -> str:
        """
        Get all tiles for a given year

        Args:
            year (int): Year,
            task (str): Task

        Returns:
            str: Path to the shapefile
        """

        path = os.path.join(self.root_dir, str(year))
        dir_contents = os.listdir(path)
        # get the path to the shapefile in the dir_contents
        shapefile_path = [os.path.join(path, file) for file in dir_contents if file.endswith('.shp')]
        assert len(shapefile_path) == 1, "There should be exactly one shapefile in the directory"
        shapefile_path = shapefile_path[0]
        
        return shapefile_path
    
    def __filter_features_on_task(self, gpd_features: gpd.GeoDataFrame, task: str) -> list[dict]:
        """
        Filter the shapefile based on the task and extent. Returns the same shapefile but with the filtered features.
        Uses a regular expression to match the start of the 'NTP' attribute.
        """
        
        assert task in ['ntp', 'arable_land'], "Task must be one of 'ntp' or 'arable_land'"

        if task == 'ntp':
            pattern = r'^(010|02|04|101|200)' # arable lands, temp_perennials, grasslands, scrubs, forest
        elif task == 'arable_land':
            pattern = r'^010' # arable lands
        else:
            raise ValueError(f"Task {task} not supported")

        filtered = gpd_features[gpd_features['NTP'].astype(str).str.match(pattern)]
        filtered = self.map_ntp_classes_to_task(filtered, task)

        return filtered
    
    def map_ntp_classes_to_task(self, gpd_features: gpd.GeoDataFrame, task: str) -> dict:
        """
        Map NTP classes to task classes. Returns the same shapefile but with the mapped classes.
        """
        assert task in ['ntp', 'arable_land'], "Task must be one of 'ntp' or 'arable_land'"
        
        if task == 'ntp':
            # map all types of temp_perennials to 020
            pattern = r'^(02)'
            filtered = gpd_features[gpd_features['NTP'].astype(str).str.match(pattern)]
            gpd_features.loc[filtered.index, 'NTP'] = '020'
            
            # map all types of grasslands to 040
            pattern = r'^(04)'
            filtered = gpd_features[gpd_features['NTP'].astype(str).str.match(pattern)]
            gpd_features.loc[filtered.index, 'NTP'] = '040'
            
        return gpd_features

    def __get_aoi_gdf(self) -> gpd.GeoDataFrame:
        """Return AOI GeoDataFrame from cache, loading it once if needed."""
        if self._aoi_gdf is None:
            self._aoi_gdf = gpd.read_file(self.aoi_shapefile_path)
        return self._aoi_gdf
        
    def __get_year_features(self, year: int, verbose: bool = False) -> gpd.GeoDataFrame:
        """Return raw features GeoDataFrame for a given year from cache, loading once if needed."""
        if year not in self._year_to_features:
            shapefile_path = self.__get_shapefile_path_for_year(year)
            if verbose:
                print(f"Loading shapefile from {shapefile_path}")
            gdf = gpd.read_file(shapefile_path)
            # Drop null/empty geometries early
            gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)]
            # Make geometries valid to avoid TopologyException on overlay/clip
            try:
                from shapely.validation import make_valid
                gdf.geometry = make_valid(gdf.geometry)
            except Exception:
                # Fallback for older Shapely versions
                gdf.geometry = gdf.geometry.buffer(0)
            # Filter again to remove empties created by fixing
            gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)]
            # Keep only polygonal geometries for rasterization
            gdf = gdf[gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]
            
            #save the gdf to a shapefile
            gdf.to_file(os.path.join(self.root_dir, f"aoi_gdf_{year}_valid.shp"))
            
            self._year_to_features[year] = gdf
            if verbose:
                print(f"Loaded {len(self._year_to_features[year])} features from shapefile")
        return self._year_to_features[year]
    
    def get_year_features(self, year: int, verbose: bool = False) -> gpd.GeoDataFrame:
        """
        Get the features for a given year and task.
        """
        features = self.__get_year_features(year, verbose=verbose)
        return features
    
    def __get_cached_bbox(self, zone: str, tile: Optional[str]) -> object:
        """Return cached bbox for (zone, tile) or compute and cache it."""
        assert zone is not None, "Either zone or tile must be provided"
        cache_key = (zone, tile)
        if cache_key in self._bbox_cache:
            return self._bbox_cache[cache_key]
        
        aoi_gdf = self.__get_aoi_gdf()
        if tile is not None:
            full_name = tile if ("-" in tile) else (f"{zone}-{tile}" if zone is not None else None)
            if full_name is not None:
                zone_features = aoi_gdf[aoi_gdf['Full_name'] == full_name]
                if zone_features.empty and "-" not in tile:
                    zone_features = aoi_gdf[aoi_gdf['Full_name'].str.endswith(f"-{tile}")]
            else:
                zone_features = aoi_gdf[aoi_gdf['Full_name'].str.endswith(f"-{tile}")]
        else:
            zone_features = aoi_gdf[aoi_gdf['Zone_'] == zone]
        
        if zone_features is None or len(zone_features) == 0:
            raise ValueError(f"No AOI features found for zone='{zone}' and tile='{tile}'. Ensure tile id matches 'Full_name' in AOI (e.g. '{zone}-123') or pass correct zone.")
        
        min_x = zone_features.geometry.bounds.minx.min()
        min_y = zone_features.geometry.bounds.miny.min()
        max_x = zone_features.geometry.bounds.maxx.max()
        max_y = zone_features.geometry.bounds.maxy.max()
        bbox = box(min_x, min_y, max_x, max_y)
        self._bbox_cache[cache_key] = bbox
        return bbox
        
    def get_tiles_in_zone(self, zone: str) -> list[str]:
        """
        Get the tiles in a given zone.
        """
        aoi_shapefile = self.__get_aoi_gdf()
        return aoi_shapefile[aoi_shapefile['Zone_'] == zone]['Full_name'].tolist()
    
    def get_bbox(self, zone: str, tile: Optional[str] = None) -> tuple:
        """
        Get the extent for a given zone and tile number. If tile number is not provided, the extent for the whole zone is returned.
        
        Args:
            zone (str): Zone
            tile_no (str): Tile number
            
        Returns:
        """
        return self.__get_cached_bbox(zone=zone, tile=tile)
    
    def __rasterize_features(self, features: gpd.GeoDataFrame, out_shape: Tuple[int,int], transform, dtype=np.int16) -> np.ndarray:
        """
        Rasterize a single class shapefile using the same template as the raster file
        
        Args:
        shp_path (str): path to the shapefile
        out_shape (tuple): shape of the output raster
        transform (Affine): affine transformation of the output raster
        dtype (numpy.dtype): data type of the output raster
        
        Returns:
        numpy.ndarray: rasterized shapefile with dimensions (height, width)
        
        """
        # If there are no features, return an empty mask
        if features is None or len(features) == 0:
            return np.zeros(out_shape, dtype=dtype)
        
        ntp_classes = features['NTP'].astype(str)
        unknown = sorted(set(ntp_classes.unique()) - set(self.ntp_label_mapper.keys()))
        if len(unknown) > 0:
            raise ValueError(f"Unmapped NTP classes: {unknown}.")
        
        shapes = [(geom, self.ntp_label_mapper[ntp_class]) for geom, ntp_class in zip(features.geometry, ntp_classes)]
        
        rasterized = rasterize(shapes,
                                    out_shape = out_shape,
                                    out = None,
                                    transform = transform,
                                    all_touched = False,
                                    default_value = 0,
                                    dtype = dtype)
        return rasterized

    def rasterize_features(self, features: gpd.GeoDataFrame, out_shape: Tuple[int,int], transform, dtype=np.int16) -> np.ndarray:
        """
        Public wrapper to rasterize a GeoDataFrame of features.
        """
        return self.__rasterize_features(features, out_shape, transform, dtype)
        
    def save_features_to_shp(self, features: gpd.GeoDataFrame, output_path: str):
        """
        Save features as a shapefile
        """
        features.to_file(output_path)
    
    def save_features_to_raster(self, features: gpd.GeoDataFrame, output_path: str, meta: dict = None):
        """
        Save features as a raster
        """
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        out_shape = (meta['height'], meta['width'])
        transform = meta['transform']
        dtype = meta['dtype']
        rasterized = self.__rasterize_features(features, out_shape, transform, dtype)
        
        meta['count'] = 1
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(rasterized, 1)
        
    def extract_features(self, 
                     year: int, 
                     task: Optional[str] = None, 
                     zone: Optional[str] = None,
                     tile_no: Optional[str] = None,
                     verbose: bool = False) -> list[dict]:
        """
        Extract features from the MZ data for a given year, task and zone.
        
        Args:
            year (int): Year
            output_path (str): Path to the output shapefile
            task (str): Task
            zone (str): Zone
            verbose (bool): Whether to print verbose output
            Returns:
                gpd.GeoDataFrame: Filtered shapefile
        """
        if task is None and zone is None:
            raise ValueError("Either task or zone must be provided")
        features = self.__get_year_features(year, verbose=verbose)
        
        if zone is not None or tile_no is not None:
            if verbose:
                print(f"Filtering shapefile on zone: {zone}")
            bbox = self.get_bbox(zone=zone, tile=tile_no)
            features = features.clip(bbox)
        
        if task is not None:
            if verbose:
                print(f"Filtering shapefile on task: {task}")
            features = self.__filter_features_on_task(features, task)
        else:
            raise ValueError("Either task or extent must be provided")
         
        return features