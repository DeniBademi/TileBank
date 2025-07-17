import pytest
import numpy as np
from datetime import datetime, timedelta
from rasterio.transform import Affine
from repository.core.tile_bank import TileBankRepository
@pytest.fixture(scope="session", autouse=True)
def do_something(request):

    global repo 
    repo = TileBankRepository(db_path="test_utils/mydb.db", save_dir="test_utils/files")

    global dummy_data 
    global wrong_data 
    global pixel_res 
    global top_left_x 
    global top_left_y 
    global transform 
    global ts_dummy_data 
    
    dummy_data = np.ones(shape=(3,32,32))
    wrong_data = np.ones(shape=(3,32,30))

    pixel_res = 10
    top_left_x = 500_000
    top_left_y = 450_000

    transform = Affine(pixel_res, 0, top_left_x,
                    0, -pixel_res, top_left_y)

    ts_dummy_data = np.ones((2,3,32,32))



def test_assert_tyle_arrayNot3d():
    with pytest.raises(AssertionError):
        repo.add_single_tile_from_array(
        array=ts_dummy_data,
        satellite_name="Sentinel-2",
        date_origin="2024-12-12",
        crs="EPSG:32635",
        transform=tuple(transform)[:6],
        file_format="tif"
        )


def test_tyle_futureDatetime():
    with pytest.raises(AssertionError):
        repo.add_single_tile_from_array(
        array=dummy_data,
        satellite_name="Sentinel-2",
        date_origin= datetime.now()+timedelta(days=2),
        crs="EPSG:32635",
        transform=tuple(transform)[:6],
        file_format="gif"
        )


def test_tyle_nonSquareTyle():
    with pytest.raises(AssertionError):
        repo.add_single_tile_from_array(
        array=wrong_data,
        satellite_name="Sentinel-2",
        date_origin="2024-12-12",
        crs="EPSG:32635",
        transform=tuple(transform)[:6],
        file_format="tif"
        )


def test_tyle_wrongFileFormat():
    with pytest.raises(AssertionError):
        repo.add_single_tile_from_array(
        array=dummy_data,
        satellite_name="Sentinel-2",
        date_origin="2024-12-12",
        crs="EPSG:32635",
        transform=tuple(transform)[:6],
        file_format="gif"
        )


def test_timeseries_arrayNot4d():
    with pytest.raises(AssertionError):
        repo.add_timeseries_from_array(
        data=dummy_data,
        satellite_name="Sentinel-2",
        date_origins=["2024-12-12"]*2,
        crs="EPSG:32635",
        transform=tuple(transform)[:6],
        file_format="gif"
        )


def test_modular_tyleArrayNot3d():
    with pytest.raises(AssertionError):
        repo.add_multimodal_from_array(
        high_res_data=ts_dummy_data,
        timeseries_data=ts_dummy_data,
        satellite_name="Sentinel-2",
        date_origin="2024-12-12",
        date_origin_timeseries=["2024-12-12"]*2,
        crs="EPSG:32635",
        transform=tuple(transform)[:6],
        file_format="tif"
        )


def test_modular_timeseriesArrayNot4d():
    with pytest.raises(AssertionError):
        repo.add_multimodal_from_array(
        high_res_data=dummy_data,
        timeseries_data=dummy_data,
        satellite_name="Sentinel-2",
        date_origin="2024-12-12",
        date_origin_timeseries=["2024-12-12"]*2,
        crs="EPSG:32635",
        transform=tuple(transform)[:6],
        file_format="tif"
        )

