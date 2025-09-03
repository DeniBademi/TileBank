from setuptools import setup, find_packages

setup(
    name="repository",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "rasterio",
        "shapely",
        "duckdb",
    ],
    python_requires=">=3.6",
) 