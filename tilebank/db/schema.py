"""
Database schema and initialization functions.
"""

import duckdb

def create_database(db_path: str) -> duckdb.DuckDBPyConnection:
    """Create the TileBank database schema.
    
    Args:
        db_path (str): Path where to create the database
        
    Returns:
        duckdb.DuckDBPyConnection: Database connection
    """
    conn = duckdb.connect(db_path)
    
    # Create satellite_type enum
    conn.sql("""
            CREATE TYPE IF NOT EXISTS satellite_type AS ENUM ('optic', 'radar')
    """)

    # Table to store satellite information
    conn.sql("""CREATE SEQUENCE seq_satellite_id START 1;""")
    conn.sql("""
        CREATE TABLE IF NOT EXISTS satellite (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_satellite_id'),
            name VARCHAR(50) NOT NULL,
            resolution_cm INTEGER NOT NULL,
            type satellite_type NOT NULL,
            date_created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
        
    # Table to store timeseries information
    conn.sql("""CREATE SEQUENCE seq_timeseries_id START 1;""")
    conn.sql("""
        CREATE TABLE IF NOT EXISTS timeseries (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_timeseries_id'),
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            date_created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Table to store tile information with geographical coordinates
    conn.sql("""CREATE SEQUENCE seq_tile_id START 1;""")
    conn.sql("""
            CREATE TABLE IF NOT EXISTS tile (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_tile_id'),
                path VARCHAR(300) NOT NULL UNIQUE,
                satellite_id INTEGER NOT NULL REFERENCES satellite(id),
                date_created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                date_origin DATE NULL,
                -- Geographical coordinates of tile corners (in default CRS)
                min_lon DOUBLE NULL,
                min_lat DOUBLE NULL,
                max_lon DOUBLE NULL,
                max_lat DOUBLE NULL,
                -- Pixel dimensions
                width INTEGER NULL,
                height INTEGER NULL,
                -- Spatial reference system identifier (SRID)
                -- NULL if using a non-EPSG CRS (stored in tile_crs table)
                srid INTEGER NULL
            )
        """)
    
    # Table to store non-EPSG coordinate reference systems
    conn.sql("""CREATE SEQUENCE seq_tile_crs_id START 1;""")
    conn.sql("""
        CREATE TABLE IF NOT EXISTS tile_crs (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_tile_crs_id'),
            tile_id INTEGER NOT NULL REFERENCES tile(id),
            crs_wkt TEXT NOT NULL,  -- Well-Known Text representation of the CRS
            date_created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(tile_id)
        )
    """)
    
    # Create spatial index on tile coordinates
    conn.sql("""
        CREATE INDEX IF NOT EXISTS idx_tile_spatial 
        ON tile (min_lon, min_lat, max_lon, max_lat)
        WHERE min_lon IS NOT NULL 
        AND min_lat IS NOT NULL 
        AND max_lon IS NOT NULL 
        AND max_lat IS NOT NULL;
    """)
    
    # Table to store timeseries tile link information
    conn.sql("""CREATE SEQUENCE seq_timeseries_tile_link_id START 1;""")
    conn.sql("""
        CREATE TABLE IF NOT EXISTS timeseries_tile_link (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_timeseries_tile_link_id'),
            timeseries_id INTEGER NOT NULL REFERENCES timeseries(id),
            tile_id INTEGER NOT NULL REFERENCES tile(id),
            date_created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create mask_task_type enum
    conn.sql("""
        CREATE TYPE IF NOT EXISTS mask_task_type AS ENUM ('ntp', 'field_delineation', 'perm_structures')
    """)
    
    # Table to store mask information with geographical coordinates
    conn.sql("""CREATE SEQUENCE seq_mask_id START 1;""")
    conn.sql("""
            CREATE TABLE IF NOT EXISTS mask (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_mask_id'),
                task mask_task_type NOT NULL,
                path VARCHAR(300) NOT NULL UNIQUE,
                path_border VARCHAR(300) NULL,
                path_distance VARCHAR(300) NULL,
                tile_id INTEGER NOT NULL REFERENCES tile(id),
                date_created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                date_origin DATE NULL,
                mask_type VARCHAR(50) NOT NULL,
                timeseries_id INTEGER NULL REFERENCES timeseries(id),
                -- Geographical coordinates (should match parent tile)
                min_lon DOUBLE NULL,
                min_lat DOUBLE NULL,
                max_lon DOUBLE NULL,
                max_lat DOUBLE NULL,
                -- Use same CRS as parent tile
                srid INTEGER NULL
            )
        """)
        
    conn.sql("""CREATE SEQUENCE seq_multimodal_id START 1;""")
    conn.sql("""
            CREATE TABLE IF NOT EXISTS multimodal (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_multimodal_id'),
                timeseries_id INTEGER NOT NULL REFERENCES timeseries(id),
                high_resolution_id INTEGER NOT NULL REFERENCES tile(id),
                date_created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

    return conn

def seed_data(db_path: str):
    """Seed the database with initial data.
    
    Args:
        db_path (str): Path to the database
    """
    conn = duckdb.connect(db_path)
    
    # Insert sample satellite data
    satellites_data = [
        ('Sentinel-2', 100, 'optic'),
        ('Sentinel-1', 100, 'radar'),
        ('Pleiades-50', 50, 'optic'),
        ('PleiadesNEO', 30, 'optic'),
        ('ortophoto25', 25, 'optic'),
    ]
    
    for row in satellites_data:
        conn.sql(f"""
            INSERT INTO satellite (name, resolution_cm, type)
            VALUES ('{row[0]}', {row[1]}, '{row[2]}')
            RETURNING *
        """).fetchdf()
    
    conn.close() 