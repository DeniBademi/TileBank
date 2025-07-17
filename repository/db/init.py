import duckdb

def create_database(db_path: str) -> duckdb.DuckDBPyConnection:
    
    conn = duckdb.connect(db_path)
    
    # Enable and load spatial extension properly
    conn.sql("INSTALL spatial;")
    conn.sql("LOAD spatial;")
    # conn.sql("SET enable_progress_bar=false;")
    
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
                -- Spatial information
                crs VARCHAR(50) NOT NULL,  -- Coordinate Reference System (e.g. 'EPSG:4326')
                bounds GEOMETRY NOT NULL,   -- Polygon representing tile bounds
                pixel_size_x DOUBLE NOT NULL,  -- Pixel size in CRS units
                pixel_size_y DOUBLE NOT NULL,
                width INTEGER NOT NULL,     -- Raster dimensions
                height INTEGER NOT NULL
            )
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

    # Table to store tagtype information
    conn.sql("""CREATE SEQUENCE seq_tagtype_id START 1;""")
    conn.sql("""
            CREATE TABLE IF NOT EXISTS tagtype (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_tagtype_id'),
                name VARCHAR(50) NOT NULL,
                description VARCHAR(200) NOT NULL
            )
        """)
    
    # Table to store tag information
    conn.sql("""CREATE SEQUENCE seq_tag_id START 1;""")
    conn.sql("""
            CREATE TABLE IF NOT EXISTS tag (
                id INTEGER PRIMARY KEY DEFAULT nextval('seq_tag_id'),
                name VARCHAR(50) NOT NULL,
                tagtype_id INTEGER NOT NULL REFERENCES tagtype(id)
            )
        """)
    
    # Table to store tile tag link information
    conn.sql("""CREATE SEQUENCE seq_tile_tag_link_id START 1;""")
    conn.sql("""
        CREATE TABLE IF NOT EXISTS tile_tag_link (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_tile_tag_link_id'),
            tag_id INTEGER NOT NULL REFERENCES tag(id),
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
                -- Spatial information (must match parent tile's CRS)
                bounds GEOMETRY NOT NULL,   -- Polygon representing mask bounds
                raster_transform VARCHAR(300) NOT NULL  -- Affine transform parameters as JSON
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



    # Create spatial indices
    conn.sql("""
        CREATE INDEX idx_tile_bounds ON tile(bounds);
    """)

    conn.sql("""
        CREATE INDEX idx_mask_bounds ON mask(bounds);
    """)

    return conn
        
        
def seed_data(db_path: str):
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
        record = conn.sql(f"""
                INSERT INTO satellite (name, resolution_cm, type)
                VALUES ('{row[0]}', {row[1]}, '{row[2]}')
                RETURNING *
            """).fetchdf()
        
    conn.close() 