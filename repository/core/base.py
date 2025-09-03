import duckdb
import pandas as pd
from functools import wraps

def check_table(func):
    """This decorator checks if the table exists in the database before executing the function."""
    @wraps(func)
    def wrapper(self, table: str, *args, **kwargs):
        if table not in self.list_tables():
            raise ValueError(f"Table {table} not found in the database! Available tables: {self.list_tables()}")
        return func(self, table, *args, **kwargs)
    return wrapper

def transactional(func):
    """This decorator starts a transaction before executing the function if it is not already in a transaction.
    If an exception is raised, the transaction is aborted."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._in_transaction:
            self.conn.sql("BEGIN TRANSACTION;")
            self._in_transaction = True
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.abort_changes()
            raise e
    return wrapper

class BaseRepository:
    """Basic repository class for DuckDB that provides 
    transactional CRUD operations, basic table management and query operations.
    
    Args:
        db_path (str): The path to the DuckDB database file
    """
    def __init__(self, db_path):
        self.conn = duckdb.connect(db_path)
        
        self._in_transaction = False
        
    @check_table
    def get_table_schema(self, table: str) -> pd.DataFrame:
        """
        Get the schema of a table
        
        Args:
            table (str): The name of the table to query
            
        Returns:
            pandas.DataFrame: A DataFrame containing the schema
        """
        return self.conn.sql(f"""
            DESCRIBE {table}
        """).fetchdf()
        
    def list_tables(self) -> list[str]:
        """
        List all tables in the database
        
        Returns:
            list[str]: A list of table names
        """
        return self.conn.sql("SHOW TABLES").fetchdf()['name'].tolist()
    
    @check_table
    def get_by_id(self, table: str, id: int) -> pd.DataFrame:
        """
        Get a record by its id
        
        Args:
            table (str): The name of the table to query
            id (int): The id of the record to retrieve
            
        Returns:
            pandas.DataFrame: A DataFrame containing the record
        """
        record = self.conn.sql(f"""
            SELECT * FROM {table} WHERE id = {id}
        """).fetchdf()
        if record.empty:
            raise ValueError(f"Record with id {id} not found in table {table}")
        return record
    
    @check_table
    def find(self, table: str, **kwargs) -> pd.DataFrame:
        """
        Find records in a table by a given criteria
        
        Args:
            table (str): The name of the table to query
            **kwargs: The criteria to search for
            
        Returns:
            pandas.DataFrame: A DataFrame containing the record
        """
        
        # Assert all the keys are valid columns in the table
        for key in kwargs.keys():
            if key not in self.get_table_schema(table)['column_name'].tolist():
                raise ValueError(f"Column {key} not found in table {table}")
        
        result = self.conn.sql(f"""
            SELECT * FROM {table} WHERE {', '.join([f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}" for k, v in kwargs.items()])}
        """).fetchdf()
        if result.empty:
            raise ValueError(f"No records found in table {table} with the given criteria")
        if len(result) > 1:
            raise ValueError(f"Multiple records found in table {table} with the given criteria")
        return result
    
    @check_table
    def get_all(self, table: str, columns: list[str] | str = "*") -> pd.DataFrame:
        """
        Get all records from a table
        
        Args:
            table (str): The name of the table to query
            columns (list[str]): The columns to select, defaults to all columns
            
        Returns:
            pandas.DataFrame: A DataFrame containing all records
        """

        column_expr = ', '.join(columns) if isinstance(columns, list) else columns
        
        return self.conn.sql(f"""
            SELECT {column_expr} FROM {table}
        """).fetchdf()
        
    @transactional
    @check_table
    def add_record(self, table: str, data: dict) -> pd.DataFrame:
        """
        Add a record to a table
        
        Args:
            table (str): The name of the table to add the record to
            data (dict): A dictionary containing the data to add
            
        Returns:
            pandas.DataFrame: A DataFrame containing the record
        """
        record = self.conn.sql(f"""
            INSERT INTO {table} ({', '.join(data.keys())})
            VALUES ({', '.join([f"'{s}'" if isinstance(s, str) else str(s) for s in data.values()])})
            RETURNING *
        """).fetchdf()
        return record
        
    @transactional
    @check_table
    def update_record(self, table: str, id: int, data: dict) -> bool:
        """
        Update a record in a table
        
        Args:
            table (str): The name of the table to update the record in
            id (int): The id of the record to update
            data (dict): A dictionary containing the data to update
            
        Returns:
            bool: True if the record was updated successfully, False otherwise
        """
        new_record = self.conn.sql(f"""
        UPDATE {table} SET {', '.join([f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}" for k, v in data.items()])}
        WHERE id = {id}
        RETURNING *
        """).fetchdf()
        if new_record.empty:
            raise ValueError(f"Record with id {id} not found in table {table}")
        return new_record
        
    @transactional
    @check_table
    def delete_record(self, table: str, id: int) -> bool:
        """
        Delete a record from a table
        
        Args:
            table (str): The name of the table to delete the record from
            id (int): The id of the record to delete
            
        Returns:
            bool: True if the record was deleted successfully, False otherwise
        """
        self.conn.sql(f"""
            DELETE FROM {table} WHERE id = {id}
        """)
        return True
    
    @transactional
    @check_table
    def create_colomn_in_table(self, table: str, new_colomn_name: str, data_type: str): 

        if data_type == "str":
            data_type = "varchar(255)"
        elif data_type == "int":
            data_type = "INT"
        elif data_type == "float":
            data_type = "FLOAT"
        elif data_type == "bool":
            data_type = "BOOL"
        elif data_type == "datetime":
            data_type = "TIMESTAMP"
        else:
            raise ValueError(f"Invalid data type: {data_type}")

        self.conn.sql(f"""
                ALTER TABLE {table}
                ADD {new_colomn_name} {data_type};
            """)

    @transactional
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a query and return the result as a DataFrame
        
        Args:
            query (str): The query to execute
            
        Returns:
            pandas.DataFrame: A DataFrame containing the result
        """
        return self.conn.sql(query).fetchdf()
        
    def save_changes(self):
        """
        Commit a transaction
        """
        self.conn.sql("COMMIT;")
        self._in_transaction = False
    
    def abort_changes(self):
        """
        Abort a transaction
        """
        self.conn.sql("ABORT;")
        self._in_transaction = False
        
    def sql(self, query: str) -> duckdb.DuckDBPyConnection:
        """
        Execute a query
        """
        return self.conn.sql(query) 
    
    
   