"""
Database connection utilities.
"""
import pandas as pd
from sqlalchemy import create_engine
import pyodbc

class DBConnection:
    def __init__(self, config):
        self.config = config
        self.engine = self._create_engine()
        
    def _create_engine(self):
        if self.config['use_windows_auth']:
            conn_str = (
                f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                f'SERVER={self.config["server"]};'
                f'DATABASE={self.config["database"]};'
                f'Trusted_Connection=yes;'
            )
        else:
            conn_str = (
                f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                f'SERVER={self.config["server"]};'
                f'DATABASE={self.config["database"]};'
                f'UID={self.config["username"]};'
                f'PWD={self.config["password"]};'
            )
        
        engine = create_engine(f'mssql+pyodbc:///?odbc_connect={conn_str}')
        print(f"Connected to {self.config['database']} on {self.config['server']}")
        return engine
    
    def query(self, sql):
        return pd.read_sql(sql, self.engine)