import pyodbc
import requests
import time
from datetime import datetime
from typing import List, Dict, Optional
import json


class BaseCollegeBasketballETL:
    """Base class for College Basketball API ETL operations"""
    
    def __init__(self, api_key: str, db_connection_string: str):
        """
        Initialize the ETL process
        
        Args:
            api_key: Your API key for collegebasketballdata.com
            db_connection_string: MSSQL connection string
        """
        self.api_key = api_key
        self.base_url = "https://api.collegebasketballdata.com"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.conn_string = db_connection_string
    
    def get_db_connection(self):
        """Get database connection"""
        return pyodbc.connect(self.conn_string)
    
    def fetch_api(self, endpoint: str, params: Dict = None, timeout: int = 30) -> List[Dict]:
        """
        Fetch data from API endpoint
        
        Args:
            endpoint: API endpoint path (e.g., '/games' or '/substitutions/game/111914')
            params: Query parameters
            timeout: Request timeout in seconds
            
        Returns:
            List of dictionaries from API response
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else [data]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {endpoint}: {e}")
            return []
    
    def parse_datetime(self, dt_string: str) -> Optional[datetime]:
        """Parse ISO datetime string to datetime object"""
        if not dt_string:
            return None
        try:
            return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
    
    def safe_get(self, data: Dict, key: str, default=None):
        """Safely get value from dictionary"""
        return data.get(key, default)
    
    def json_serialize(self, obj) -> str:
        """Serialize object to JSON string"""
        if obj is None:
            return None
        return json.dumps(obj)
    
    def execute_merge(self, cursor, sql: str, params: tuple, record_id: str = ""):
        """
        Execute a MERGE statement with error handling
        
        Args:
            cursor: Database cursor
            sql: SQL MERGE statement
            params: Parameters for the statement
            record_id: Identifier for error logging
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor.execute(sql, params)
            return True
        except Exception as e:
            print(f"Error inserting record {record_id}: {e}")
            return False
    
    def get_season_date_ranges(self, season: int) -> List[tuple]:
        """
        Get date ranges for a basketball season
        
        Args:
            season: Season year (e.g., 2006 for 2005-06 season)
            
        Returns:
            List of (start_date, end_date) tuples for each month
        """
        start_year = season - 1
        end_year = season
        
        return [
            (f"{start_year}-11-01T00:00:00Z", f"{start_year}-12-01T00:00:00Z"),  # November
            (f"{start_year}-12-01T00:00:00Z", f"{end_year}-01-01T00:00:00Z"),    # December
            (f"{end_year}-01-01T00:00:00Z", f"{end_year}-02-01T00:00:00Z"),      # January
            (f"{end_year}-02-01T00:00:00Z", f"{end_year}-03-01T00:00:00Z"),      # February
            (f"{end_year}-03-01T00:00:00Z", f"{end_year}-04-01T00:00:00Z"),      # March
            (f"{end_year}-04-01T00:00:00Z", f"{end_year}-05-01T00:00:00Z"),      # April
        ]
    
    def rate_limit_sleep(self, seconds: float = 0.5):
        """Sleep to respect API rate limits"""
        time.sleep(seconds)


class GameIDFetcher:
    """Helper class to fetch game IDs for batch processing"""
    
    def __init__(self, db_connection_string: str):
        self.conn_string = db_connection_string
    
    def get_game_ids_for_season(self, season: int) -> List[int]:
        """
        Fetch all game IDs for a given season from the database
        
        Args:
            season: Season year
            
        Returns:
            List of game IDs
        """
        conn = pyodbc.connect(self.conn_string)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT id FROM games WHERE season = ? ORDER BY startDate", (season,))
            game_ids = [row[0] for row in cursor.fetchall()]
            return game_ids
        except Exception as e:
            print(f"Error fetching game IDs for season {season}: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def get_all_game_ids(self, start_season: int, end_season: int) -> Dict[int, List[int]]:
        """
        Fetch all game IDs for multiple seasons
        
        Args:
            start_season: Starting season year
            end_season: Ending season year
            
        Returns:
            Dictionary mapping season -> list of game IDs
        """
        all_game_ids = {}
        for season in range(start_season, end_season + 1):
            game_ids = self.get_game_ids_for_season(season)
            all_game_ids[season] = game_ids
            print(f"Found {len(game_ids)} games for season {season}")
        
        return all_game_ids