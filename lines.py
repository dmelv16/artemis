from base_etl import BaseCollegeBasketballETL
from typing import List, Dict
from datetime import datetime, timedelta


class LinesETL(BaseCollegeBasketballETL):
    """ETL for college basketball betting lines data"""
    
    def create_table(self, cursor):
        """Create the lines table if it doesn't exist"""
        create_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='lines' AND xtype='U')
        CREATE TABLE lines (
            id INT IDENTITY(1,1) PRIMARY KEY,
            gameId INT NOT NULL,
            season INT,
            seasonType NVARCHAR(20),
            startDate DATETIME2,
            homeTeamId INT,
            homeTeam NVARCHAR(100),
            homeConference NVARCHAR(50),
            homeScore INT,
            awayTeamId INT,
            awayTeam NVARCHAR(100),
            awayConference NVARCHAR(50),
            awayScore INT,
            
            -- Betting line details
            provider NVARCHAR(100),
            spread FLOAT,
            overUnder FLOAT,
            homeMoneyline INT,
            awayMoneyline INT,
            spreadOpen FLOAT,
            overUnderOpen FLOAT,
            
            CONSTRAINT UC_Line UNIQUE (gameId, provider),
            INDEX IX_Lines_GameId (gameId),
            INDEX IX_Lines_Season (season),
            INDEX IX_Lines_Provider (provider)
        )
        """
        cursor.execute(create_table_sql)
        print("Table 'lines' verified/created")
    
    def fetch_lines_batch(self, season: int, start_date: str = None, end_date: str = None) -> List[Dict]:
        """
        Fetch a batch of betting lines (up to 3000)
        
        Args:
            season: Season year
            start_date: Optional start date in ISO format
            end_date: Optional end date in ISO format
            
        Returns:
            List of game dictionaries with lines
        """
        endpoint = "/lines"
        params = {"season": season}
        if start_date:
            params["startDateRange"] = start_date
        if end_date:
            params["endDateRange"] = end_date
        
        return self.fetch_api(endpoint, params)
    
    def fetch_all_lines_for_season(self, season: int) -> List[Dict]:
        """
        Fetch ALL betting lines for a season by breaking into monthly chunks if needed
        
        Args:
            season: Season year
            
        Returns:
            List of all game dictionaries with lines for the season
        """
        all_games = []
        
        # Get season date ranges from base class
        date_ranges = self.get_season_date_ranges(season)
        
        print(f"Fetching lines for season {season} in monthly chunks...")
        
        for start_date, end_date in date_ranges:
            games = self.fetch_lines_batch(season, start_date, end_date)
            
            if games:
                print(f"  {start_date[:7]}: {len(games)} games with lines")
                
                # If we hit the 3000 limit, break the month into weeks
                if len(games) >= 3000:
                    print(f"  Warning: Hit 3000 game limit for {start_date[:7]}, fetching by weeks...")
                    monthly_games = self.fetch_lines_by_week(season, start_date, end_date)
                    all_games.extend(monthly_games)
                else:
                    all_games.extend(games)
            
            self.rate_limit_sleep(0.5)
        
        # Remove duplicates based on gameId
        unique_games = {game['gameId']: game for game in all_games}.values()
        print(f"Total unique games with lines for season {season}: {len(unique_games)}")
        
        return list(unique_games)
    
    def fetch_lines_by_week(self, season: int, month_start: str, month_end: str) -> List[Dict]:
        """
        Fetch betting lines by week for months with >3000 games
        
        Args:
            season: Season year
            month_start: Start of month in ISO format
            month_end: End of month in ISO format
            
        Returns:
            List of game dictionaries with lines
        """
        all_games = []
        start = datetime.fromisoformat(month_start.replace('Z', '+00:00'))
        end = datetime.fromisoformat(month_end.replace('Z', '+00:00'))
        
        current = start
        while current < end:
            week_end = min(current + timedelta(days=7), end)
            
            start_str = current.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_str = week_end.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            games = self.fetch_lines_batch(season, start_str, end_str)
            
            if games:
                print(f"    {start_str[:10]} to {end_str[:10]}: {len(games)} games")
                all_games.extend(games)
            
            current = week_end
            self.rate_limit_sleep(0.5)
        
        return all_games
    
    def insert_lines(self, games_with_lines: List[Dict], cursor) -> int:
        """
        Insert betting lines into the database
        
        Args:
            games_with_lines: List of game dictionaries containing lines arrays
            cursor: Database cursor
            
        Returns:
            Number of line records inserted
        """
        insert_sql = """
        MERGE lines AS target
        USING (SELECT ? AS gameId, ? AS provider) AS source
        ON target.gameId = source.gameId AND target.provider = source.provider
        WHEN NOT MATCHED THEN
            INSERT (gameId, season, seasonType, startDate, homeTeamId, homeTeam, 
                    homeConference, homeScore, awayTeamId, awayTeam, awayConference, 
                    awayScore, provider, spread, overUnder, homeMoneyline, awayMoneyline,
                    spreadOpen, overUnderOpen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for game in games_with_lines:
            # Some games might have empty lines array
            lines = game.get('lines', [])
            
            if not lines:
                continue
            
            # Process each betting line for this game
            for line in lines:
                try:
                    params = (
                        # For MERGE condition
                        game.get('gameId'),
                        line.get('provider'),
                        # For INSERT
                        game.get('gameId'),
                        game.get('season'),
                        game.get('seasonType'),
                        self.parse_datetime(game.get('startDate')),
                        game.get('homeTeamId'),
                        game.get('homeTeam'),
                        game.get('homeConference'),
                        game.get('homeScore'),
                        game.get('awayTeamId'),
                        game.get('awayTeam'),
                        game.get('awayConference'),
                        game.get('awayScore'),
                        line.get('provider'),
                        line.get('spread'),
                        line.get('overUnder'),
                        line.get('homeMoneyline'),
                        line.get('awayMoneyline'),
                        line.get('spreadOpen'),
                        line.get('overUnderOpen')
                    )
                    
                    if self.execute_merge(cursor, insert_sql, params, 
                                         f"game {game.get('gameId')} provider {line.get('provider')}"):
                        inserted += 1
                        
                except Exception as e:
                    print(f"Error processing line for game {game.get('gameId')}: {e}")
                    continue
        
        return inserted
    
    def run_etl(self, start_season: int = 2006, end_season: int = 2025):
        """
        Run the complete ETL process for betting lines
        
        Args:
            start_season: Starting season year
            end_season: Ending season year
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Create table if needed
            self.create_table(cursor)
            conn.commit()
            
            total_inserted = 0
            
            # Process each season
            for season in range(start_season, end_season + 1):
                print(f"\n{'='*60}")
                print(f"Processing betting lines for season {season}...")
                print(f"{'='*60}")
                
                games_with_lines = self.fetch_all_lines_for_season(season)
                
                if games_with_lines:
                    inserted = self.insert_lines(games_with_lines, cursor)
                    conn.commit()
                    total_inserted += inserted
                    print(f"✓ Inserted {inserted} betting lines for season {season}")
                
                # Be nice to the API between seasons
                self.rate_limit_sleep(2)
            
            print(f"\n{'='*60}")
            print(f"=== ETL Complete ===")
            print(f"{'='*60}")
            print(f"Total betting lines inserted: {total_inserted}")
            
        except Exception as e:
            print(f"ETL process error: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()


# Example usage
if __name__ == "__main__":
    # Configuration
    API_KEY = "your_api_key_here"
    
    # MSSQL connection string
    DB_CONNECTION = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=your_server_name;"
        "DATABASE=your_database_name;"
        "UID=your_username;"
        "PWD=your_password"
    )
    
    # Initialize and run ETL
    etl = LinesETL(API_KEY, DB_CONNECTION)
    
    # Process all seasons
    etl.run_etl(start_season=2006, end_season=2025)
    
    # Or process a single season
    # etl.run_etl(start_season=2024, end_season=2024)