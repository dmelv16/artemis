from etl.base_etl import BaseCollegeBasketballETL, GameIDFetcher
from typing import List, Dict


class SubstitutionsETL(BaseCollegeBasketballETL):
    """ETL for college basketball substitution data"""
    
    def create_table(self, cursor):
        """Create the substitutions table if it doesn't exist"""
        create_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='substitutions' AND xtype='U')
        CREATE TABLE substitutions (
            id INT IDENTITY(1,1) PRIMARY KEY,
            gameId INT NOT NULL,
            startDate DATETIME2,
            teamId INT,
            team NVARCHAR(100),
            conference NVARCHAR(50),
            athleteId INT,
            athlete NVARCHAR(200),
            position NVARCHAR(10),
            opponentId INT,
            opponent NVARCHAR(100),
            opponentConference NVARCHAR(50),
            
            -- Sub In details
            subInOpponentPoints INT,
            subInTeamPoints INT,
            subInSecondsRemaining INT,
            subInPeriod INT,
            
            -- Sub Out details
            subOutOpponentPoints INT,
            subOutTeamPoints INT,
            subOutSecondsRemaining INT,
            subOutPeriod INT,
            
            CONSTRAINT UC_Substitution UNIQUE (gameId, athleteId, subInPeriod, subInSecondsRemaining)
        )
        """
        cursor.execute(create_table_sql)
        print("Table 'substitutions' verified/created")
    
    def fetch_substitutions_for_game(self, game_id: int) -> List[Dict]:
        """
        Fetch substitutions for a specific game
        
        Args:
            game_id: The game ID
            
        Returns:
            List of substitution dictionaries
        """
        endpoint = f"/substitutions/game/{game_id}"
        return self.fetch_api(endpoint)
    
    def insert_substitutions(self, substitutions: List[Dict], cursor) -> int:
        """
        Insert substitutions into the database
        
        Args:
            substitutions: List of substitution dictionaries
            cursor: Database cursor
            
        Returns:
            Number of records inserted
        """
        insert_sql = """
        MERGE substitutions AS target
        USING (SELECT ? AS gameId, ? AS athleteId, ? AS subInPeriod, ? AS subInSecondsRemaining) AS source
        ON target.gameId = source.gameId 
           AND target.athleteId = source.athleteId 
           AND target.subInPeriod = source.subInPeriod
           AND target.subInSecondsRemaining = source.subInSecondsRemaining
        WHEN NOT MATCHED THEN
            INSERT (gameId, startDate, teamId, team, conference, athleteId, athlete, position,
                    opponentId, opponent, opponentConference,
                    subInOpponentPoints, subInTeamPoints, subInSecondsRemaining, subInPeriod,
                    subOutOpponentPoints, subOutTeamPoints, subOutSecondsRemaining, subOutPeriod)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for sub in substitutions:
            try:
                sub_in = sub.get('subIn', {})
                sub_out = sub.get('subOut', {})
                
                params = (
                    # For MERGE condition
                    sub.get('gameId'),
                    sub.get('athleteId'),
                    sub_in.get('period'),
                    sub_in.get('secondsRemaining'),
                    # For INSERT
                    sub.get('gameId'),
                    self.parse_datetime(sub.get('startDate')),
                    sub.get('teamId'),
                    sub.get('team'),
                    sub.get('conference'),
                    sub.get('athleteId'),
                    sub.get('athlete'),
                    sub.get('position'),
                    sub.get('opponentId'),
                    sub.get('opponent'),
                    sub.get('opponentConference'),
                    sub_in.get('opponentPoints'),
                    sub_in.get('teamPoints'),
                    sub_in.get('secondsRemaining'),
                    sub_in.get('period'),
                    sub_out.get('opponentPoints'),
                    sub_out.get('teamPoints'),
                    sub_out.get('secondsRemaining'),
                    sub_out.get('period')
                )
                
                if self.execute_merge(cursor, insert_sql, params, 
                                     f"game {sub.get('gameId')} athlete {sub.get('athleteId')}"):
                    inserted += 1
                    
            except Exception as e:
                print(f"Error processing substitution: {e}")
                continue
        
        return inserted
    
    def process_game(self, game_id: int, cursor) -> int:
        """
        Process substitutions for a single game
        
        Args:
            game_id: Game ID to process
            cursor: Database cursor
            
        Returns:
            Number of substitutions inserted
        """
        substitutions = self.fetch_substitutions_for_game(game_id)
        
        if not substitutions:
            return 0
        
        inserted = self.insert_substitutions(substitutions, cursor)
        return inserted
    
    def run_etl(self, start_season: int = 2006, end_season: int = 2025, 
                batch_size: int = 100):
        """
        Run the complete ETL process for substitutions
        
        Args:
            start_season: Starting season year
            end_season: Ending season year
            batch_size: Number of games to process before committing
        """
        # First, get all game IDs from the database
        fetcher = GameIDFetcher(self.conn_string)
        all_game_ids = fetcher.get_all_game_ids(start_season, end_season)
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Create table if needed
            self.create_table(cursor)
            conn.commit()
            
            total_inserted = 0
            
            # Process each season
            for season in range(start_season, end_season + 1):
                game_ids = all_game_ids.get(season, [])
                
                if not game_ids:
                    print(f"No games found for season {season}")
                    continue
                
                print(f"\n{'='*60}")
                print(f"Processing {len(game_ids)} games for season {season}...")
                print(f"{'='*60}")
                
                season_inserted = 0
                
                # Process games in batches
                for i, game_id in enumerate(game_ids, 1):
                    try:
                        inserted = self.process_game(game_id, cursor)
                        season_inserted += inserted
                        
                        # Commit in batches
                        if i % batch_size == 0:
                            conn.commit()
                            print(f"  Processed {i}/{len(game_ids)} games... ({season_inserted} substitutions)")
                        
                        # Rate limiting
                        self.rate_limit_sleep(0.3)
                        
                    except Exception as e:
                        print(f"Error processing game {game_id}: {e}")
                        continue
                
                # Final commit for the season
                conn.commit()
                total_inserted += season_inserted
                print(f"✓ Season {season} complete: {season_inserted} substitutions inserted")
            
            print(f"\n{'='*60}")
            print(f"=== ETL Complete ===")
            print(f"{'='*60}")
            print(f"Total substitutions inserted: {total_inserted}")
            
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
    etl = SubstitutionsETL(API_KEY, DB_CONNECTION)
    
    # Process all seasons
    etl.run_etl(start_season=2006, end_season=2025, batch_size=100)
    
    # Or process a single season
    # etl.run_etl(start_season=2024, end_season=2024, batch_size=50)