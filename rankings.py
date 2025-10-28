from base_etl import BaseCollegeBasketballETL
from typing import List, Dict


class RankingsETL(BaseCollegeBasketballETL):
    """ETL for college basketball rankings/polls data"""
    
    def create_table(self, cursor):
        """Create the rankings table if it doesn't exist"""
        create_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='rankings' AND xtype='U')
        CREATE TABLE rankings (
            id INT IDENTITY(1,1) PRIMARY KEY,
            season INT NOT NULL,
            seasonType NVARCHAR(20),
            week INT NOT NULL,
            pollDate DATE,
            pollType NVARCHAR(50) NOT NULL,
            teamId INT NOT NULL,
            team NVARCHAR(100),
            conference NVARCHAR(50),
            ranking INT,
            points INT,
            firstPlaceVotes INT,
            
            CONSTRAINT UC_Ranking UNIQUE (season, week, pollType, teamId),
            INDEX IX_Rankings_Season (season),
            INDEX IX_Rankings_Week (week),
            INDEX IX_Rankings_TeamId (teamId),
            INDEX IX_Rankings_PollType (pollType)
        )
        """
        cursor.execute(create_table_sql)
        print("Table 'rankings' verified/created")
    
    def fetch_rankings(self, season: int, week: int) -> List[Dict]:
        """
        Fetch rankings for a specific season and week
        
        Args:
            season: Season year
            week: Week number
            
        Returns:
            List of ranking dictionaries
        """
        endpoint = "/rankings"
        params = {
            "season": season,
            "week": week
        }
        return self.fetch_api(endpoint, params)
    
    def insert_rankings(self, rankings: List[Dict], cursor) -> int:
        """
        Insert rankings into the database
        
        Args:
            rankings: List of ranking dictionaries
            cursor: Database cursor
            
        Returns:
            Number of records inserted
        """
        insert_sql = """
        MERGE rankings AS target
        USING (SELECT ? AS season, ? AS week, ? AS pollType, ? AS teamId) AS source
        ON target.season = source.season 
           AND target.week = source.week 
           AND target.pollType = source.pollType
           AND target.teamId = source.teamId
        WHEN NOT MATCHED THEN
            INSERT (season, seasonType, week, pollDate, pollType, teamId, team, 
                    conference, ranking, points, firstPlaceVotes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for rank in rankings:
            try:
                # Parse poll date if present
                poll_date = rank.get('pollDate')
                if poll_date:
                    try:
                        from datetime import datetime
                        poll_date = datetime.strptime(poll_date, '%Y-%m-%d').date()
                    except:
                        poll_date = None
                
                params = (
                    # For MERGE condition
                    rank.get('season'),
                    rank.get('week'),
                    rank.get('pollType'),
                    rank.get('teamId'),
                    # For INSERT
                    rank.get('season'),
                    rank.get('seasonType'),
                    rank.get('week'),
                    poll_date,
                    rank.get('pollType'),
                    rank.get('teamId'),
                    rank.get('team'),
                    rank.get('conference'),
                    rank.get('ranking'),
                    rank.get('points'),
                    rank.get('firstPlaceVotes')
                )
                
                if self.execute_merge(cursor, insert_sql, params, 
                                     f"season {rank.get('season')} week {rank.get('week')} {rank.get('team')}"):
                    inserted += 1
                    
            except Exception as e:
                print(f"Error processing ranking: {e}")
                continue
        
        return inserted
    
    def process_season(self, season: int, cursor, max_weeks: int = 25) -> int:
        """
        Process all weeks for a given season
        
        Args:
            season: Season year
            cursor: Database cursor
            max_weeks: Maximum number of weeks to try (default 25)
            
        Returns:
            Total number of rankings inserted for the season
        """
        season_inserted = 0
        consecutive_empty = 0
        
        for week in range(1, max_weeks + 1):
            rankings = self.fetch_rankings(season, week)
            
            if not rankings:
                consecutive_empty += 1
                # If we get 3 consecutive empty weeks, assume season is done
                if consecutive_empty >= 3:
                    print(f"  No more rankings found after week {week - 3}")
                    break
                continue
            else:
                consecutive_empty = 0  # Reset counter when we find data
            
            inserted = self.insert_rankings(rankings, cursor)
            season_inserted += inserted
            
            if inserted > 0:
                print(f"  Week {week}: {inserted} rankings")
            
            # Rate limiting
            self.rate_limit_sleep(0.3)
        
        return season_inserted
    
    def run_etl(self, start_season: int = 2006, end_season: int = 2025, 
                batch_commit_weeks: int = 5):
        """
        Run the complete ETL process for rankings
        
        Args:
            start_season: Starting season year
            end_season: Ending season year
            batch_commit_weeks: Number of weeks to process before committing
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
                print(f"Processing rankings for season {season}...")
                print(f"{'='*60}")
                
                season_inserted = 0
                consecutive_empty = 0
                
                # Process weeks 1-25 (covers full season including postseason)
                for week in range(1, 26):
                    rankings = self.fetch_rankings(season, week)
                    
                    if not rankings:
                        consecutive_empty += 1
                        # If we get 3 consecutive empty weeks, assume season is done
                        if consecutive_empty >= 3:
                            print(f"  No more rankings found after week {week - 3}")
                            break
                        continue
                    else:
                        consecutive_empty = 0  # Reset counter when we find data
                    
                    inserted = self.insert_rankings(rankings, cursor)
                    season_inserted += inserted
                    
                    if inserted > 0:
                        print(f"  Week {week}: {inserted} rankings")
                    
                    # Commit in batches
                    if week % batch_commit_weeks == 0:
                        conn.commit()
                    
                    # Rate limiting
                    self.rate_limit_sleep(0.3)
                
                # Final commit for the season
                conn.commit()
                total_inserted += season_inserted
                print(f"✓ Season {season} complete: {season_inserted} rankings inserted")
            
            print(f"\n{'='*60}")
            print(f"=== ETL Complete ===")
            print(f"{'='*60}")
            print(f"Total rankings inserted: {total_inserted}")
            
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
    etl = RankingsETL(API_KEY, DB_CONNECTION)
    
    # Process all seasons
    etl.run_etl(start_season=2006, end_season=2025)
    
    # Or process a single season
    # etl.run_etl(start_season=2024, end_season=2024)