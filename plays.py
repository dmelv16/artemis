from base_etl import BaseCollegeBasketballETL, GameIDFetcher
from typing import List, Dict


class PlaysETL(BaseCollegeBasketballETL):
    """ETL for college basketball play-by-play data"""
    
    def create_table(self, cursor):
        """Create the plays table if it doesn't exist"""
        create_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='plays' AND xtype='U')
        CREATE TABLE plays (
            id INT PRIMARY KEY,
            sourceId NVARCHAR(50),
            gameId INT NOT NULL,
            gameSourceId NVARCHAR(50),
            gameStartDate DATETIME2,
            season INT,
            seasonType NVARCHAR(20),
            gameType NVARCHAR(10),
            tournament NVARCHAR(50),
            playType NVARCHAR(50),
            isHomeTeam BIT,
            teamId INT,
            team NVARCHAR(100),
            conference NVARCHAR(50),
            teamSeed INT,
            opponentId INT,
            opponent NVARCHAR(100),
            opponentConference NVARCHAR(50),
            opponentSeed INT,
            period INT,
            clock NVARCHAR(20),
            secondsRemaining INT,
            homeScore INT,
            awayScore INT,
            homeWinProbability FLOAT,
            scoringPlay BIT,
            shootingPlay BIT,
            scoreValue INT,
            wallclock DATETIME2,
            playText NVARCHAR(MAX),
            
            -- Participants array stored as JSON
            participants NVARCHAR(MAX),
            
            -- OnFloor array stored as JSON
            onFloor NVARCHAR(MAX),
            
            -- Shot Info fields
            shotShooterName NVARCHAR(200),
            shotShooterId INT,
            shotMade BIT,
            shotRange NVARCHAR(20),
            shotAssisted BIT,
            shotAssistedByName NVARCHAR(200),
            shotAssistedById INT,
            shotLocationX FLOAT,
            shotLocationY FLOAT,
            
            INDEX IX_Plays_GameId (gameId),
            INDEX IX_Plays_Season (season),
            INDEX IX_Plays_TeamId (teamId)
        )
        """
        cursor.execute(create_table_sql)
        print("Table 'plays' verified/created")
    
    def fetch_plays_for_game(self, game_id: int) -> List[Dict]:
        """
        Fetch play-by-play data for a specific game
        
        Args:
            game_id: The game ID
            
        Returns:
            List of play dictionaries
        """
        endpoint = f"/plays/game/{game_id}"
        return self.fetch_api(endpoint)
    
    def insert_plays(self, plays: List[Dict], cursor) -> int:
        """
        Insert plays into the database
        
        Args:
            plays: List of play dictionaries
            cursor: Database cursor
            
        Returns:
            Number of records inserted
        """
        insert_sql = """
        MERGE plays AS target
        USING (SELECT ? AS id) AS source
        ON target.id = source.id
        WHEN NOT MATCHED THEN
            INSERT (id, sourceId, gameId, gameSourceId, gameStartDate, season, seasonType, 
                    gameType, tournament, playType, isHomeTeam, teamId, team, conference, 
                    teamSeed, opponentId, opponent, opponentConference, opponentSeed,
                    period, clock, secondsRemaining, homeScore, awayScore, homeWinProbability,
                    scoringPlay, shootingPlay, scoreValue, wallclock, playText,
                    participants, onFloor,
                    shotShooterName, shotShooterId, shotMade, shotRange, shotAssisted,
                    shotAssistedByName, shotAssistedById, shotLocationX, shotLocationY)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for play in plays:
            try:
                shot_info = play.get('shotInfo', {})
                shooter = shot_info.get('shooter', {}) if shot_info else {}
                assisted_by = shot_info.get('assistedBy', {}) if shot_info else {}
                location = shot_info.get('location', {}) if shot_info else {}
                
                params = (
                    # For MERGE condition
                    play.get('id'),
                    # For INSERT
                    play.get('id'),
                    play.get('sourceId'),
                    play.get('gameId'),
                    play.get('gameSourceId'),
                    self.parse_datetime(play.get('gameStartDate')),
                    play.get('season'),
                    play.get('seasonType'),
                    play.get('gameType'),
                    play.get('tournament'),
                    play.get('playType'),
                    play.get('isHomeTeam'),
                    play.get('teamId'),
                    play.get('team'),
                    play.get('conference'),
                    play.get('teamSeed'),
                    play.get('opponentId'),
                    play.get('opponent'),
                    play.get('opponentConference'),
                    play.get('opponentSeed'),
                    play.get('period'),
                    play.get('clock'),
                    play.get('secondsRemaining'),
                    play.get('homeScore'),
                    play.get('awayScore'),
                    play.get('homeWinProbability'),
                    play.get('scoringPlay'),
                    play.get('shootingPlay'),
                    play.get('scoreValue'),
                    self.parse_datetime(play.get('wallclock')),
                    play.get('playText'),
                    # JSON arrays
                    self.json_serialize(play.get('participants')),
                    self.json_serialize(play.get('onFloor')),
                    # Shot info
                    shooter.get('name'),
                    shooter.get('id'),
                    shot_info.get('made') if shot_info else None,
                    shot_info.get('range') if shot_info else None,
                    shot_info.get('assisted') if shot_info else None,
                    assisted_by.get('name'),
                    assisted_by.get('id'),
                    location.get('x'),
                    location.get('y')
                )
                
                if self.execute_merge(cursor, insert_sql, params, 
                                     f"play {play.get('id')}"):
                    inserted += 1
                    
            except Exception as e:
                print(f"Error processing play {play.get('id')}: {e}")
                continue
        
        return inserted
    
    def process_game(self, game_id: int, cursor) -> int:
        """
        Process plays for a single game
        
        Args:
            game_id: Game ID to process
            cursor: Database cursor
            
        Returns:
            Number of plays inserted
        """
        plays = self.fetch_plays_for_game(game_id)
        
        if not plays:
            return 0
        
        inserted = self.insert_plays(plays, cursor)
        return inserted
    
    def run_etl(self, start_season: int = 2006, end_season: int = 2025, 
                batch_size: int = 50):
        """
        Run the complete ETL process for play-by-play data
        
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
                            print(f"  Processed {i}/{len(game_ids)} games... ({season_inserted} plays)")
                        
                        # Rate limiting - plays endpoint may have more data
                        self.rate_limit_sleep(0.5)
                        
                    except Exception as e:
                        print(f"Error processing game {game_id}: {e}")
                        continue
                
                # Final commit for the season
                conn.commit()
                total_inserted += season_inserted
                print(f"✓ Season {season} complete: {season_inserted} plays inserted")
            
            print(f"\n{'='*60}")
            print(f"=== ETL Complete ===")
            print(f"{'='*60}")
            print(f"Total plays inserted: {total_inserted}")
            
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
    etl = PlaysETL(API_KEY, DB_CONNECTION)
    
    # Process all seasons
    etl.run_etl(start_season=2006, end_season=2025, batch_size=50)
    
    # Or process a single season
    # etl.run_etl(start_season=2024, end_season=2024, batch_size=25)