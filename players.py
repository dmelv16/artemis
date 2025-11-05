from base_etl import BaseCollegeBasketballETL
from typing import List, Dict
from datetime import datetime, timedelta


class PlayerGamesETL(BaseCollegeBasketballETL):
    """ETL for college basketball player game statistics"""
    
    def create_table(self, cursor):
        """Create the player_games table if it doesn't exist"""
        create_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='player_games' AND xtype='U')
        CREATE TABLE player_games (
            id INT IDENTITY(1,1) PRIMARY KEY,
            gameId INT NOT NULL,
            season INT,
            seasonLabel NVARCHAR(20),
            seasonType NVARCHAR(20),
            tournament NVARCHAR(50),
            startDate DATETIME2,
            startTimeTbd BIT,
            conferenceGame BIT,
            neutralSite BIT,
            isHome BIT,
            gameType NVARCHAR(10),
            notes NVARCHAR(500),
            teamId INT,
            team NVARCHAR(100),
            conference NVARCHAR(50),
            teamSeed INT,
            opponentId INT,
            opponent NVARCHAR(100),
            opponentConference NVARCHAR(50),
            opponentSeed INT,
            gameMinutes INT,
            gamePace FLOAT,
            
            -- Player Info
            athleteId INT,
            athleteSourceId NVARCHAR(50),
            name NVARCHAR(200),
            position NVARCHAR(10),
            starter BIT,
            ejected BIT,
            
            -- Player Stats
            minutes INT,
            points INT,
            turnovers INT,
            fouls INT,
            assists INT,
            steals INT,
            blocks INT,
            gameScore FLOAT,
            offensiveRating FLOAT,
            defensiveRating FLOAT,
            netRating FLOAT,
            usage FLOAT,
            effectiveFieldGoalPct FLOAT,
            trueShootingPct FLOAT,
            assistsTurnoverRatio FLOAT,
            freeThrowRate FLOAT,
            offensiveReboundPct FLOAT,
            
            -- Field Goals
            fieldGoalsMade INT,
            fieldGoalsAttempted INT,
            fieldGoalsPct FLOAT,
            twoPointFGMade INT,
            twoPointFGAttempted INT,
            twoPointFGPct FLOAT,
            threePointFGMade INT,
            threePointFGAttempted INT,
            threePointFGPct FLOAT,
            freeThrowsMade INT,
            freeThrowsAttempted INT,
            freeThrowsPct FLOAT,
            
            -- Rebounds
            offensiveRebounds INT,
            defensiveRebounds INT,
            totalRebounds INT,
            
            CONSTRAINT UC_PlayerGame UNIQUE (gameId, athleteId),
            INDEX IX_PlayerGames_GameId (gameId),
            INDEX IX_PlayerGames_AthleteId (athleteId),
            INDEX IX_PlayerGames_Season (season),
            INDEX IX_PlayerGames_TeamId (teamId)
        )
        """
        cursor.execute(create_table_sql)
        print("Table 'player_games' verified/created")
    
    def fetch_player_games_for_season(self, season: int) -> List[Dict]:
        """Fetch all player games for a season with pagination (1000 record limit)"""
        all_games = []
        date_ranges = self.get_season_date_ranges(season)
        
        print(f"Fetching player games for season {season}...")
        
        for start_date, end_date in date_ranges:
            params = {
                "season": season,
                "startDateRange": start_date,
                "endDateRange": end_date
            }
            
            games = self.fetch_api("/games/players", params)
            
            if games:
                print(f"  {start_date[:7]}: {len(games)} game records")
                
                # If we hit the 1000 limit, break into weeks
                if len(games) >= 1000:
                    print(f"  Hit 1000 limit, fetching by weeks...")
                    games = self._fetch_by_week(season, start_date, end_date)
                
                all_games.extend(games)
            
            self.rate_limit_sleep(0.5)
        
        # Remove duplicates based on gameId + athleteId
        unique = {}
        for game in all_games:
            key = f"{game.get('gameId')}_{game.get('athleteId')}"
            unique[key] = game
        
        print(f"Total unique player-game records: {len(unique)}")
        return list(unique.values())
    
    def _fetch_by_week(self, season: int, month_start: str, month_end: str) -> List[Dict]:
        """Helper to fetch by week when hitting limits"""
        all_games = []
        start = datetime.fromisoformat(month_start.replace('Z', '+00:00'))
        end = datetime.fromisoformat(month_end.replace('Z', '+00:00'))
        
        current = start
        while current < end:
            week_end = min(current + timedelta(days=7), end)
            params = {
                "season": season,
                "startDateRange": current.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "endDateRange": week_end.strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            
            games = self.fetch_api("/games/players", params)
            if games:
                all_games.extend(games)
            
            current = week_end
            self.rate_limit_sleep(0.5)
        
        return all_games
    
    def insert_player_games(self, player_games_data: List[Dict], cursor) -> int:
        """
        Insert player games into the database
        
        Args:
            player_games_data: List of player game dictionaries (each contains game info + players array)
            cursor: Database cursor
            
        Returns:
            Number of records inserted
        """
        insert_sql = """
        MERGE player_games AS target
        USING (SELECT ? AS gameId, ? AS athleteId) AS source
        ON target.gameId = source.gameId AND target.athleteId = source.athleteId
        WHEN NOT MATCHED THEN
            INSERT (gameId, season, seasonLabel, seasonType, tournament, startDate, startTimeTbd,
                    conferenceGame, neutralSite, isHome, gameType, notes,
                    teamId, team, conference, teamSeed, opponentId, opponent, opponentConference, opponentSeed,
                    gameMinutes, gamePace,
                    athleteId, athleteSourceId, name, position, starter, ejected,
                    minutes, points, turnovers, fouls, assists, steals, blocks,
                    gameScore, offensiveRating, defensiveRating, netRating, usage,
                    effectiveFieldGoalPct, trueShootingPct, assistsTurnoverRatio, freeThrowRate, offensiveReboundPct,
                    fieldGoalsMade, fieldGoalsAttempted, fieldGoalsPct,
                    twoPointFGMade, twoPointFGAttempted, twoPointFGPct,
                    threePointFGMade, threePointFGAttempted, threePointFGPct,
                    freeThrowsMade, freeThrowsAttempted, freeThrowsPct,
                    offensiveRebounds, defensiveRebounds, totalRebounds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for game_data in player_games_data:
            # Parse startDate once for this game
            start_date = self.parse_datetime(game_data.get('startDate'))
            
            # Process each player in the game
            players = game_data.get('players', [])
            for player in players:
                try:
                    # Extract nested shooting stats
                    fg = player.get('fieldGoals', {})
                    two_pt = player.get('twoPointFieldGoals', {})
                    three_pt = player.get('threePointFieldGoals', {})
                    ft = player.get('freeThrows', {})
                    reb = player.get('rebounds', {})
                    
                    params = (
                        # For MERGE condition
                        game_data.get('gameId'), 
                        player.get('athleteId'),
                        # For INSERT
                        game_data.get('gameId'), 
                        game_data.get('season'), 
                        game_data.get('seasonLabel'),
                        game_data.get('seasonType'), 
                        game_data.get('tournament'), 
                        start_date,
                        game_data.get('startTimeTbd', False), 
                        game_data.get('conferenceGame', False),
                        game_data.get('neutralSite', False), 
                        game_data.get('isHome', False),
                        game_data.get('gameType'), 
                        game_data.get('notes'),
                        game_data.get('teamId'), 
                        game_data.get('team'), 
                        game_data.get('conference'),
                        game_data.get('teamSeed'), 
                        game_data.get('opponentId'), 
                        game_data.get('opponent'),
                        game_data.get('opponentConference'), 
                        game_data.get('opponentSeed'),
                        game_data.get('gameMinutes'), 
                        game_data.get('gamePace'),
                        # Player info
                        player.get('athleteId'), 
                        player.get('athleteSourceId'), 
                        player.get('name'),
                        player.get('position'), 
                        player.get('starter', False), 
                        player.get('ejected', False),
                        # Player stats
                        player.get('minutes'), 
                        player.get('points'), 
                        player.get('turnovers'),
                        player.get('fouls'), 
                        player.get('assists'), 
                        player.get('steals'), 
                        player.get('blocks'),
                        player.get('gameScore'), 
                        player.get('offensiveRating'), 
                        player.get('defensiveRating'),
                        player.get('netRating'), 
                        player.get('usage'), 
                        player.get('effectiveFieldGoalPct'),
                        player.get('trueShootingPct'), 
                        player.get('assistsTurnoverRatio'),
                        player.get('freeThrowRate'), 
                        player.get('offensiveReboundPct'),
                        # Shooting
                        fg.get('made'), 
                        fg.get('attempted'), 
                        fg.get('pct'),
                        two_pt.get('made'), 
                        two_pt.get('attempted'), 
                        two_pt.get('pct'),
                        three_pt.get('made'), 
                        three_pt.get('attempted'), 
                        three_pt.get('pct'),
                        ft.get('made'), 
                        ft.get('attempted'), 
                        ft.get('pct'),
                        # Rebounds
                        reb.get('offensive'), 
                        reb.get('defensive'), 
                        reb.get('total')
                    )
                    
                    if self.execute_merge(cursor, insert_sql, params, 
                                         f"player {player.get('name')} in game {game_data.get('gameId')}"):
                        inserted += 1
                    
                except Exception as e:
                    print(f"Error inserting player {player.get('name')} in game {game_data.get('gameId')}: {e}")
                    continue
        
        return inserted
    
    def run_etl(self, start_season: int = 2006, end_season: int = 2025):
        """
        Run the complete ETL process for player games
        
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
                print(f"Processing player games for season {season}...")
                print(f"{'='*60}")
                
                player_games = self.fetch_player_games_for_season(season)
                
                if player_games:
                    inserted = self.insert_player_games(player_games, cursor)
                    conn.commit()
                    total_inserted += inserted
                    print(f"✓ Inserted {inserted} player game records for season {season}")
                
                # Be nice to the API between seasons
                self.rate_limit_sleep(2)
            
            print(f"\n{'='*60}")
            print(f"=== ETL Complete ===")
            print(f"{'='*60}")
            print(f"Total player game records inserted: {total_inserted}")
            
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
    etl = PlayerGamesETL(API_KEY, DB_CONNECTION)
    
    # Process all seasons
    etl.run_etl(start_season=2006, end_season=2025)
    
    # Or process a single season
    # etl.run_etl(start_season=2024, end_season=2024)