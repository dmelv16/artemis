import requests
import pyodbc
import json
from datetime import datetime
from typing import List, Dict
import time

class CollegeBasketballETL:
    def __init__(self, api_key: str, db_connection_string: str):
        """
        Initialize the ETL process
        
        Args:
            api_key: Your API key for collegebasketballdata.com
            db_connection_string: MSSQL connection string
        """
        self.api_key = api_key
        self.games_url = "https://api.collegebasketballdata.com/games"
        self.team_games_url = "https://api.collegebasketballdata.com/games/teams"
        self.player_games_url = "https://api.collegebasketballdata.com/games/players"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.conn_string = db_connection_string
        
    def fetch_games_batch(self, season: int, start_date: str = None, end_date: str = None, endpoint: str = "games") -> List[Dict]:
        """
        Fetch a batch of games (up to 3000)
        
        Args:
            season: Season year (e.g., 2006 for 2005-06 season)
            start_date: Optional start date in ISO format
            end_date: Optional end date in ISO format
            endpoint: Which endpoint to use - "games" or "team_games"
            
        Returns:
            List of game dictionaries
        """
        url = self.games_url if endpoint == "games" else self.team_games_url
        params = {"season": season}
        if start_date:
            params["startDateRange"] = start_date
        if end_date:
            params["endDateRange"] = end_date
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            games = response.json()
            return games
        except requests.exceptions.RequestException as e:
            print(f"Error fetching games: {e}")
            return []
    
    def fetch_all_games_for_season(self, season: int, endpoint: str = "games") -> List[Dict]:
        """
        Fetch ALL games for a season by breaking into monthly chunks if needed
        
        Args:
            season: Season year (e.g., 2006 for 2005-06 season)
            endpoint: Which endpoint to use - "games" or "team_games"
            
        Returns:
            List of all game dictionaries for the season
        """
        all_games = []
        
        # College basketball season runs from November to April (of next year)
        # For season 2006, that's Nov 2005 - Apr 2006
        start_year = season - 1
        end_year = season
        
        # Define monthly date ranges
        date_ranges = [
            (f"{start_year}-11-01T00:00:00Z", f"{start_year}-12-01T00:00:00Z"),  # November
            (f"{start_year}-12-01T00:00:00Z", f"{end_year}-01-01T00:00:00Z"),    # December
            (f"{end_year}-01-01T00:00:00Z", f"{end_year}-02-01T00:00:00Z"),      # January
            (f"{end_year}-02-01T00:00:00Z", f"{end_year}-03-01T00:00:00Z"),      # February
            (f"{end_year}-03-01T00:00:00Z", f"{end_year}-04-01T00:00:00Z"),      # March
            (f"{end_year}-04-01T00:00:00Z", f"{end_year}-05-01T00:00:00Z"),      # April
        ]
        
        print(f"Fetching season {season} in monthly chunks...")
        
        for start_date, end_date in date_ranges:
            games = self.fetch_games_batch(season, start_date, end_date, endpoint)
            
            if games:
                print(f"  {start_date[:7]}: {len(games)} games")
                
                # If we hit the 3000 limit, break the month into weeks
                if len(games) >= 3000:
                    print(f"  Warning: Hit 3000 game limit for {start_date[:7]}, fetching by weeks...")
                    monthly_games = self.fetch_games_by_week(season, start_date, end_date, endpoint)
                    all_games.extend(monthly_games)
                else:
                    all_games.extend(games)
            
            time.sleep(0.5)  # Be nice to the API
        
        # Remove duplicates based on game id (or gameId for team_games)
        id_field = 'gameId' if endpoint == "team_games" else 'id'
        unique_games = {game[id_field]: game for game in all_games if id_field in game}.values()
        print(f"Total unique games for season {season}: {len(unique_games)}")
        
        return list(unique_games)
    
    def fetch_games_by_week(self, season: int, month_start: str, month_end: str, endpoint: str = "games") -> List[Dict]:
        """
        Fetch games by week for months with >3000 games
        
        Args:
            season: Season year
            month_start: Start of month in ISO format
            month_end: End of month in ISO format
            endpoint: Which endpoint to use - "games" or "team_games"
            
        Returns:
            List of game dictionaries
        """
        from datetime import datetime, timedelta
        
        all_games = []
        start = datetime.fromisoformat(month_start.replace('Z', '+00:00'))
        end = datetime.fromisoformat(month_end.replace('Z', '+00:00'))
        
        current = start
        while current < end:
            week_end = min(current + timedelta(days=7), end)
            
            start_str = current.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_str = week_end.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            games = self.fetch_games_batch(season, start_str, end_str, endpoint)
            
            if games:
                print(f"    {start_str[:10]} to {end_str[:10]}: {len(games)} games")
                all_games.extend(games)
            
            current = week_end
            time.sleep(0.5)
        
        return all_games
    
    def create_table_if_not_exists(self, cursor):
        """Create the games and team_games tables if they don't exist"""
        create_games_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='games' AND xtype='U')
        CREATE TABLE games (
            id INT PRIMARY KEY,
            sourceId NVARCHAR(50),
            seasonLabel NVARCHAR(20),
            season INT,
            seasonType NVARCHAR(20),
            tournament NVARCHAR(50),
            startDate DATETIME2,
            startTimeTbd BIT,
            neutralSite BIT,
            conferenceGame BIT,
            gameType NVARCHAR(10),
            status NVARCHAR(20),
            gameNotes NVARCHAR(500),
            attendance INT,
            homeTeamId INT,
            homeTeam NVARCHAR(100),
            homeConferenceId INT,
            homeConference NVARCHAR(50),
            homeSeed INT,
            homePoints INT,
            homePeriodPoints NVARCHAR(MAX),
            homeWinner BIT,
            awayTeamId INT,
            awayTeam NVARCHAR(100),
            awayConferenceId INT,
            awayConference NVARCHAR(50),
            awaySeed INT,
            awayPoints INT,
            awayPeriodPoints NVARCHAR(MAX),
            awayWinner BIT,
            excitement FLOAT,
            venueId INT,
            venue NVARCHAR(200),
            city NVARCHAR(100),
            state NVARCHAR(10)
        )
        """
        
        create_team_games_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='team_games' AND xtype='U')
        CREATE TABLE team_games (
            id INT IDENTITY(1,1) PRIMARY KEY,
            gameId INT NOT NULL,
            season INT,
            seasonLabel NVARCHAR(20),
            seasonType NVARCHAR(20),
            tournament NVARCHAR(50),
            startDate DATETIME2,
            startTimeTbd BIT,
            teamId INT,
            team NVARCHAR(100),
            conference NVARCHAR(50),
            teamSeed INT,
            opponentId INT,
            opponent NVARCHAR(100),
            opponentConference NVARCHAR(50),
            opponentSeed INT,
            neutralSite BIT,
            isHome BIT,
            conferenceGame BIT,
            gameType NVARCHAR(10),
            notes NVARCHAR(500),
            gameMinutes INT,
            pace FLOAT,
            
            -- Team Stats
            possessions FLOAT,
            assists INT,
            steals INT,
            blocks INT,
            trueShooting FLOAT,
            rating FLOAT,
            gameScore FLOAT,
            
            -- Team Points
            points INT,
            pointsByPeriod NVARCHAR(MAX),
            largestLead INT,
            fastBreakPoints INT,
            pointsInPaint INT,
            pointsOffTurnovers INT,
            
            -- Team Shooting
            twoPointFGMade INT,
            twoPointFGAttempted INT,
            twoPointFGPct FLOAT,
            threePointFGMade INT,
            threePointFGAttempted INT,
            threePointFGPct FLOAT,
            freeThrowsMade INT,
            freeThrowsAttempted INT,
            freeThrowsPct FLOAT,
            fieldGoalsMade INT,
            fieldGoalsAttempted INT,
            fieldGoalsPct FLOAT,
            
            -- Team Turnovers & Rebounds
            turnovers INT,
            teamTurnovers INT,
            offensiveRebounds INT,
            defensiveRebounds INT,
            totalRebounds INT,
            
            -- Team Fouls
            fouls INT,
            technicalFouls INT,
            flagrantFouls INT,
            
            -- Team Four Factors
            effectiveFieldGoalPct FLOAT,
            freeThrowRate FLOAT,
            turnoverRatio FLOAT,
            offensiveReboundPct FLOAT,
            
            -- Opponent Stats
            oppPossessions FLOAT,
            oppAssists INT,
            oppSteals INT,
            oppBlocks INT,
            oppTrueShooting FLOAT,
            oppRating FLOAT,
            oppGameScore FLOAT,
            
            -- Opponent Points
            oppPoints INT,
            oppPointsByPeriod NVARCHAR(MAX),
            oppLargestLead INT,
            oppFastBreakPoints INT,
            oppPointsInPaint INT,
            oppPointsOffTurnovers INT,
            
            -- Opponent Shooting
            oppTwoPointFGMade INT,
            oppTwoPointFGAttempted INT,
            oppTwoPointFGPct FLOAT,
            oppThreePointFGMade INT,
            oppThreePointFGAttempted INT,
            oppThreePointFGPct FLOAT,
            oppFreeThrowsMade INT,
            oppFreeThrowsAttempted INT,
            oppFreeThrowsPct FLOAT,
            oppFieldGoalsMade INT,
            oppFieldGoalsAttempted INT,
            oppFieldGoalsPct FLOAT,
            
            -- Opponent Turnovers & Rebounds
            oppTurnovers INT,
            oppTeamTurnovers INT,
            oppOffensiveRebounds INT,
            oppDefensiveRebounds INT,
            oppTotalRebounds INT,
            
            -- Opponent Fouls
            oppFouls INT,
            oppTechnicalFouls INT,
            oppFlagrantFouls INT,
            
            -- Opponent Four Factors
            oppEffectiveFieldGoalPct FLOAT,
            oppFreeThrowRate FLOAT,
            oppTurnoverRatio FLOAT,
            oppOffensiveReboundPct FLOAT,
            
            CONSTRAINT UC_TeamGame UNIQUE (gameId, teamId)
        )
        """
        
        cursor.execute(create_games_table_sql)
        cursor.execute(create_team_games_table_sql)
        print("Tables 'games' and 'team_games' verified/created")
    
    def insert_games(self, games: List[Dict], cursor):
        """
        Insert games into MSSQL database
        
        Args:
            games: List of game dictionaries
            cursor: Database cursor
        """
        insert_sql = """
        MERGE games AS target
        USING (SELECT ? AS id) AS source
        ON target.id = source.id
        WHEN NOT MATCHED THEN
            INSERT (id, sourceId, seasonLabel, season, seasonType, tournament,
                    startDate, startTimeTbd, neutralSite, conferenceGame, gameType,
                    status, gameNotes, attendance, homeTeamId, homeTeam,
                    homeConferenceId, homeConference, homeSeed, homePoints,
                    homePeriodPoints, homeWinner, awayTeamId, awayTeam,
                    awayConferenceId, awayConference, awaySeed, awayPoints,
                    awayPeriodPoints, awayWinner, excitement, venueId, venue,
                    city, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for game in games:
            try:
                # Convert period points to JSON string
                home_period = json.dumps(game.get('homePeriodPoints', []))
                away_period = json.dumps(game.get('awayPeriodPoints', []))
                
                # Parse startDate if it's a string
                start_date = game.get('startDate')
                if isinstance(start_date, str):
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                
                params = (
                    game.get('id'),  # For MERGE condition
                    game.get('id'),
                    game.get('sourceId'),
                    game.get('seasonLabel'),
                    game.get('season'),
                    game.get('seasonType'),
                    game.get('tournament'),
                    start_date,
                    game.get('startTimeTbd', False),
                    game.get('neutralSite', False),
                    game.get('conferenceGame', False),
                    game.get('gameType'),
                    game.get('status'),
                    game.get('gameNotes'),
                    game.get('attendance'),
                    game.get('homeTeamId'),
                    game.get('homeTeam'),
                    game.get('homeConferenceId'),
                    game.get('homeConference'),
                    game.get('homeSeed'),
                    game.get('homePoints'),
                    home_period,
                    game.get('homeWinner', False),
                    game.get('awayTeamId'),
                    game.get('awayTeam'),
                    game.get('awayConferenceId'),
                    game.get('awayConference'),
                    game.get('awaySeed'),
                    game.get('awayPoints'),
                    away_period,
                    game.get('awayWinner', False),
                    game.get('excitement'),
                    game.get('venueId'),
                    game.get('venue'),
                    game.get('city'),
                    game.get('state')
                )
                
                cursor.execute(insert_sql, params)
                inserted += 1
                
            except Exception as e:
                print(f"Error inserting game {game.get('id')}: {e}")
                continue
        
        return inserted
    
    def insert_team_games(self, team_games: List[Dict], cursor):
        """
        Insert team games into MSSQL database
        
        Args:
            team_games: List of team game dictionaries
            cursor: Database cursor
        """
        insert_sql = """
        MERGE team_games AS target
        USING (SELECT ? AS gameId, ? AS teamId) AS source
        ON target.gameId = source.gameId AND target.teamId = source.teamId
        WHEN NOT MATCHED THEN
            INSERT (gameId, season, seasonLabel, seasonType, tournament, startDate, startTimeTbd,
                    teamId, team, conference, teamSeed, opponentId, opponent, opponentConference,
                    opponentSeed, neutralSite, isHome, conferenceGame, gameType, notes, gameMinutes, pace,
                    possessions, assists, steals, blocks, trueShooting, rating, gameScore,
                    points, pointsByPeriod, largestLead, fastBreakPoints, pointsInPaint, pointsOffTurnovers,
                    twoPointFGMade, twoPointFGAttempted, twoPointFGPct, threePointFGMade, threePointFGAttempted, threePointFGPct,
                    freeThrowsMade, freeThrowsAttempted, freeThrowsPct, fieldGoalsMade, fieldGoalsAttempted, fieldGoalsPct,
                    turnovers, teamTurnovers, offensiveRebounds, defensiveRebounds, totalRebounds,
                    fouls, technicalFouls, flagrantFouls,
                    effectiveFieldGoalPct, freeThrowRate, turnoverRatio, offensiveReboundPct,
                    oppPossessions, oppAssists, oppSteals, oppBlocks, oppTrueShooting, oppRating, oppGameScore,
                    oppPoints, oppPointsByPeriod, oppLargestLead, oppFastBreakPoints, oppPointsInPaint, oppPointsOffTurnovers,
                    oppTwoPointFGMade, oppTwoPointFGAttempted, oppTwoPointFGPct, oppThreePointFGMade, oppThreePointFGAttempted, oppThreePointFGPct,
                    oppFreeThrowsMade, oppFreeThrowsAttempted, oppFreeThrowsPct, oppFieldGoalsMade, oppFieldGoalsAttempted, oppFieldGoalsPct,
                    oppTurnovers, oppTeamTurnovers, oppOffensiveRebounds, oppDefensiveRebounds, oppTotalRebounds,
                    oppFouls, oppTechnicalFouls, oppFlagrantFouls,
                    oppEffectiveFieldGoalPct, oppFreeThrowRate, oppTurnoverRatio, oppOffensiveReboundPct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for tg in team_games:
            try:
                # Parse startDate
                start_date = tg.get('startDate')
                if isinstance(start_date, str):
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                
                # Extract nested stats
                team_stats = tg.get('teamStats', {})
                opp_stats = tg.get('opponentStats', {})
                
                team_points = team_stats.get('points', {})
                opp_points = opp_stats.get('points', {})
                
                team_2pt = team_stats.get('twoPointFieldGoals', {})
                team_3pt = team_stats.get('threePointFieldGoals', {})
                team_ft = team_stats.get('freeThrows', {})
                team_fg = team_stats.get('fieldGoals', {})
                team_to = team_stats.get('turnovers', {})
                team_reb = team_stats.get('rebounds', {})
                team_fouls = team_stats.get('fouls', {})
                team_ff = team_stats.get('fourFactors', {})
                
                opp_2pt = opp_stats.get('twoPointFieldGoals', {})
                opp_3pt = opp_stats.get('threePointFieldGoals', {})
                opp_ft = opp_stats.get('freeThrows', {})
                opp_fg = opp_stats.get('fieldGoals', {})
                opp_to = opp_stats.get('turnovers', {})
                opp_reb = opp_stats.get('rebounds', {})
                opp_fouls = opp_stats.get('fouls', {})
                opp_ff = opp_stats.get('fourFactors', {})
                
                params = (
                    tg.get('gameId'), tg.get('teamId'),  # For MERGE
                    tg.get('gameId'), tg.get('season'), tg.get('seasonLabel'), tg.get('seasonType'),
                    tg.get('tournament'), start_date, tg.get('startTimeTbd', False),
                    tg.get('teamId'), tg.get('team'), tg.get('conference'), tg.get('teamSeed'),
                    tg.get('opponentId'), tg.get('opponent'), tg.get('opponentConference'), tg.get('opponentSeed'),
                    tg.get('neutralSite', False), tg.get('isHome', False), tg.get('conferenceGame', False),
                    tg.get('gameType'), tg.get('notes'), tg.get('gameMinutes'), tg.get('pace'),
                    # Team stats
                    team_stats.get('possessions'), team_stats.get('assists'), team_stats.get('steals'),
                    team_stats.get('blocks'), team_stats.get('trueShooting'), team_stats.get('rating'),
                    team_stats.get('gameScore'),
                    team_points.get('total'), json.dumps(team_points.get('byPeriod', [])),
                    team_points.get('largestLead'), team_points.get('fastBreak'),
                    team_points.get('inPaint'), team_points.get('offTurnovers'),
                    team_2pt.get('made'), team_2pt.get('attempted'), team_2pt.get('pct'),
                    team_3pt.get('made'), team_3pt.get('attempted'), team_3pt.get('pct'),
                    team_ft.get('made'), team_ft.get('attempted'), team_ft.get('pct'),
                    team_fg.get('made'), team_fg.get('attempted'), team_fg.get('pct'),
                    team_to.get('total'), team_to.get('teamTotal'),
                    team_reb.get('offensive'), team_reb.get('defensive'), team_reb.get('total'),
                    team_fouls.get('total'), team_fouls.get('technical'), team_fouls.get('flagrant'),
                    team_ff.get('effectiveFieldGoalPct'), team_ff.get('freeThrowRate'),
                    team_ff.get('turnoverRatio'), team_ff.get('offensiveReboundPct'),
                    # Opponent stats
                    opp_stats.get('possessions'), opp_stats.get('assists'), opp_stats.get('steals'),
                    opp_stats.get('blocks'), opp_stats.get('trueShooting'), opp_stats.get('rating'),
                    opp_stats.get('gameScore'),
                    opp_points.get('total'), json.dumps(opp_points.get('byPeriod', [])),
                    opp_points.get('largestLead'), opp_points.get('fastBreak'),
                    opp_points.get('inPaint'), opp_points.get('offTurnovers'),
                    opp_2pt.get('made'), opp_2pt.get('attempted'), opp_2pt.get('pct'),
                    opp_3pt.get('made'), opp_3pt.get('attempted'), opp_3pt.get('pct'),
                    opp_ft.get('made'), opp_ft.get('attempted'), opp_ft.get('pct'),
                    opp_fg.get('made'), opp_fg.get('attempted'), opp_fg.get('pct'),
                    opp_to.get('total'), opp_to.get('teamTotal'),
                    opp_reb.get('offensive'), opp_reb.get('defensive'), opp_reb.get('total'),
                    opp_fouls.get('total'), opp_fouls.get('technical'), opp_fouls.get('flagrant'),
                    opp_ff.get('effectiveFieldGoalPct'), opp_ff.get('freeThrowRate'),
                    opp_ff.get('turnoverRatio'), opp_ff.get('offensiveReboundPct')
                )
                
                cursor.execute(insert_sql, params)
                inserted += 1
                
            except Exception as e:
                print(f"Error inserting team game {tg.get('gameId')}-{tg.get('teamId')}: {e}")
                continue
        
        return inserted
    
    def run_etl(self, start_season: int = 2006, end_season: int = 2025, 
                fetch_games: bool = True, fetch_team_games: bool = True):
        """
        Run the complete ETL process
        
        Args:
            start_season: Starting season year (default 2006 for 2005-06)
            end_season: Ending season year (default 2025 for 2024-25)
            fetch_games: Whether to fetch games endpoint (default True)
            fetch_team_games: Whether to fetch team_games endpoint (default True)
        """
        try:
            # Connect to database
            conn = pyodbc.connect(self.conn_string)
            cursor = conn.cursor()
            
            # Create tables if needed
            self.create_table_if_not_exists(cursor)
            conn.commit()
            
            total_games_inserted = 0
            total_team_games_inserted = 0
            
            # Process each season
            for season in range(start_season, end_season + 1):
                print(f"\n{'='*60}")
                print(f"Processing season {season}...")
                print(f"{'='*60}")
                
                # Fetch and insert games
                if fetch_games:
                    print(f"\n--- Games Endpoint ---")
                    games = self.fetch_all_games_for_season(season, endpoint="games")
                    
                    if games:
                        inserted = self.insert_games(games, cursor)
                        conn.commit()
                        total_games_inserted += inserted
                        print(f"✓ Inserted {inserted} games for season {season}")
                
                # Fetch and insert team games
                if fetch_team_games:
                    print(f"\n--- Team Games Endpoint ---")
                    team_games = self.fetch_all_games_for_season(season, endpoint="team_games")
                    
                    if team_games:
                        inserted = self.insert_team_games(team_games, cursor)
                        conn.commit()
                        total_team_games_inserted += inserted
                        print(f"✓ Inserted {inserted} team game records for season {season}")
                
                # Be nice to the API between seasons
                time.sleep(2)
            
            print(f"\n{'='*60}")
            print(f"=== ETL Complete ===")
            print(f"{'='*60}")
            if fetch_games:
                print(f"Total games inserted/updated: {total_games_inserted}")
            if fetch_team_games:
                print(f"Total team game records inserted/updated: {total_team_games_inserted}")
            
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
    
    # MSSQL connection string - adjust as needed
    DB_CONNECTION = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=your_server_name;"
        "DATABASE=your_database_name;"
        "UID=your_username;"
        "PWD=your_password"
    )
    
    # Alternative for Windows Authentication:
    # DB_CONNECTION = (
    #     "DRIVER={ODBC Driver 17 for SQL Server};"
    #     "SERVER=your_server_name;"
    #     "DATABASE=your_database_name;"
    #     "Trusted_Connection=yes;"
    # )
    
    # Initialize and run ETL
    etl = CollegeBasketballETL(API_KEY, DB_CONNECTION)
    
    # Fetch both games and team_games (default)
    etl.run_etl(start_season=2006, end_season=2025)
    
    # Or fetch only one endpoint:
    # etl.run_etl(start_season=2006, end_season=2025, fetch_games=True, fetch_team_games=False)
    # etl.run_etl(start_season=2006, end_season=2025, fetch_games=False, fetch_team_games=True)