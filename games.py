from base_etl import BaseCollegeBasketballETL
from typing import List, Dict
from datetime import datetime, timedelta


class GamesETL(BaseCollegeBasketballETL):
    """ETL for college basketball games and team games data"""

    def get_existing_game_ids(self, cursor, season: int) -> set:
        """Get set of game IDs that already exist in database for a season"""
        cursor.execute("SELECT id FROM games WHERE season = ?", (season,))
        return {row[0] for row in cursor.fetchall()}

    def get_existing_team_game_keys(self, cursor, season: int) -> set:
        """Get set of (gameId, teamId) tuples that already exist for a season"""
        cursor.execute("SELECT gameId, teamId FROM team_games WHERE season = ?", (season,))
        return {(row[0], row[1]) for row in cursor.fetchall()}
    
    def create_tables(self, cursor):
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
            state NVARCHAR(50)
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
            points INT,
            pointsByPeriod NVARCHAR(MAX),
            largestLead INT,
            fastBreakPoints INT,
            pointsInPaint INT,
            pointsOffTurnovers INT,
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
            turnovers INT,
            teamTurnovers INT,
            offensiveRebounds INT,
            defensiveRebounds INT,
            totalRebounds INT,
            fouls INT,
            technicalFouls INT,
            flagrantFouls INT,
            effectiveFieldGoalPct FLOAT,
            freeThrowRate FLOAT,
            turnoverRatio FLOAT,
            offensiveReboundPct FLOAT,
            
            -- Opponent Stats (same structure)
            oppPossessions FLOAT,
            oppAssists INT,
            oppSteals INT,
            oppBlocks INT,
            oppTrueShooting FLOAT,
            oppRating FLOAT,
            oppGameScore FLOAT,
            oppPoints INT,
            oppPointsByPeriod NVARCHAR(MAX),
            oppLargestLead INT,
            oppFastBreakPoints INT,
            oppPointsInPaint INT,
            oppPointsOffTurnovers INT,
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
            oppTurnovers INT,
            oppTeamTurnovers INT,
            oppOffensiveRebounds INT,
            oppDefensiveRebounds INT,
            oppTotalRebounds INT,
            oppFouls INT,
            oppTechnicalFouls INT,
            oppFlagrantFouls INT,
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
    
    def fetch_games_for_season(self, season: int, endpoint: str = "games") -> List[Dict]:
        """Fetch all games for a season with pagination"""
        all_games = []
        batch_limit = 1000 if endpoint == "player_games" else 3000
        date_ranges = self.get_season_date_ranges(season)
        
        print(f"Fetching {endpoint} for season {season}...")
        
        for start_date, end_date in date_ranges:
            params = {
                "season": season,
                "startDateRange": start_date,
                "endDateRange": end_date
            }
            
            url_map = {
                "games": "/games",
                "team_games": "/games/teams",
                "player_games": "/games/players"
            }
            
            games = self.fetch_api(url_map[endpoint], params)
            
            if games:
                print(f"  {start_date[:7]}: {len(games)} records")
                
                if len(games) >= batch_limit:
                    print(f"  Hit {batch_limit} limit, fetching by weeks...")
                    games = self._fetch_by_week(season, start_date, end_date, endpoint)
                
                all_games.extend(games)
            
            self.rate_limit_sleep(0.5)
        
        # Remove duplicates
        id_field = 'gameId' if endpoint != "games" else 'id'
        if endpoint == "player_games":
            unique = {f"{g.get('gameId')}_{g.get('athleteId')}": g for g in all_games}
        else:
            unique = {g.get(id_field): g for g in all_games if id_field in g}
        
        print(f"Total unique records: {len(unique)}")
        return list(unique.values())
    
    def _fetch_by_week(self, season: int, month_start: str, month_end: str, endpoint: str) -> List[Dict]:
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
            
            url_map = {
                "games": "/games",
                "team_games": "/games/teams",
                "player_games": "/games/players"
            }
            
            games = self.fetch_api(url_map[endpoint], params)
            if games:
                all_games.extend(games)
            
            current = week_end
            self.rate_limit_sleep(0.5)
        
        return all_games
    
    def insert_games(self, games: List[Dict], cursor) -> int:
        """Insert games into database"""
        sql = """
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
            params = (
                game.get('id'),  # MERGE condition
                game.get('id'), game.get('sourceId'), game.get('seasonLabel'),
                game.get('season'), game.get('seasonType'), game.get('tournament'),
                self.parse_datetime(game.get('startDate')),
                game.get('startTimeTbd', False), game.get('neutralSite', False),
                game.get('conferenceGame', False), game.get('gameType'),
                game.get('status'), game.get('gameNotes'), game.get('attendance'),
                game.get('homeTeamId'), game.get('homeTeam'),
                game.get('homeConferenceId'), game.get('homeConference'),
                game.get('homeSeed'), game.get('homePoints'),
                self.json_serialize(game.get('homePeriodPoints')),
                game.get('homeWinner', False),
                game.get('awayTeamId'), game.get('awayTeam'),
                game.get('awayConferenceId'), game.get('awayConference'),
                game.get('awaySeed'), game.get('awayPoints'),
                self.json_serialize(game.get('awayPeriodPoints')),
                game.get('awayWinner', False), game.get('excitement'),
                game.get('venueId'), game.get('venue'),
                game.get('city'), game.get('state')
            )
            
            if self.execute_merge(cursor, sql, params, f"game {game.get('id')}"):
                inserted += 1
        
        return inserted
    
    def insert_team_games(self, team_games: List[Dict], cursor) -> int:
        """Insert team games into database - 96 columns, 96 ? marks"""
        sql = """
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
                    ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for tg in team_games:
            ts = tg.get('teamStats') or {}
            os = tg.get('opponentStats') or {}
            
            params = (
                # MERGE (2)
                tg.get('gameId'), tg.get('teamId'),
                
                # INSERT (96) = 22 + 37 + 37
                tg.get('gameId'), tg.get('season'), tg.get('seasonLabel'),
                tg.get('seasonType'), tg.get('tournament'),
                self.parse_datetime(tg.get('startDate')), tg.get('startTimeTbd', False),
                tg.get('teamId'), tg.get('team'), tg.get('conference'), tg.get('teamSeed'),
                tg.get('opponentId'), tg.get('opponent'), tg.get('opponentConference'),
                tg.get('opponentSeed'), tg.get('neutralSite', False), tg.get('isHome', False),
                tg.get('conferenceGame', False), tg.get('gameType'), tg.get('notes'),
                tg.get('gameMinutes'), tg.get('pace'),
                # Team stats (37)
                ts.get('possessions'), ts.get('assists'), ts.get('steals'), ts.get('blocks'),
                ts.get('trueShooting'), ts.get('rating'), ts.get('gameScore'),
                ts.get('points', {}).get('total') if ts.get('points') else None,
                self.json_serialize(ts.get('points', {}).get('byPeriod')) if ts.get('points') else None,
                ts.get('points', {}).get('largestLead') if ts.get('points') else None,
                ts.get('points', {}).get('fastBreak') if ts.get('points') else None,
                ts.get('points', {}).get('inPaint') if ts.get('points') else None,
                ts.get('points', {}).get('offTurnovers') if ts.get('points') else None,
                ts.get('twoPointFieldGoals', {}).get('made') if ts.get('twoPointFieldGoals') else None,
                ts.get('twoPointFieldGoals', {}).get('attempted') if ts.get('twoPointFieldGoals') else None,
                ts.get('twoPointFieldGoals', {}).get('pct') if ts.get('twoPointFieldGoals') else None,
                ts.get('threePointFieldGoals', {}).get('made') if ts.get('threePointFieldGoals') else None,
                ts.get('threePointFieldGoals', {}).get('attempted') if ts.get('threePointFieldGoals') else None,
                ts.get('threePointFieldGoals', {}).get('pct') if ts.get('threePointFieldGoals') else None,
                ts.get('freeThrows', {}).get('made') if ts.get('freeThrows') else None,
                ts.get('freeThrows', {}).get('attempted') if ts.get('freeThrows') else None,
                ts.get('freeThrows', {}).get('pct') if ts.get('freeThrows') else None,
                ts.get('fieldGoals', {}).get('made') if ts.get('fieldGoals') else None,
                ts.get('fieldGoals', {}).get('attempted') if ts.get('fieldGoals') else None,
                ts.get('fieldGoals', {}).get('pct') if ts.get('fieldGoals') else None,
                ts.get('turnovers', {}).get('total') if ts.get('turnovers') else None,
                ts.get('turnovers', {}).get('teamTotal') if ts.get('turnovers') else None,
                ts.get('rebounds', {}).get('offensive') if ts.get('rebounds') else None,
                ts.get('rebounds', {}).get('defensive') if ts.get('rebounds') else None,
                ts.get('rebounds', {}).get('total') if ts.get('rebounds') else None,
                ts.get('fouls', {}).get('total') if ts.get('fouls') else None,
                ts.get('fouls', {}).get('technical') if ts.get('fouls') else None,
                ts.get('fouls', {}).get('flagrant') if ts.get('fouls') else None,
                ts.get('fourFactors', {}).get('effectiveFieldGoalPct') if ts.get('fourFactors') else None,
                ts.get('fourFactors', {}).get('freeThrowRate') if ts.get('fourFactors') else None,
                ts.get('fourFactors', {}).get('turnoverRatio') if ts.get('fourFactors') else None,
                ts.get('fourFactors', {}).get('offensiveReboundPct') if ts.get('fourFactors') else None,
                # Opponent stats (37)
                os.get('possessions'), os.get('assists'), os.get('steals'), os.get('blocks'),
                os.get('trueShooting'), os.get('rating'), os.get('gameScore'),
                os.get('points', {}).get('total') if os.get('points') else None,
                self.json_serialize(os.get('points', {}).get('byPeriod')) if os.get('points') else None,
                os.get('points', {}).get('largestLead') if os.get('points') else None,
                os.get('points', {}).get('fastBreak') if os.get('points') else None,
                os.get('points', {}).get('inPaint') if os.get('points') else None,
                os.get('points', {}).get('offTurnovers') if os.get('points') else None,
                os.get('twoPointFieldGoals', {}).get('made') if os.get('twoPointFieldGoals') else None,
                os.get('twoPointFieldGoals', {}).get('attempted') if os.get('twoPointFieldGoals') else None,
                os.get('twoPointFieldGoals', {}).get('pct') if os.get('twoPointFieldGoals') else None,
                os.get('threePointFieldGoals', {}).get('made') if os.get('threePointFieldGoals') else None,
                os.get('threePointFieldGoals', {}).get('attempted') if os.get('threePointFieldGoals') else None,
                os.get('threePointFieldGoals', {}).get('pct') if os.get('threePointFieldGoals') else None,
                os.get('freeThrows', {}).get('made') if os.get('freeThrows') else None,
                os.get('freeThrows', {}).get('attempted') if os.get('freeThrows') else None,
                os.get('freeThrows', {}).get('pct') if os.get('freeThrows') else None,
                os.get('fieldGoals', {}).get('made') if os.get('fieldGoals') else None,
                os.get('fieldGoals', {}).get('attempted') if os.get('fieldGoals') else None,
                os.get('fieldGoals', {}).get('pct') if os.get('fieldGoals') else None,
                os.get('turnovers', {}).get('total') if os.get('turnovers') else None,
                os.get('turnovers', {}).get('teamTotal') if os.get('turnovers') else None,
                os.get('rebounds', {}).get('offensive') if os.get('rebounds') else None,
                os.get('rebounds', {}).get('defensive') if os.get('rebounds') else None,
                os.get('rebounds', {}).get('total') if os.get('rebounds') else None,
                os.get('fouls', {}).get('total') if os.get('fouls') else None,
                os.get('fouls', {}).get('technical') if os.get('fouls') else None,
                os.get('fouls', {}).get('flagrant') if os.get('fouls') else None,
                os.get('fourFactors', {}).get('effectiveFieldGoalPct') if os.get('fourFactors') else None,
                os.get('fourFactors', {}).get('freeThrowRate') if os.get('fourFactors') else None,
                os.get('fourFactors', {}).get('turnoverRatio') if os.get('fourFactors') else None,
                os.get('fourFactors', {}).get('offensiveReboundPct') if os.get('fourFactors') else None
            )
            
            if self.execute_merge(cursor, sql, params, f"team game {tg.get('gameId')}-{tg.get('teamId')}"):
                inserted += 1
        
        return inserted
    
    def run_etl(self, start_season: int = 2006, end_season: int = 2025,
            fetch_games: bool = True, fetch_team_games: bool = True,
            skip_existing: bool = True):  # Add this parameter
        """Run ETL for games and team games"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            self.create_tables(cursor)
            conn.commit()
            
            total_games = 0
            total_team_games = 0
            
            for season in range(start_season, end_season + 1):
                print(f"\n{'='*60}\nSeason {season}\n{'='*60}")
                
                if fetch_games:
                    # Get existing games if skip_existing is True
                    if skip_existing:
                        existing_ids = self.get_existing_game_ids(cursor, season)
                        print(f"Found {len(existing_ids)} existing games for season {season}")
                    else:
                        existing_ids = set()
                    
                    games = self.fetch_games_for_season(season, "games")
                    if games:
                        # Filter out existing games
                        if skip_existing:
                            games = [g for g in games if g.get('id') not in existing_ids]
                            print(f"Processing {len(games)} new games")
                        
                        if games:  # Only insert if there are new games
                            inserted = self.insert_games(games, cursor)
                            conn.commit()
                            total_games += inserted
                            print(f"✓ {inserted} games inserted")
                        else:
                            print("No new games to insert")
                
                if fetch_team_games:
                    # Get existing team games if skip_existing is True
                    if skip_existing:
                        existing_keys = self.get_existing_team_game_keys(cursor, season)
                        print(f"Found {len(existing_keys)} existing team games for season {season}")
                    else:
                        existing_keys = set()
                    
                    team_games = self.fetch_games_for_season(season, "team_games")
                    if team_games:
                        # Filter out existing team games
                        if skip_existing:
                            team_games = [tg for tg in team_games 
                                        if (tg.get('gameId'), tg.get('teamId')) not in existing_keys]
                            print(f"Processing {len(team_games)} new team games")
                        
                        if team_games:  # Only insert if there are new team games
                            inserted = self.insert_team_games(team_games, cursor)
                            conn.commit()
                            total_team_games += inserted
                            print(f"✓ {inserted} team games inserted")
                        else:
                            print("No new team games to insert")
                
                self.rate_limit_sleep(2)
            
            print(f"\n{'='*60}\nComplete: {total_games} games, {total_team_games} team games\n{'='*60}")
            
        except Exception as e:
            print(f"Error: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()


if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    DB_CONNECTION = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=your_server_name;"
        "DATABASE=your_database_name;"
        "UID=your_username;"
        "PWD=your_password"
    )
    
    etl = GamesETL(API_KEY, DB_CONNECTION)
    etl.run_etl(start_season=2006, end_season=2025)