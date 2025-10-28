from base_etl import BaseCollegeBasketballETL
from typing import List, Dict
from datetime import datetime, timedelta


class GamesETL(BaseCollegeBasketballETL):
    """ETL for college basketball games and team games data"""
    
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
        """Insert team games into database"""
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
                    ?, ?, ?, ?, ?, ?);
        """
        
        inserted = 0
        for tg in team_games:
            ts = tg.get('teamStats', {})
            os = tg.get('opponentStats', {})
            
            params = (
                tg.get('gameId'), tg.get('teamId'),  # MERGE condition
                tg.get('gameId'), tg.get('season'), tg.get('seasonLabel'),
                tg.get('seasonType'), tg.get('tournament'),
                self.parse_datetime(tg.get('startDate')), tg.get('startTimeTbd', False),
                tg.get('teamId'), tg.get('team'), tg.get('conference'), tg.get('teamSeed'),
                tg.get('opponentId'), tg.get('opponent'), tg.get('opponentConference'),
                tg.get('opponentSeed'), tg.get('neutralSite', False), tg.get('isHome', False),
                tg.get('conferenceGame', False), tg.get('gameType'), tg.get('notes'),
                tg.get('gameMinutes'), tg.get('pace'),
                # Team stats
                ts.get('possessions'), ts.get('assists'), ts.get('steals'), ts.get('blocks'),
                ts.get('trueShooting'), ts.get('rating'), ts.get('gameScore'),
                ts.get('points', {}).get('total'),
                self.json_serialize(ts.get('points', {}).get('byPeriod')),
                ts.get('points', {}).get('largestLead'),
                ts.get('points', {}).get('fastBreak'),
                ts.get('points', {}).get('inPaint'),
                ts.get('points', {}).get('offTurnovers'),
                ts.get('twoPointFieldGoals', {}).get('made'),
                ts.get('twoPointFieldGoals', {}).get('attempted'),
                ts.get('twoPointFieldGoals', {}).get('pct'),
                ts.get('threePointFieldGoals', {}).get('made'),
                ts.get('threePointFieldGoals', {}).get('attempted'),
                ts.get('threePointFieldGoals', {}).get('pct'),
                ts.get('freeThrows', {}).get('made'),
                ts.get('freeThrows', {}).get('attempted'),
                ts.get('freeThrows', {}).get('pct'),
                ts.get('fieldGoals', {}).get('made'),
                ts.get('fieldGoals', {}).get('attempted'),
                ts.get('fieldGoals', {}).get('pct'),
                ts.get('turnovers', {}).get('total'),
                ts.get('turnovers', {}).get('teamTotal'),
                ts.get('rebounds', {}).get('offensive'),
                ts.get('rebounds', {}).get('defensive'),
                ts.get('rebounds', {}).get('total'),
                ts.get('fouls', {}).get('total'),
                ts.get('fouls', {}).get('technical'),
                ts.get('fouls', {}).get('flagrant'),
                ts.get('fourFactors', {}).get('effectiveFieldGoalPct'),
                ts.get('fourFactors', {}).get('freeThrowRate'),
                ts.get('fourFactors', {}).get('turnoverRatio'),
                ts.get('fourFactors', {}).get('offensiveReboundPct'),
                # Opponent stats (same structure)
                os.get('possessions'), os.get('assists'), os.get('steals'), os.get('blocks'),
                os.get('trueShooting'), os.get('rating'), os.get('gameScore'),
                os.get('points', {}).get('total'),
                self.json_serialize(os.get('points', {}).get('byPeriod')),
                os.get('points', {}).get('largestLead'),
                os.get('points', {}).get('fastBreak'),
                os.get('points', {}).get('inPaint'),
                os.get('points', {}).get('offTurnovers'),
                os.get('twoPointFieldGoals', {}).get('made'),
                os.get('twoPointFieldGoals', {}).get('attempted'),
                os.get('twoPointFieldGoals', {}).get('pct'),
                os.get('threePointFieldGoals', {}).get('made'),
                os.get('threePointFieldGoals', {}).get('attempted'),
                os.get('threePointFieldGoals', {}).get('pct'),
                os.get('freeThrows', {}).get('made'),
                os.get('freeThrows', {}).get('attempted'),
                os.get('freeThrows', {}).get('pct'),
                os.get('fieldGoals', {}).get('made'),
                os.get('fieldGoals', {}).get('attempted'),
                os.get('fieldGoals', {}).get('pct'),
                os.get('turnovers', {}).get('total'),
                os.get('turnovers', {}).get('teamTotal'),
                os.get('rebounds', {}).get('offensive'),
                os.get('rebounds', {}).get('defensive'),
                os.get('rebounds', {}).get('total'),
                os.get('fouls', {}).get('total'),
                os.get('fouls', {}).get('technical'),
                os.get('fouls', {}).get('flagrant'),
                os.get('fourFactors', {}).get('effectiveFieldGoalPct'),
                os.get('fourFactors', {}).get('freeThrowRate'),
                os.get('fourFactors', {}).get('turnoverRatio'),
                os.get('fourFactors', {}).get('offensiveReboundPct')
            )
            
            if self.execute_merge(cursor, sql, params, f"team game {tg.get('gameId')}-{tg.get('teamId')}"):
                inserted += 1
        
        return inserted
    
    def run_etl(self, start_season: int = 2006, end_season: int = 2025,
                fetch_games: bool = True, fetch_team_games: bool = True):
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
                    games = self.fetch_games_for_season(season, "games")
                    if games:
                        inserted = self.insert_games(games, cursor)
                        conn.commit()
                        total_games += inserted
                        print(f"✓ {inserted} games inserted")
                
                if fetch_team_games:
                    team_games = self.fetch_games_for_season(season, "team_games")
                    if team_games:
                        inserted = self.insert_team_games(team_games, cursor)
                        conn.commit()
                        total_team_games += inserted
                        print(f"✓ {inserted} team games inserted")
                
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