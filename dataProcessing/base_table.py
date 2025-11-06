"""
Creates the base table with games and targets.
"""
import pandas as pd

class BaseTableBuilder:
    def __init__(self, db_conn):
        self.db = db_conn
        
    def create(self, season_start, season_end):
        print("Creating base table with targets...")
        
        query = f"""
        SELECT 
            g.id as gameId,
            g.sourceId as gameSourceId,
            g.season,
            g.seasonType,
            g.tournament,
            g.startDate,
            g.neutralSite,
            g.conferenceGame,
            g.gameType,
            g.homeTeamId,
            g.homeTeam,
            g.homeConferenceId,
            g.homeConference,
            g.awayTeamId,
            g.awayTeam,
            g.awayConferenceId,
            g.awayConference,
            g.homePoints,
            g.awayPoints,
            g.venueId,
            g.venue,
            g.city,
            g.state,
            (g.homePoints - g.awayPoints) as spread_actual,
            (g.homePoints + g.awayPoints) as total_actual,
            CASE WHEN g.homePoints > g.awayPoints THEN 1 ELSE 0 END as home_won
        FROM games g
        WHERE g.season >= {season_start} 
            AND g.season <= {season_end}
            AND g.status = 'Final'
            AND g.homePoints IS NOT NULL
            AND g.awayPoints IS NOT NULL
        ORDER BY g.startDate
        """
        
        df = self.db.query(query)
        print(f"Base table created with {len(df)} games")
        return df