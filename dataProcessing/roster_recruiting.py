"""
Adds roster and recruiting features.
"""
import pandas as pd

class RosterRecruitingProcessor:
    def __init__(self, db_conn):
        self.db = db_conn
        
    def add_features(self, df):
        print("Adding roster and recruiting features...")
        
        # Get roster stats
        roster_df = self.db.query("""
            SELECT 
                r.teamId,
                r.season,
                COUNT(DISTINCT r.playerId) as roster_size,
                AVG(CAST(r.height as float)) as avg_height,
                AVG(CAST(r.weight as float)) as avg_weight,
                COUNT(CASE WHEN r.position IN ('G', 'PG', 'SG') THEN 1 END) as num_guards,
                COUNT(CASE WHEN r.position IN ('F', 'SF', 'PF') THEN 1 END) as num_forwards,
                COUNT(CASE WHEN r.position IN ('C') THEN 1 END) as num_centers
            FROM rosters r
            GROUP BY r.teamId, r.season
        """)
        
        # Get recruiting stats
        recruiting_df = self.db.query("""
            SELECT 
                rc.committedToId as teamId,
                rc.year as season,
                AVG(CAST(rc.stars as float)) as avg_recruit_stars,
                AVG(rc.rating) as avg_recruit_rating,
                MAX(rc.stars) as max_recruit_stars,
                COUNT(*) as num_recruits
            FROM recruiting rc
            WHERE rc.committedToId IS NOT NULL
            GROUP BY rc.committedToId, rc.year
        """)
        
        # Merge for home and away teams
        for team_type in ['home', 'away']:
            df = self._merge_team_features(df, roster_df, recruiting_df, team_type)
        
        print("Added roster and recruiting features")
        return df
    
    def _merge_team_features(self, df, roster_df, recruiting_df, team_type):
        # Merge roster
        df = df.merge(
            roster_df,
            left_on=[f'{team_type}TeamId', 'season'],
            right_on=['teamId', 'season'],
            how='left',
            suffixes=('', f'_{team_type}_roster')
        )
        
        # Rename columns
        roster_cols = ['roster_size', 'avg_height', 'avg_weight', 'num_guards', 'num_forwards', 'num_centers']
        for col in roster_cols:
            if col in df.columns:
                df.rename(columns={col: f'{team_type}_{col}'}, inplace=True)
        
        # Merge recruiting
        df = df.merge(
            recruiting_df,
            left_on=[f'{team_type}TeamId', 'season'],
            right_on=['teamId', 'season'],
            how='left',
            suffixes=('', f'_{team_type}_recruit')
        )
        
        recruiting_cols = ['avg_recruit_stars', 'avg_recruit_rating', 'max_recruit_stars', 'num_recruits']
        for col in recruiting_cols:
            if col in df.columns:
                df.rename(columns={col: f'{team_type}_{col}'}, inplace=True)
        
        # Clean duplicates
        df = df.loc[:, ~df.columns.duplicated()]
        return df