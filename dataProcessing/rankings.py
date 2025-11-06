"""
Adds team ranking features.
"""
import pandas as pd
import numpy as np

class RankingsProcessor:
    def __init__(self, db_conn):
        self.db = db_conn
        
    def add_features(self, df):
        print("Adding ranking features...")
        
        rankings = self.db.query("""
            SELECT r.season, r.teamId, r.pollDate, r.pollType, r.ranking, r.points as poll_points
            FROM rankings r
            WHERE r.pollType IN ('AP', 'Coaches')
            ORDER BY r.pollDate DESC
        """)
        
        for poll_type in ['AP', 'Coaches']:
            poll_rankings = rankings[rankings['pollType'] == poll_type]
            
            for team_type in ['home', 'away']:
                self._add_poll_rankings(df, poll_rankings, poll_type, team_type)
        
        print("Added ranking features")
        return df
    
    def _add_poll_rankings(self, df, poll_rankings, poll_type, team_type):
        rankings = []
        team_id_col = f'{team_type}TeamId'
        
        for idx, game in df.iterrows():
            game_date = game['startDate']
            team_id = game[team_id_col]
            
            rank_data = poll_rankings[
                (poll_rankings['teamId'] == team_id) &
                (poll_rankings['pollDate'] < game_date)
            ]
            
            if len(rank_data) > 0:
                rankings.append(rank_data.iloc[0]['ranking'])
            else:
                rankings.append(np.nan)
        
        df[f'{team_type}_{poll_type}_rank'] = rankings
        df[f'{team_type}_{poll_type}_ranked'] = ~pd.isna(rankings)