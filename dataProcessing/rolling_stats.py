"""
Calculates rolling team statistics.
"""
import pandas as pd
import numpy as np
from config import ROLLING_STAT_COLUMNS

class RollingStatsCalculator:
    def __init__(self, db_conn):
        self.db = db_conn
        self.stat_columns = ROLLING_STAT_COLUMNS
        
    def calculate(self, df, windows=[5, 10, 'season']):
        print("Calculating rolling team statistics...")
        
        # Get all team game stats
        team_games = self.db.query("""
            SELECT tg.*, g.startDate
            FROM team_games tg
            JOIN games g ON tg.gameId = g.id
            WHERE g.status = 'Final'
            ORDER BY g.startDate
        """)
        
        # Process each team type (home/away)
        for team_type in ['home', 'away']:
            print(f"  Processing {team_type} team statistics...")
            team_id_col = f'{team_type}TeamId'
            
            for team_id in df[team_id_col].unique():
                if pd.isna(team_id):
                    continue
                
                team_data = team_games[team_games['teamId'] == team_id].sort_values('startDate')
                if len(team_data) == 0:
                    continue
                
                # Calculate rolling stats for each window
                for window in windows:
                    self._calculate_window_stats(team_data, window)
                
                # Merge back to main dataframe
                self._merge_team_stats(df, team_data, team_id, team_id_col, team_type)
        
        print(f"Added {len([c for c in df.columns if '_std' in c or '_L' in c])} rolling stat features")
        return df
    
    def _calculate_window_stats(self, team_data, window):
        for stat in self.stat_columns:
            if stat not in team_data.columns:
                continue
            
            if window == 'season':
                team_data[f'{stat}_std'] = team_data[stat].expanding(min_periods=1).mean().shift(1)
            else:
                team_data[f'{stat}_L{window}'] = team_data[stat].rolling(
                    window=window, min_periods=1
                ).mean().shift(1)
    
    def _merge_team_stats(self, df, team_data, team_id, team_id_col, team_type):
        for idx, game in df[df[team_id_col] == team_id].iterrows():
            game_date = game['startDate']
            prior_games = team_data[team_data['startDate'] < game_date]
            
            if len(prior_games) > 0:
                latest_stats = prior_games.iloc[-1]
                
                for stat in self.stat_columns:
                    for suffix in ['_std', '_L5', '_L10']:
                        col_name = f'{stat}{suffix}'
                        if col_name in latest_stats:
                            feature_name = f'{team_type}_{col_name}'
                            df.at[idx, feature_name] = latest_stats[col_name]