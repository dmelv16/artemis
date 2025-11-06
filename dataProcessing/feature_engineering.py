"""
Additional feature engineering utilities.
"""
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, db_conn):
        self.db = db_conn
        
    def create_differentials(self, df):
        print("Creating differential features...")
        
        stat_cols = [col for col in df.columns if col.startswith('home_') and 
                    any(stat in col for stat in ['pace', 'rating', 'Pct', 'Ratio', 'Rate', 
                                                  'assists', 'steals', 'blocks', 'points'])]
        
        for home_col in stat_cols:
            away_col = home_col.replace('home_', 'away_')
            if away_col in df.columns:
                diff_col = home_col.replace('home_', 'diff_')
                df[diff_col] = df[home_col] - df[away_col]
        
        print(f"Created {len([c for c in df.columns if c.startswith('diff_')])} differential features")
        return df
    
    def add_temporal_features(self, df):
        print("Adding temporal features...")
        
        df['startDate'] = pd.to_datetime(df['startDate'])
        
        # Basic temporal features
        df['day_of_week'] = df['startDate'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['startDate'].dt.month
        
        # Days since season start
        season_starts = df.groupby('season')['startDate'].min()
        df['days_since_season_start'] = df.apply(
            lambda x: (x['startDate'] - season_starts[x['season']]).days, axis=1
        )
        
        # Calculate rest days
        df = self._calculate_rest_days(df)
        
        print("Added temporal features")
        return df
    
    def _calculate_rest_days(self, df):
        team_games = self.db.query("""
            SELECT teamId, gameId, startDate
            FROM team_games tg
            JOIN games g ON tg.gameId = g.id
            ORDER BY teamId, startDate
        """)
        
        rest_days_home = []
        rest_days_away = []
        
        for idx, game in df.iterrows():
            # Home team
            home_prev = team_games[
                (team_games['teamId'] == game['homeTeamId']) &
                (team_games['startDate'] < game['startDate'])
            ]
            
            if len(home_prev) > 0:
                rest_days = (game['startDate'] - pd.to_datetime(home_prev.iloc[-1]['startDate'])).days
                rest_days_home.append(rest_days)
            else:
                rest_days_home.append(7)
            
            # Away team
            away_prev = team_games[
                (team_games['teamId'] == game['awayTeamId']) &
                (team_games['startDate'] < game['startDate'])
            ]
            
            if len(away_prev) > 0:
                rest_days = (game['startDate'] - pd.to_datetime(away_prev.iloc[-1]['startDate'])).days
                rest_days_away.append(rest_days)
            else:
                rest_days_away.append(7)
        
        df['home_rest_days'] = rest_days_home
        df['away_rest_days'] = rest_days_away
        df['rest_days_diff'] = df['home_rest_days'] - df['away_rest_days']
        
        return df