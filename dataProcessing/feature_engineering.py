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
        
        # Create all differential columns at once
        diff_dict = {}
        for home_col in stat_cols:
            away_col = home_col.replace('home_', 'away_')
            if away_col in df.columns:
                diff_col = home_col.replace('home_', 'diff_')
                diff_dict[diff_col] = df[home_col] - df[away_col]
        
        # Concatenate all new columns at once
        if diff_dict:
            diff_df = pd.DataFrame(diff_dict, index=df.index)
            df = pd.concat([df, diff_df], axis=1)
        
        print(f"Created {len(diff_dict)} differential features")
        return df

    def add_temporal_features(self, df):
        print("Adding temporal features...")
        
        df['startDate'] = pd.to_datetime(df['startDate'])
        
        # Calculate all temporal features at once
        temporal_dict = {
            'day_of_week': df['startDate'].dt.dayofweek,
            'is_weekend': df['startDate'].dt.dayofweek.isin([5, 6]).astype(int),
            'month': df['startDate'].dt.month
        }
        
        # Days since season start
        season_starts = df.groupby('season')['startDate'].min()
        temporal_dict['days_since_season_start'] = df.apply(
            lambda x: (x['startDate'] - season_starts[x['season']]).days, axis=1
        )
        
        # Create DataFrame and concat once
        temporal_df = pd.DataFrame(temporal_dict, index=df.index)
        df = pd.concat([df, temporal_df], axis=1)
        
        # Calculate rest days (optimized)
        df = self._calculate_rest_days_optimized(df)
        
        print("Added temporal features")
        return df
    
    def _calculate_rest_days_optimized(self, df):
        """
        Optimized rest days calculation using vectorized operations.
        Much faster than iterating through rows.
        """
        print("Calculating rest days (optimized)...")
        
        # 1. Get all game dates for all teams
        team_games = self.db.query("""
            SELECT teamId, gameId, g.startDate
            FROM team_games tg
            JOIN games g ON tg.gameId = g.id
            ORDER BY teamId, startDate
        """)
        
        team_games['startDate'] = pd.to_datetime(team_games['startDate'])
        
        # 2. Use groupby().shift() to get each team's previous game date
        team_games['prev_game_date'] = team_games.groupby('teamId')['startDate'].shift(1)
        
        # 3. Calculate rest days
        team_games['rest_days'] = (team_games['startDate'] - team_games['prev_game_date']).dt.days
        
        # 4. Fill NaN (first games) with default value
        team_games['rest_days'] = team_games['rest_days'].fillna(7)
        
        # 5. Merge back into main dataframe
        # For home team
        df_home_rest = team_games[['gameId', 'teamId', 'rest_days']].rename(
            columns={'rest_days': 'home_rest_days', 'teamId': 'homeTeamId'}
        )
        df = pd.merge(
            df,
            df_home_rest,
            on=['gameId', 'homeTeamId'],
            how='left'
        )
        
        # For away team
        df_away_rest = team_games[['gameId', 'teamId', 'rest_days']].rename(
            columns={'rest_days': 'away_rest_days', 'teamId': 'awayTeamId'}
        )
        df = pd.merge(
            df,
            df_away_rest,
            on=['gameId', 'awayTeamId'],
            how='left'
        )
        
        # Handle any remaining NaNs
        df['home_rest_days'] = df['home_rest_days'].fillna(7)
        df['away_rest_days'] = df['away_rest_days'].fillna(7)
        
        # Calculate differential
        df['rest_days_diff'] = df['home_rest_days'] - df['away_rest_days']
        
        return df
    
    def add_game_context_features(self, df):
        """
        Add features from the games table itself.
        """
        print("Adding game context features...")
        
        # Neutral site indicator
        if 'neutralSite' in df.columns:
            df['is_neutral_site'] = (df['neutralSite'] == True).astype(int)
        
        # Conference game indicator
        if 'conferenceGame' in df.columns:
            df['is_conference_game'] = (df['conferenceGame'] == True).astype(int)
        
        # Back-to-back game indicators
        if 'home_rest_days' in df.columns and 'away_rest_days' in df.columns:
            df['home_is_b2b'] = (df['home_rest_days'] == 1).astype(int)
            df['away_is_b2b'] = (df['away_rest_days'] == 1).astype(int)
            df['b2b_diff'] = df['home_is_b2b'] - df['away_is_b2b']
        
        print("Added game context features")
        return df
    
    def add_streak_features(self, df):
        """
        Add win/loss streak features for momentum.
        This requires game outcomes sorted by date.
        """
        print("Adding streak features...")
        
        # Query game results for all teams
        results = self.db.query("""
            SELECT 
                tg.teamId,
                g.id as gameId,
                g.startDate,
                CASE 
                    WHEN tg.teamId = g.homeTeamId THEN 
                        IIF(g.homePoints > g.awayPoints, 1, 0)
                    ELSE 
                        IIF(g.awayPoints > g.homePoints, 1, 0)
                END as won
            FROM team_games tg
            JOIN games g ON tg.gameId = g.id
            WHERE g.homePoints IS NOT NULL AND g.awayPoints IS NOT NULL
            ORDER BY tg.teamId, g.startDate
        """)
        
        results['startDate'] = pd.to_datetime(results['startDate'])
        
        # Calculate streaks
        def calculate_streak(won_series):
            """Calculate current winning/losing streak for each game."""
            streaks = []
            current_streak = 0
            
            for won in won_series:
                if pd.isna(won):
                    streaks.append(0)
                    current_streak = 0
                elif won:
                    current_streak = current_streak + 1 if current_streak > 0 else 1
                    streaks.append(current_streak)
                else:
                    current_streak = current_streak - 1 if current_streak < 0 else -1
                    streaks.append(current_streak)
            
            return pd.Series(streaks, index=won_series.index)
        
        results['streak'] = results.groupby('teamId')['won'].transform(calculate_streak)
        
        # Shift to get streak BEFORE this game
        results['streak_before'] = results.groupby('teamId')['streak'].shift(1).fillna(0)
        
        # Merge for home team
        df_home_streak = results[['gameId', 'teamId', 'streak_before']].rename(
            columns={'streak_before': 'home_streak', 'teamId': 'homeTeamId'}
        )
        df = pd.merge(df, df_home_streak, on=['gameId', 'homeTeamId'], how='left')
        
        # Merge for away team
        df_away_streak = results[['gameId', 'teamId', 'streak_before']].rename(
            columns={'streak_before': 'away_streak', 'teamId': 'awayTeamId'}
        )
        df = pd.merge(df, df_away_streak, on=['gameId', 'awayTeamId'], how='left')
        
        # Fill NaN and create differential
        df['home_streak'] = df['home_streak'].fillna(0)
        df['away_streak'] = df['away_streak'].fillna(0)
        df['streak_diff'] = df['home_streak'] - df['away_streak']
        
        print("Added streak features")
        return df
    
    def engineer_all_features(self, df):
        """
        Apply all feature engineering steps in sequence.
        """
        df = self.create_differentials(df)
        df = self.add_temporal_features(df)
        df = self.add_game_context_features(df)
        df = self.add_streak_features(df)
        
        return df