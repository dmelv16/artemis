"""
Calculates rolling team statistics with fully vectorized operations.
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
        
        # 1. Pre-calculate all stats in a single vectorized operation
        all_stats_df = self._get_all_rolling_stats(windows)
        
        if all_stats_df.empty:
            print("Rolling stat calculation skipped.")
            return df

        # 2. Merge stats efficiently using vectorized operations
        df_with_stats = self._merge_stats_vectorized(df, all_stats_df, windows)
        
        # Count added features
        original_cols = set(df.columns)
        new_cols = set(df_with_stats.columns) - original_cols
        print(f"Added {len(new_cols)} rolling stat features")
        
        return df_with_stats
    
    def _get_all_rolling_stats(self, windows):
        """
        Queries and calculates all rolling/expanding stats for all teams
        in a single, vectorized operation.
        """
        print("  Querying all historical team games...")
        team_games = self.db.query("""
            SELECT tg.*, g.startDate
            FROM team_games tg
            JOIN games g ON tg.gameId = g.id
            WHERE g.status = 'Final'
            ORDER BY g.startDate
        """)
        
        # Validate columns
        missing_cols = [col for col in self.stat_columns if col not in team_games.columns]
        if missing_cols:
            print(f"Warning: Missing columns in team_games: {missing_cols}")
            self.stat_columns = [col for col in self.stat_columns if col in team_games.columns]
        
        if not self.stat_columns:
            print("Error: No valid stat columns to calculate. Aborting.")
            return pd.DataFrame()

        print("  Pre-calculating rolling statistics (vectorized)...")
        # Sort by team, then date. This is crucial for groupby operations.
        team_games.sort_values(by=['teamId', 'startDate'], inplace=True)
        
        # Group by team
        grouped = team_games.groupby('teamId')
        
        # Use a list to store new series, then concat at the end
        all_new_cols = []

        for window in windows:
            for stat in self.stat_columns:
                col_name = f'{stat}_season' if window == 'season' else f'{stat}_L{window}'
                
                if window == 'season':
                    # Season-to-date (expanding) average, shifted
                    stat_series = grouped[stat].expanding(min_periods=1).mean().shift(1)
                else:
                    # Fixed window rolling average, shifted
                    stat_series = grouped[stat].rolling(window=window, min_periods=1).mean().shift(1)
                
                stat_series.name = col_name
                all_new_cols.append(stat_series)

        # Combine the new stat columns with the original team_games df
        rolling_stats_df = pd.concat([team_games] + all_new_cols, axis=1)
        
        # We only need the stats, teamId, and startDate for merging
        lookup_cols = ['teamId', 'startDate'] + [col.name for col in all_new_cols]
        
        return rolling_stats_df[lookup_cols].sort_values(by='startDate')
    
    def _merge_stats_vectorized(self, df, all_stats_df, windows):
        """
        Efficiently merges rolling stats using two full-dataframe merge_asof calls.
        """
        print("  Merging statistics to main dataframe (vectorized)...")
        
        # We need to save the original index to restore order at the end
        df = df.reset_index(drop=False)
        original_index_name = 'index' if 'index' not in df.columns else 'original_index'
        if 'index' in df.columns and original_index_name == 'original_index':
            df = df.rename(columns={'index': original_index_name})
        
        # Both DFs must be sorted by the 'on' key (startDate) for merge_asof
        df_sorted = df.sort_values('startDate')
        
        # --- 1. Process Home Teams ---
        print("    Processing home team statistics...")
        # Prepare lookup table for home teams
        # Rename stat columns to 'home_*' and 'teamId' to 'homeTeamId' for joining
        home_suffixes = {
            col: f'home_{col}' for col in all_stats_df.columns 
            if col not in ['teamId', 'startDate']
        }
        home_lookup = all_stats_df.rename(columns=home_suffixes)
        home_lookup = home_lookup.rename(columns={'teamId': 'homeTeamId'})
        
        # Perform the first merge
        df_merged = pd.merge_asof(
            df_sorted,
            home_lookup,
            on='startDate',
            by='homeTeamId',
            direction='backward'
        )
        
        # --- 2. Process Away Teams ---
        print("    Processing away team statistics...")
        # Prepare lookup table for away teams
        away_suffixes = {
            col: f'away_{col}' for col in all_stats_df.columns 
            if col not in ['teamId', 'startDate']
        }
        away_lookup = all_stats_df.rename(columns=away_suffixes)
        away_lookup = away_lookup.rename(columns={'teamId': 'awayTeamId'})
        
        # Perform the second merge on the result of the first
        df_final = pd.merge_asof(
            df_merged,  # Use the result from the home merge
            away_lookup,
            on='startDate',
            by='awayTeamId',
            direction='backward'
        )
        
        # Restore the original dataframe order
        index_col = original_index_name if original_index_name in df_final.columns else 'index'
        if index_col in df_final.columns:
            df_final = df_final.set_index(index_col).sort_index()
        
        return df_final