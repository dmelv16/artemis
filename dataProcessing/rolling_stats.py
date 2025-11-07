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
            SELECT tg.*
            FROM team_games tg
            JOIN games g ON tg.gameId = g.id
            WHERE g.status = 'Final'
            ORDER BY g.startDate
        """)
        
        # FIX 1: Remove duplicate columns (keep first occurrence)
        team_games = team_games.loc[:, ~team_games.columns.duplicated()]
        
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
        
        # CRITICAL FIX: Reset index before grouping to avoid MultiIndex issues
        team_games = team_games.reset_index(drop=True)
        
        # Group by team
        grouped = team_games.groupby('teamId')
        
        # Use a dictionary to store new columns for easier management
        new_cols_dict = {}

        for window in windows:
            for stat in self.stat_columns:
                col_name = f'{stat}_season' if window == 'season' else f'{stat}_L{window}'
                
                if window == 'season':
                    # Season-to-date (expanding) average, shifted
                    stat_series = grouped[stat].expanding(min_periods=1).mean().shift(1)
                else:
                    # Fixed window rolling average, shifted
                    stat_series = grouped[stat].rolling(window=window, min_periods=1).mean().shift(1)
                
                # CRITICAL FIX: Reset the index to flatten the MultiIndex
                stat_series = stat_series.reset_index(level=0, drop=True)
                new_cols_dict[col_name] = stat_series

        # Create a DataFrame from the dictionary
        new_stats_df = pd.DataFrame(new_cols_dict, index=team_games.index)
        
        # Combine with the original columns we need
        rolling_stats_df = pd.concat([
            team_games[['teamId', 'startDate']], 
            new_stats_df
        ], axis=1)
        
        # CRITICAL FIX: Ensure teamId has no nulls and proper dtype
        rolling_stats_df['teamId'] = rolling_stats_df['teamId'].astype('Int64')
        
        # Verify no nulls in teamId
        if rolling_stats_df['teamId'].isna().any():
            print(f"    ERROR: Found {rolling_stats_df['teamId'].isna().sum()} null teamIds after calculation!")
            rolling_stats_df = rolling_stats_df.dropna(subset=['teamId'])
        
        return rolling_stats_df.sort_values(by='startDate').reset_index(drop=True)
    
    def _merge_stats_vectorized(self, df, all_stats_df, windows):
        """
        Efficiently merges rolling stats using two full-dataframe merge_asof calls.
        """
        print("  Merging statistics to main dataframe (vectorized)...")
        
        # FIX 2: Ensure consistent data types for merge keys
        # Convert team IDs to int64 in both dataframes
        if 'homeTeamId' in df.columns:
            df['homeTeamId'] = df['homeTeamId'].astype('Int64')
        if 'awayTeamId' in df.columns:
            df['awayTeamId'] = df['awayTeamId'].astype('Int64')
        if 'teamId' in all_stats_df.columns:
            all_stats_df['teamId'] = all_stats_df['teamId'].astype('Int64')
        
        # FIX 3: Remove rows with null team IDs from all_stats_df
        # merge_asof cannot handle null values in the 'by' column
        initial_count = len(all_stats_df)
        all_stats_df = all_stats_df.dropna(subset=['teamId'])
        if len(all_stats_df) < initial_count:
            print(f"    Removed {initial_count - len(all_stats_df)} rows with null teamId from stats")
        
        # We need to save the original index to restore order at the end
        df = df.reset_index(drop=False)
        original_index_name = 'index' if 'index' not in df.columns else 'original_index'
        if 'index' in df.columns and original_index_name == 'original_index':
            df = df.rename(columns={'index': original_index_name})
        
        # FIX 4: Track rows with null team IDs to handle them separately
        null_home_mask = df['homeTeamId'].isna()
        null_away_mask = df['awayTeamId'].isna()
        has_nulls = null_home_mask.any() or null_away_mask.any()
        
        if has_nulls:
            print(f"    Warning: Found {null_home_mask.sum()} rows with null homeTeamId and {null_away_mask.sum()} with null awayTeamId")
            # Temporarily fill nulls with a placeholder for merging, we'll handle these separately
            df['homeTeamId'] = df['homeTeamId'].fillna(-999)
            df['awayTeamId'] = df['awayTeamId'].fillna(-999)
        
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
        
        # FIX 5: Restore null team IDs if they existed
        if has_nulls:
            if 'homeTeamId' in df_final.columns:
                df_final.loc[null_home_mask, 'homeTeamId'] = None
            if 'awayTeamId' in df_final.columns:
                df_final.loc[null_away_mask, 'awayTeamId'] = None
        
        return df_final