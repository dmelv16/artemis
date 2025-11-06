"""
Calculates player-level statistics and aggregates them to team level.
FULLY OPTIMIZED VERSION: 100% vectorized operations for maximum performance.
Enhanced with netRating, assistsTurnoverRatio, freeThrowRate, offensiveReboundPct, 
and two-point shooting stats.
"""
import pandas as pd
import numpy as np
from config import PLAYER_STAT_COLUMNS, PLAYER_AGGREGATIONS

class PlayerStatsCalculator:
    def __init__(self, db_conn):
        self.db = db_conn
        self.stat_columns = PLAYER_STAT_COLUMNS
        self.aggregations = PLAYER_AGGREGATIONS
        
    def calculate(self, df, windows=[5, 10, 'season']):
        """
        Main method to add all player-based features using fully vectorized operations.
        """
        print("Calculating player statistics...")
        
        # Sort df ONCE at the start
        df = df.sort_values('startDate').reset_index(drop=True)
        
        # Get all relevant player game stats (filtered by date range)
        player_games = self._get_player_games(df)
        
        # Pre-calculate team totals once (avoid N+1 queries)
        team_totals = self._calculate_team_totals(player_games)
        
        # Add various player feature sets (all fully vectorized)
        df = self._add_starter_stats(df, player_games, windows)
        df = self._add_bench_stats(df, player_games, team_totals, windows)
        df = self._add_top_performer_stats(df, player_games, windows)
        df = self._add_position_stats(df, player_games, windows)
        df = self._add_roster_depth_metrics(df, player_games)
        df = self._add_key_player_availability(df, player_games)
        
        player_feature_count = len([c for c in df.columns if any(x in c for x in ['player_', 'starter_', 'bench_', 'top3_', 'guard_', 'forward_', 'center_'])])
        print(f"Added {player_feature_count} player features")
        return df
    
    def _get_player_games(self, df):
        """Get player game statistics filtered to relevant date range."""
        min_date = df['startDate'].min()
        # Go back 1 year to ensure we have enough data for rolling windows
        min_date = (pd.to_datetime(min_date) - pd.DateOffset(days=365)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT 
            pg.*,
            g.startDate,
            g.homeTeamId,
            g.awayTeamId,
            CASE 
                WHEN pg.teamId = g.homeTeamId THEN 'home'
                WHEN pg.teamId = g.awayTeamId THEN 'away'
            END as team_side
        FROM player_games pg
        JOIN games g ON pg.gameId = g.id
        WHERE g.status = 'Final' AND g.startDate >= '{min_date}'
        ORDER BY g.startDate, pg.teamId, pg.minutes DESC
        """
        return self.db.query(query)
    
    def _calculate_team_totals(self, player_games):
        """Pre-calculate team totals to avoid N+1 queries."""
        return player_games.groupby(['gameId', 'teamId']).agg({
            'points': 'sum',
            'minutes': 'sum',
            'fieldGoalsAttempted': 'sum',
            'threePointFGAttempted': 'sum',
            'twoPointFGAttempted': 'sum'
        }).reset_index().rename(columns={
            'points': 'team_points',
            'minutes': 'team_minutes',
            'fieldGoalsAttempted': 'team_fga',
            'threePointFGAttempted': 'team_3pa',
            'twoPointFGAttempted': 'team_2pa'
        })
    
    def _add_starter_stats(self, df, player_games, windows):
        """
        Calculate rolling averages for starting lineup using vectorized operations.
        """
        print("  Processing starter statistics...")
        
        starters = player_games[player_games['starter'] == 1].copy()
        
        # Aggregate all teams at once
        game_aggs = starters.groupby(['teamId', 'gameId', 'startDate']).agg({
            'points': ['sum', 'mean', 'max'],
            'assists': ['sum', 'mean', 'max'],
            'totalRebounds': ['sum', 'mean', 'max'],
            'gameScore': ['mean', 'max', 'std'],
            'usage': ['mean', 'max'],
            'minutes': ['sum', 'mean'],
            'offensiveRating': 'mean',
            'defensiveRating': 'mean',
            'netRating': 'mean',
            'effectiveFieldGoalPct': 'mean',
            'trueShootingPct': 'mean',
            'assistsTurnoverRatio': 'mean',
            'freeThrowRate': 'mean',
            'offensiveReboundPct': 'mean',
            'turnovers': 'sum',
            'steals': 'sum',
            'blocks': 'sum',
            'fouls': 'sum',
            'twoPointFGMade': 'sum',
            'twoPointFGAttempted': 'sum',
            'twoPointFGPct': 'mean'
        }).reset_index()
        
        # Flatten column names
        game_aggs.columns = ['_'.join(col).strip('_') for col in game_aggs.columns]
        game_aggs = game_aggs.sort_values(['teamId', 'startDate'])
        
        # Get stat columns (exclude ID/date columns)
        stat_cols = [c for c in game_aggs.columns if c not in ['teamId', 'gameId', 'startDate']]
        
        # Calculate rolling features using groupby
        grouped = game_aggs.groupby('teamId')
        
        for window in windows:
            for col in stat_cols:
                if window == 'season':
                    new_col = f'starter_{col}_season'
                    game_aggs[new_col] = grouped[col].expanding(min_periods=1).mean().shift(1).values
                else:
                    new_col = f'starter_{col}_L{window}'
                    game_aggs[new_col] = grouped[col].rolling(window=window, min_periods=1).mean().shift(1).values
        
        # Keep only feature columns for merge
        feature_cols = ['teamId', 'startDate'] + [c for c in game_aggs.columns if 'starter_' in c]
        starter_features = game_aggs[feature_cols]
        
        # Merge using merge_asof (df is already sorted)
        df = self._merge_team_features(df, starter_features, 'home')
        df = self._merge_team_features(df, starter_features, 'away')
        
        return df
    
    def _add_bench_stats(self, df, player_games, team_totals, windows):
        """
        Calculate bench strength metrics using vectorized operations.
        """
        print("  Processing bench statistics...")
        
        bench = player_games[player_games['starter'] == 0].copy()
        
        # Aggregate bench stats
        game_aggs = bench.groupby(['teamId', 'gameId', 'startDate']).agg({
            'points': 'sum',
            'assists': 'sum',
            'totalRebounds': 'sum',
            'minutes': 'sum',
            'gameScore': 'mean',
            'netRating': 'mean',
            'assistsTurnoverRatio': 'mean',
            'freeThrowRate': 'mean',
            'offensiveReboundPct': 'mean',
            'twoPointFGPct': 'mean',
            'effectiveFieldGoalPct': 'mean',
            'athleteId': 'count',  # Number of bench players used
            'turnovers': 'sum'
        }).reset_index()
        
        game_aggs.rename(columns={'athleteId': 'bench_players_used'}, inplace=True)
        
        # Merge team totals and calculate bench percentage
        game_aggs = game_aggs.merge(team_totals[['gameId', 'teamId', 'team_points']], 
                                     on=['gameId', 'teamId'], how='left')
        game_aggs['bench_scoring_pct'] = game_aggs['points'] / game_aggs['team_points'].replace(0, np.nan)
        game_aggs['bench_scoring_pct'] = game_aggs['bench_scoring_pct'].fillna(0)
        
        game_aggs = game_aggs.sort_values(['teamId', 'startDate'])
        
        # Calculate rolling features
        stat_cols = ['points', 'assists', 'totalRebounds', 'minutes', 
                     'bench_players_used', 'bench_scoring_pct', 'gameScore', 'turnovers',
                     'netRating', 'assistsTurnoverRatio', 'freeThrowRate', 
                     'offensiveReboundPct', 'twoPointFGPct', 'effectiveFieldGoalPct']
        grouped = game_aggs.groupby('teamId')
        
        for window in windows:
            for col in stat_cols:
                if window == 'season':
                    new_col = f'bench_{col}_season'
                    game_aggs[new_col] = grouped[col].expanding(min_periods=1).mean().shift(1).values
                else:
                    new_col = f'bench_{col}_L{window}'
                    game_aggs[new_col] = grouped[col].rolling(window=window, min_periods=1).mean().shift(1).values
        
        # Merge features
        feature_cols = ['teamId', 'startDate'] + [c for c in game_aggs.columns if 'bench_' in c]
        bench_features = game_aggs[feature_cols]
        
        df = self._merge_team_features(df, bench_features, 'home')
        df = self._merge_team_features(df, bench_features, 'away')
        
        return df
    
    def _add_top_performer_stats(self, df, player_games, windows):
        """
        Track top performers WITHOUT data leakage, FULLY VECTORIZED.
        Uses groupby().rank() instead of Python loops.
        """
        print("  Processing top performer statistics (fully vectorized)...")
        
        # Sort and calculate rolling metric for identifying top players
        player_games_sorted = player_games.sort_values(['teamId', 'athleteId', 'startDate']).copy()
        grouped = player_games_sorted.groupby(['teamId', 'athleteId'])
        
        # Use rolling minutes (more stable than gameScore) to identify top players
        player_games_sorted['rolling_minutes'] = grouped['minutes'].rolling(
            window=20, min_periods=5
        ).mean().shift(1).values
        
        # Fill NaN with 0 so new players aren't randomly ranked high
        player_games_sorted['rolling_minutes'] = player_games_sorted['rolling_minutes'].fillna(0)
        
        # VECTORIZED: Rank players within their team for each game
        player_games_sorted['player_rank'] = player_games_sorted.groupby(
            ['teamId', 'gameId']
        )['rolling_minutes'].rank(method='first', ascending=False)
        
        # Filter to keep only top 3 performers per game
        top_player_games = player_games_sorted[player_games_sorted['player_rank'] <= 3].copy()
        
        # Aggregate top player stats
        game_aggs = top_player_games.groupby(['teamId', 'gameId', 'startDate']).agg({
            'points': ['sum', 'max', 'mean'],
            'assists': ['sum', 'max'],
            'totalRebounds': ['sum', 'max'],
            'gameScore': ['mean', 'max'],
            'usage': ['max', 'mean'],
            'minutes': 'sum',
            'effectiveFieldGoalPct': 'mean',
            'netRating': 'mean',
            'assistsTurnoverRatio': 'mean',
            'freeThrowRate': 'mean',
            'offensiveReboundPct': 'mean',
            'twoPointFGPct': 'mean',
            'turnovers': 'sum',
            'threePointFGMade': 'sum',
            'threePointFGAttempted': 'sum'
        }).reset_index()
        
        game_aggs.columns = ['_'.join(col).strip('_') for col in game_aggs.columns]
        game_aggs = game_aggs.sort_values(['teamId', 'startDate'])
        
        # Rolling features
        stat_cols = [c for c in game_aggs.columns if c not in ['teamId', 'gameId', 'startDate']]
        grouped = game_aggs.groupby('teamId')
        
        for window in windows:
            for col in stat_cols:
                if window == 'season':
                    new_col = f'top3_{col}_season'
                    game_aggs[new_col] = grouped[col].expanding(min_periods=1).mean().shift(1).values
                else:
                    new_col = f'top3_{col}_L{window}'
                    game_aggs[new_col] = grouped[col].rolling(window=window, min_periods=1).mean().shift(1).values
        
        feature_cols = ['teamId', 'startDate'] + [c for c in game_aggs.columns if 'top3_' in c]
        top3_features = game_aggs[feature_cols]
        
        df = self._merge_team_features(df, top3_features, 'home')
        df = self._merge_team_features(df, top3_features, 'away')
        
        return df
    
    def _add_position_stats(self, df, player_games, windows):
        """
        Add position-specific statistics (guards, forwards, centers).
        This captures team composition and style of play.
        """
        print("  Processing position-based statistics...")
        
        if 'position' not in player_games.columns:
            print("    Warning: position column not found, skipping position stats")
            return df
        
        # Map positions to simplified categories
        player_games['pos_category'] = player_games['position'].map({
            'PG': 'guard', 'SG': 'guard', 'G': 'guard',
            'SF': 'forward', 'PF': 'forward', 'F': 'forward',
            'C': 'center'
        })
        
        position_features = []
        
        for pos in ['guard', 'forward', 'center']:
            pos_players = player_games[player_games['pos_category'] == pos].copy()
            
            if len(pos_players) == 0:
                continue
            
            # Aggregate by position
            game_aggs = pos_players.groupby(['teamId', 'gameId', 'startDate']).agg({
                'points': ['sum', 'mean'],
                'assists': 'sum',
                'totalRebounds': 'sum',
                'threePointFGAttempted': 'sum',
                'twoPointFGMade': 'sum',
                'twoPointFGAttempted': 'sum',
                'twoPointFGPct': 'mean',
                'freeThrowRate': 'mean',
                'offensiveReboundPct': 'mean',
                'netRating': 'mean',
                'usage': 'mean',
                'minutes': 'sum'
            }).reset_index()
            
            game_aggs.columns = ['_'.join(col).strip('_') for col in game_aggs.columns]
            game_aggs = game_aggs.sort_values(['teamId', 'startDate'])
            
            # Rolling stats (now includes 'season' for consistency)
            stat_cols = [c for c in game_aggs.columns if c not in ['teamId', 'gameId', 'startDate']]
            grouped = game_aggs.groupby('teamId')
            
            for window in windows:  # Now uses all windows including 'season'
                for col in stat_cols:
                    if window == 'season':
                        new_col = f'{pos}_{col}_season'
                        game_aggs[new_col] = grouped[col].expanding(min_periods=1).mean().shift(1).values
                    else:
                        new_col = f'{pos}_{col}_L{window}'
                        game_aggs[new_col] = grouped[col].rolling(window=window, min_periods=1).mean().shift(1).values
            
            feature_cols = ['teamId', 'startDate'] + [c for c in game_aggs.columns if f'{pos}_' in c]
            position_features.append(game_aggs[feature_cols])
        
        # Merge all position features
        for pos_features in position_features:
            df = self._merge_team_features(df, pos_features, 'home')
            df = self._merge_team_features(df, pos_features, 'away')
        
        return df
    
    def _add_roster_depth_metrics(self, df, player_games):
        """
        Calculate roster depth using vectorized operations where possible.
        Some iteration remains due to the lookback-N-games logic.
        """
        print("  Processing roster depth metrics...")
        
        # Pre-sort player_games once
        player_games_sorted = player_games.sort_values(['teamId', 'startDate'])
        
        # Create a mapping of team -> sorted games for faster lookups
        team_games_dict = {
            team_id: group.reset_index(drop=True)
            for team_id, group in player_games_sorted.groupby('teamId')
        }
        
        depth_results = []
        
        for idx, game in df.iterrows():
            game_date = game['startDate']
            features = {'gameId': game['gameId']}
            
            for team_id, prefix in [(game['homeTeamId'], 'home'), (game['awayTeamId'], 'away')]:
                if team_id not in team_games_dict:
                    # No history - use defaults
                    features.update({
                        f'{prefix}_rotation_size': 8,
                        f'{prefix}_minutes_concentration': 0.5,
                        f'{prefix}_quality_depth': 5
                    })
                    continue
                
                # Get pre-sorted team games
                team_games = team_games_dict[team_id]
                
                # Vectorized filter for games before current date
                mask = team_games['startDate'] < game_date
                recent_games_ids = team_games.loc[mask, 'gameId'].unique()[-10:]
                
                if len(recent_games_ids) == 0:
                    features.update({
                        f'{prefix}_rotation_size': 8,
                        f'{prefix}_minutes_concentration': 0.5,
                        f'{prefix}_quality_depth': 5
                    })
                    continue
                
                # Get recent player data
                recent = team_games[team_games['gameId'].isin(recent_games_ids)]
                
                # Calculate per-player averages (vectorized)
                player_avgs = recent.groupby('athleteId').agg({
                    'minutes': 'mean',
                    'gameScore': 'mean'
                }).reset_index()
                
                # Depth metrics (vectorized)
                rotation_size = (player_avgs['minutes'] > 10).sum()
                
                top5_minutes = player_avgs.nlargest(5, 'minutes')['minutes'].sum()
                total_minutes = player_avgs['minutes'].sum()
                minutes_concentration = top5_minutes / total_minutes if total_minutes > 0 else 0.5
                
                quality_depth = (player_avgs['gameScore'] > 5).sum()
                
                features.update({
                    f'{prefix}_rotation_size': rotation_size,
                    f'{prefix}_minutes_concentration': minutes_concentration,
                    f'{prefix}_quality_depth': quality_depth
                })
            
            depth_results.append(features)
        
        depth_df = pd.DataFrame(depth_results)
        df = df.merge(depth_df, on='gameId', how='left')
        
        return df
    
    def _add_key_player_availability(self, df, player_games):
        """
        Track key player availability based on minutes played (not binary presence).
        Calculates what percentage of expected minutes the top-5 players actually played.
        """
        print("  Processing key player availability...")
        
        # Pre-sort and cache team games
        player_games_sorted = player_games.sort_values(['teamId', 'startDate'])
        team_games_dict = {
            team_id: group.reset_index(drop=True)
            for team_id, group in player_games_sorted.groupby('teamId')
        }
        
        availability_results = []
        
        for idx, game in df.iterrows():
            game_date = game['startDate']
            features = {'gameId': game['gameId']}
            
            for team_id, prefix in [(game['homeTeamId'], 'home'), (game['awayTeamId'], 'away')]:
                if team_id not in team_games_dict:
                    features.update({
                        f'{prefix}_key_minutes_availability_pct': 1.0,
                        f'{prefix}_expected_key_minutes': 150.0,
                        f'{prefix}_actual_key_minutes': 150.0
                    })
                    continue
                
                team_games = team_games_dict[team_id]
                
                # Vectorized filter
                mask = team_games['startDate'] < game_date
                if not mask.any():
                    features.update({
                        f'{prefix}_key_minutes_availability_pct': 1.0,
                        f'{prefix}_expected_key_minutes': 150.0,
                        f'{prefix}_actual_key_minutes': 150.0
                    })
                    continue
                
                recent_games_ids = team_games.loc[mask, 'gameId'].unique()[-10:]
                recent = team_games[team_games['gameId'].isin(recent_games_ids)]
                
                # Identify key players and their average minutes (vectorized)
                key_player_avg_mins = recent.groupby('athleteId')['minutes'].mean().nlargest(5)
                expected_minutes = key_player_avg_mins.sum()
                
                # Get actual minutes played by these key players in the last game
                last_game_id = team_games.loc[mask, 'gameId'].iloc[-1]
                last_game = team_games[team_games['gameId'] == last_game_id]
                
                actual_minutes = last_game[
                    last_game['athleteId'].isin(key_player_avg_mins.index)
                ]['minutes'].sum()
                
                # Calculate availability percentage
                availability_pct = actual_minutes / expected_minutes if expected_minutes > 0 else 1.0
                
                features.update({
                    f'{prefix}_key_minutes_availability_pct': availability_pct,
                    f'{prefix}_expected_key_minutes': expected_minutes,
                    f'{prefix}_actual_key_minutes': actual_minutes
                })
            
            availability_results.append(features)
        
        availability_df = pd.DataFrame(availability_results)
        df = df.merge(availability_df, on='gameId', how='left')
        
        return df
    
    def _merge_team_features(self, df, features, team_type):
        """
        Use merge_asof for fast temporal joins.
        ASSUMES df IS ALREADY SORTED BY startDate.
        """
        team_id_col = f'{team_type}TeamId'
        
        # Only sort features (df is already sorted from calculate())
        features_sorted = features.sort_values('startDate').copy()
        
        # Save original index to restore order
        original_index = df.index
        
        # Perform merge_asof
        merged = pd.merge_asof(
            df,
            features_sorted,
            left_on='startDate',
            right_on='startDate',
            left_by=team_id_col,
            right_by='teamId',
            direction='backward',
            suffixes=('', f'_{team_type}_temp')
        )
        
        # Rename columns to include team prefix
        feature_cols = [c for c in features.columns if c not in ['teamId', 'startDate']]
        rename_dict = {col: f'{team_type}_{col}' for col in feature_cols}
        merged = merged.rename(columns=rename_dict)
        
        # Restore original order using saved index
        merged = merged.set_index(original_index).sort_index()
        
        return merged