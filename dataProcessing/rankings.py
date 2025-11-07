"""
Adds team ranking features using vectorized operations.
"""
import pandas as pd
import numpy as np

class RankingsProcessor:
    def __init__(self, db_conn):
        self.db = db_conn
         
    def add_features(self, df):
        print("Adding ranking features...")
        
        # Always ensure season column exists by deriving from startDate
        if 'season' not in df.columns:
            print("  Deriving 'season' column from startDate...")
            df['startDate'] = pd.to_datetime(df['startDate'])
            df['season'] = df['startDate'].apply(
                lambda x: x.year + 1 if x.month >= 8 else x.year
            )
        
        # Ensure week column exists
        if 'week' not in df.columns:
            raise ValueError("'week' column is required. Ensure add_week_column() was called before this processor")
        
        # Add a unique game index to preserve order and handle duplicates
        df = df.reset_index(drop=True)
        df['_game_idx'] = df.index
        
        # Ensure proper data types for merge keys
        df['season'] = df['season'].astype('int64')
        df['week'] = df['week'].astype('int64')
        df['homeTeamId'] = df['homeTeamId'].astype('int64')
        df['awayTeamId'] = df['awayTeamId'].astype('int64')
        
        rankings = self.db.query("""
            SELECT r.season, r.week, r.teamId, r.pollType, r.ranking, r.points as poll_points
            FROM rankings r
            WHERE r.pollType IN ('AP Top 25', 'Coaches Poll')
                AND r.ranking IS NOT NULL
                AND r.seasonType = 'regular'
        """)
        
        if len(rankings) == 0:
            print("Warning: No rankings data found")
            return df
        
        # Ensure proper data types in rankings
        rankings['season'] = rankings['season'].astype('int64')
        rankings['week'] = rankings['week'].astype('int64')
        rankings['teamId'] = rankings['teamId'].astype('int64')
        
        # Remove duplicates from rankings
        rankings = rankings.drop_duplicates(subset=['season', 'week', 'teamId', 'pollType'], keep='first')
        
        for poll_type in ['AP Top 25', 'Coaches Poll']:
            poll_short = 'AP' if poll_type == 'AP Top 25' else 'Coaches'
            poll_rankings = rankings[rankings['pollType'] == poll_type].copy()
            
            if len(poll_rankings) == 0:
                print(f"Warning: No rankings found for {poll_type}")
                continue
            
            # HOME TEAM
            home_ranks = poll_rankings[['season', 'week', 'teamId', 'ranking', 'poll_points']].copy()
            home_ranks = home_ranks.rename(columns={
                'teamId': 'homeTeamId',
                'ranking': f'home_{poll_short}_rank',
                'poll_points': f'home_{poll_short}_poll_points'
            })
            
            # Use a simple merge on exact week match, then forward fill within season/team
            df = df.merge(
                home_ranks,
                on=['season', 'homeTeamId', 'week'],
                how='left'
            )
            
            # Forward fill rankings within each season/team group
            df = df.sort_values(['season', 'homeTeamId', 'week', '_game_idx'])
            df[f'home_{poll_short}_rank'] = df.groupby(['season', 'homeTeamId'])[f'home_{poll_short}_rank'].ffill()
            df[f'home_{poll_short}_poll_points'] = df.groupby(['season', 'homeTeamId'])[f'home_{poll_short}_poll_points'].ffill()
            
            # AWAY TEAM
            away_ranks = poll_rankings[['season', 'week', 'teamId', 'ranking', 'poll_points']].copy()
            away_ranks = away_ranks.rename(columns={
                'teamId': 'awayTeamId',
                'ranking': f'away_{poll_short}_rank',
                'poll_points': f'away_{poll_short}_poll_points'
            })
            
            df = df.merge(
                away_ranks,
                on=['season', 'awayTeamId', 'week'],
                how='left'
            )
            
            # Forward fill rankings within each season/team group
            df = df.sort_values(['season', 'awayTeamId', 'week', '_game_idx'])
            df[f'away_{poll_short}_rank'] = df.groupby(['season', 'awayTeamId'])[f'away_{poll_short}_rank'].ffill()
            df[f'away_{poll_short}_poll_points'] = df.groupby(['season', 'awayTeamId'])[f'away_{poll_short}_poll_points'].ffill()
            
            self._add_derived_features(df, poll_short)
        
        # Restore original order and drop helper column
        df = df.sort_values('_game_idx').drop(columns=['_game_idx']).reset_index(drop=True)
        
        print("Added ranking features")
        return df
    
    def _add_derived_features(self, df, poll_short):
        """Add derived features from ranking data."""
        home_rank_col = f'home_{poll_short}_rank'
        away_rank_col = f'away_{poll_short}_rank'
        home_points_col = f'home_{poll_short}_poll_points'
        away_points_col = f'away_{poll_short}_poll_points'
        
        df[f'home_{poll_short}_ranked'] = df[home_rank_col].notna()
        df[f'away_{poll_short}_ranked'] = df[away_rank_col].notna()
        
        unranked_rank_val = 50
        home_rank_filled = df[home_rank_col].fillna(unranked_rank_val)
        away_rank_filled = df[away_rank_col].fillna(unranked_rank_val)
        home_points_filled = df[home_points_col].fillna(0)
        away_points_filled = df[away_points_col].fillna(0)
        
        df[f'{poll_short}_rank_diff'] = away_rank_filled - home_rank_filled
        df[f'{poll_short}_points_diff'] = home_points_filled - away_points_filled
        
        for team_type in ['home', 'away']:
            rank_col = f'{team_type}_{poll_short}_rank'
            df[f'{team_type}_{poll_short}_tier'] = pd.cut(
                df[rank_col],
                bins=[0, 5, 10, 25, np.inf],
                labels=['Top_5', 'Top_10', 'Top_25', 'Unranked'],
                right=True
            )
            # Simply fill NaN values with 'Unranked' - no need to add_categories since it's already a label
            df[f'{team_type}_{poll_short}_tier'] = df[f'{team_type}_{poll_short}_tier'].fillna('Unranked')
        
        df[f'{poll_short}_matchup_type'] = 'Unranked_vs_Unranked'
        both_ranked = df[f'home_{poll_short}_ranked'] & df[f'away_{poll_short}_ranked']
        home_only = df[f'home_{poll_short}_ranked'] & ~df[f'away_{poll_short}_ranked']
        away_only = ~df[f'home_{poll_short}_ranked'] & df[f'away_{poll_short}_ranked']
        
        df.loc[both_ranked, f'{poll_short}_matchup_type'] = 'Ranked_vs_Ranked'
        df.loc[home_only, f'{poll_short}_matchup_type'] = 'Ranked_vs_Unranked'
        df.loc[away_only, f'{poll_short}_matchup_type'] = 'Unranked_vs_Ranked'