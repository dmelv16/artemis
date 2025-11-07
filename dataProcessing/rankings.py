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
        
        rankings = self.db.query("""
            SELECT r.season, r.week, r.teamId, r.pollType, r.ranking, r.points as poll_points
            FROM rankings r
            WHERE r.pollType IN ('AP Top 25', 'Coaches Poll')
            ORDER BY r.season DESC, r.week DESC
        """)
        
        # Ensure df has required columns
        required_cols = ['season', 'week', 'homeTeamId', 'awayTeamId']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")
        
        for poll_type in ['AP Top 25', 'Coaches Poll']:
            poll_short = 'AP' if poll_type == 'AP Top 25' else 'Coaches'
            poll_rankings = rankings[rankings['pollType'] == poll_type].copy()
            
            if len(poll_rankings) == 0:
                print(f"Warning: No rankings found for {poll_type}")
                continue
            
            # Merge for home team
            home_rankings = poll_rankings[['season', 'week', 'teamId', 'ranking', 'poll_points']].copy()
            home_rankings = home_rankings.rename(columns={
                'teamId': 'homeTeamId',
                'ranking': f'home_{poll_short}_rank',
                'poll_points': f'home_{poll_short}_poll_points'
            })
            
            # Sort both dataframes
            home_rankings = home_rankings.sort_values(
                by=['season', 'homeTeamId', 'week']
            ).reset_index(drop=True)
            
            df = df.sort_values(
                by=['season', 'homeTeamId', 'week']
            ).reset_index(drop=True)

            df = pd.merge_asof(
                df,
                home_rankings,
                by=['season', 'homeTeamId'],
                on='week',
                direction='backward',
                suffixes=('', '_dup')
            )
            
            # Drop any duplicate columns that might have been created
            dup_cols = [col for col in df.columns if col.endswith('_dup')]
            if dup_cols:
                df = df.drop(columns=dup_cols)

            # Merge for away team
            away_rankings = poll_rankings[['season', 'week', 'teamId', 'ranking', 'poll_points']].copy()
            away_rankings = away_rankings.rename(columns={
                'teamId': 'awayTeamId',
                'ranking': f'away_{poll_short}_rank',
                'poll_points': f'away_{poll_short}_poll_points'
            })
            
            # Sort both dataframes
            away_rankings = away_rankings.sort_values(
                by=['season', 'awayTeamId', 'week']
            ).reset_index(drop=True)
            
            df = df.sort_values(
                by=['season', 'awayTeamId', 'week']
            ).reset_index(drop=True)

            df = pd.merge_asof(
                df,
                away_rankings,
                by=['season', 'awayTeamId'],
                on='week',
                direction='backward',
                suffixes=('', '_dup')
            )
            
            # Drop any duplicate columns that might have been created
            dup_cols = [col for col in df.columns if col.endswith('_dup')]
            if dup_cols:
                df = df.drop(columns=dup_cols)
            
            self._add_derived_features(df, poll_short)
        
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
            df[f'{team_type}_{poll_short}_tier'] = df[f'{team_type}_{poll_short}_tier'].cat.add_categories('Unranked').fillna('Unranked')
        
        df[f'{poll_short}_matchup_type'] = 'Unranked_vs_Unranked'
        both_ranked = df[f'home_{poll_short}_ranked'] & df[f'away_{poll_short}_ranked']
        home_only = df[f'home_{poll_short}_ranked'] & ~df[f'away_{poll_short}_ranked']
        away_only = ~df[f'home_{poll_short}_ranked'] & df[f'away_{poll_short}_ranked']
        
        df.loc[both_ranked, f'{poll_short}_matchup_type'] = 'Ranked_vs_Ranked'
        df.loc[home_only, f'{poll_short}_matchup_type'] = 'Ranked_vs_Unranked'
        df.loc[away_only, f'{poll_short}_matchup_type'] = 'Unranked_vs_Ranked'