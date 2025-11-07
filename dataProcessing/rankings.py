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
        
        # FIXED: Query now uses week and matches actual pollType values
        rankings = self.db.query("""
            SELECT r.season, r.week, r.teamId, r.pollType, r.ranking, r.points as poll_points
            FROM rankings r
            WHERE r.pollType IN ('AP Top 25', 'Coaches Poll')
            ORDER BY r.season DESC, r.week DESC
        """)
        
        # Ensure df is sorted by week for merge_asof
        df = df.sort_values(by=['season', 'week']).reset_index(drop=True)
        
        # Process each poll type
        for poll_type in ['AP Top 25', 'Coaches Poll']:
            poll_short = 'AP' if poll_type == 'AP Top 25' else 'Coaches'
            poll_rankings = rankings[rankings['pollType'] == poll_type].copy()
            
            if len(poll_rankings) == 0:
                print(f"Warning: No rankings found for {poll_type}")
                continue
            
            # Sort poll rankings by week
            poll_rankings = poll_rankings.sort_values(by=['season', 'week'])
            
            # Merge for home team
            df = pd.merge_asof(
                df,
                poll_rankings[['season', 'week', 'teamId', 'ranking', 'poll_points']],
                on='week',
                by=['season', 'homeTeamId'],
                right_by=['season', 'teamId'],
                direction='backward',
                suffixes=('', '_temp')
            )
            
            # Rename columns for home team
            df = df.rename(columns={
                'ranking_temp': f'home_{poll_short}_rank',
                'poll_points_temp': f'home_{poll_short}_poll_points'
            })
            
            # Merge for away team
            df = pd.merge_asof(
                df,
                poll_rankings[['season', 'week', 'teamId', 'ranking', 'poll_points']],
                on='week',
                by=['season', 'awayTeamId'],
                right_by=['season', 'teamId'],
                direction='backward',
                suffixes=('', '_temp')
            )
            
            # Rename columns for away team
            df = df.rename(columns={
                'ranking_temp': f'away_{poll_short}_rank',
                'poll_points_temp': f'away_{poll_short}_poll_points'
            })
            
            # Add derived features
            self._add_derived_features(df, poll_short)
        
        print("Added ranking features")
        return df
    
    def _add_derived_features(self, df, poll_short):
        """Add rich derived features from ranking data."""
        home_rank_col = f'home_{poll_short}_rank'
        away_rank_col = f'away_{poll_short}_rank'
        home_points_col = f'home_{poll_short}_poll_points'
        away_points_col = f'away_{poll_short}_poll_points'
        
        # Boolean: is team ranked?
        df[f'home_{poll_short}_ranked'] = df[home_rank_col].notna()
        df[f'away_{poll_short}_ranked'] = df[away_rank_col].notna()
        
        # Fill NaNs with logical values for unranked teams
        # Use 50 for rank (higher than any ranked team)
        # Use 0 for points (unranked teams have no poll points)
        unranked_rank_val = 50
        home_rank_filled = df[home_rank_col].fillna(unranked_rank_val)
        away_rank_filled = df[away_rank_col].fillna(unranked_rank_val)
        home_points_filled = df[home_points_col].fillna(0)
        away_points_filled = df[away_points_col].fillna(0)
        
        # Rank difference (positive means home team ranked higher)
        # Now handles unranked teams: #5 vs unranked = 50-5=45
        df[f'{poll_short}_rank_diff'] = away_rank_filled - home_rank_filled
        
        # Poll points difference (positive means home team has more points)
        # Now handles unranked teams: ranked vs unranked = points-0=points
        df[f'{poll_short}_points_diff'] = home_points_filled - away_points_filled
        
        # Categorical rank tiers
        for team_type in ['home', 'away']:
            rank_col = f'{team_type}_{poll_short}_rank'
            df[f'{team_type}_{poll_short}_tier'] = pd.cut(
                df[rank_col],
                bins=[0, 5, 10, 25, np.inf],
                labels=['Top_5', 'Top_10', 'Top_25', 'Unranked'],
                right=True
            )
            # Fill NaN with 'Unranked'
            df[f'{team_type}_{poll_short}_tier'] = df[f'{team_type}_{poll_short}_tier'].cat.add_categories('Unranked').fillna('Unranked')
        
        # Matchup type feature
        df[f'{poll_short}_matchup_type'] = 'Unranked_vs_Unranked'
        both_ranked = df[f'home_{poll_short}_ranked'] & df[f'away_{poll_short}_ranked']
        home_only = df[f'home_{poll_short}_ranked'] & ~df[f'away_{poll_short}_ranked']
        away_only = ~df[f'home_{poll_short}_ranked'] & df[f'away_{poll_short}_ranked']
        
        df.loc[both_ranked, f'{poll_short}_matchup_type'] = 'Ranked_vs_Ranked'
        df.loc[home_only, f'{poll_short}_matchup_type'] = 'Ranked_vs_Unranked'
        df.loc[away_only, f'{poll_short}_matchup_type'] = 'Unranked_vs_Ranked'