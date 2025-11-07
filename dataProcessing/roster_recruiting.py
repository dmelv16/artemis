"""
Adds comprehensive roster and recruiting features with proper parsing and efficiency.
"""
import pandas as pd
import numpy as np
from datetime import datetime

class RosterRecruitingProcessor:
    def __init__(self, db_conn):
        self.db = db_conn
        
    def add_features(self, df):
        print("Adding roster and recruiting features...")
        
        # Get comprehensive roster stats
        roster_df = self._get_roster_features()
        
        # Get comprehensive recruiting stats
        recruiting_df = self._get_recruiting_features()
        
        # Get recruiting trends (multi-year) - pass the aggregated df to avoid redundant query
        recruiting_trends_df = self._get_recruiting_trends(recruiting_df)
        
        # Merge features efficiently for both teams
        df = self._merge_all_features(df, roster_df, recruiting_df, recruiting_trends_df)
        
        added_features = len([c for c in df.columns if c.startswith(('home_', 'away_'))])
        print(f"Added {added_features} roster and recruiting features")
        return df
    
    def _parse_height_inches(self, height_value):
        """Convert height to numeric inches (already in inches format)."""
        return pd.to_numeric(height_value, errors='coerce')
    
    def _standardize_position(self, position):
        """Standardize position strings to simple categories."""
        if pd.isna(position):
            return 'Unknown'
        
        pos = str(position).upper().strip()
        
        # Handle multi-position players (take first position)
        if '/' in pos:
            pos = pos.split('/')[0]
        
        # Map to standard positions
        if pos in ['PG', 'POINT GUARD', 'POINT', 'G']:
            return 'G'
        elif pos in ['SG', 'SHOOTING GUARD', 'SHOOTING']:
            return 'G'
        elif pos in ['SF', 'SMALL FORWARD', 'SMALL']:
            return 'F'
        elif pos in ['PF', 'POWER FORWARD', 'POWER']:
            return 'F'
        elif pos in ['C', 'CENTER']:
            return 'C'
        elif pos in ['F', 'FORWARD']:
            return 'F'
        elif pos in ['GUARD']:
            return 'G'
        else:
            return 'Unknown'
    
    def _get_roster_features(self):
        """Calculate comprehensive roster statistics."""
        print("  Calculating roster features...")
        
        # Get raw roster data
        rosters = self.db.query("""
            SELECT 
                r.teamId,
                r.team,
                r.season,
                r.playerId,
                r.position,
                r.height,
                r.weight,
                r.startSeason,
                r.endSeason,
                r.hometownState,
                r.hometownCountry,
                r.hometownLatitude,
                r.hometownLongitude
            FROM rosters r
        """)
        
        # Parse heights and standardize positions
        rosters['height_inches'] = rosters['height'].apply(self._parse_height_inches)
        rosters['weight_pounds'] = pd.to_numeric(rosters['weight'], errors='coerce')
        rosters['position_std'] = rosters['position'].apply(self._standardize_position)
        
        # Calculate experience (years in college)
        rosters['years_experience'] = rosters['season'] - rosters['startSeason']
        rosters['years_experience'] = rosters['years_experience'].clip(lower=0, upper=4)
        
        # Classify by year
        rosters['is_freshman'] = (rosters['years_experience'] == 0).astype(int)
        rosters['is_sophomore'] = (rosters['years_experience'] == 1).astype(int)
        rosters['is_junior'] = (rosters['years_experience'] == 2).astype(int)
        rosters['is_senior'] = (rosters['years_experience'] >= 3).astype(int)
        
        # Position flags
        rosters['is_guard'] = (rosters['position_std'] == 'G').astype(int)
        rosters['is_forward'] = (rosters['position_std'] == 'F').astype(int)
        rosters['is_center'] = (rosters['position_std'] == 'C').astype(int)
        
        # Group by team and season to create aggregated features
        roster_features = rosters.groupby(['teamId', 'season']).agg({
            'playerId': 'count',  # roster_size
            'height_inches': ['mean', 'std', 'min', 'max'],
            'weight_pounds': ['mean', 'std', 'min', 'max'],
            'years_experience': ['mean', 'std'],
            'is_freshman': 'sum',
            'is_sophomore': 'sum',
            'is_junior': 'sum',
            'is_senior': 'sum',
            'is_guard': 'sum',
            'is_forward': 'sum',
            'is_center': 'sum',
            'hometownState': lambda x: x.notna().sum(),  # players with state data
            'hometownCountry': lambda x: (x == 'USA').sum(),  # US players
        }).reset_index()
        
        # Flatten column names
        roster_features.columns = [
            'teamId', 'season',
            'roster_size',
            'avg_height', 'std_height', 'min_height', 'max_height',
            'avg_weight', 'std_weight', 'min_weight', 'max_weight',
            'avg_experience', 'std_experience',
            'num_freshmen', 'num_sophomores', 'num_juniors', 'num_seniors',
            'num_guards', 'num_forwards', 'num_centers',
            'players_with_hometown', 'num_us_players'
        ]
        
        # Calculate derived features
        roster_features['pct_freshmen'] = roster_features['num_freshmen'] / roster_features['roster_size']
        roster_features['pct_seniors'] = roster_features['num_seniors'] / roster_features['roster_size']
        roster_features['pct_guards'] = roster_features['num_guards'] / roster_features['roster_size']
        roster_features['pct_forwards'] = roster_features['num_forwards'] / roster_features['roster_size']
        roster_features['pct_centers'] = roster_features['num_centers'] / roster_features['roster_size']
        roster_features['pct_us_players'] = roster_features['num_us_players'] / roster_features['roster_size']
        roster_features['height_range'] = roster_features['max_height'] - roster_features['min_height']
        roster_features['weight_range'] = roster_features['max_weight'] - roster_features['min_weight']
        
        # Calculate roster continuity (returning players from previous year)
        roster_features = self._add_roster_continuity(rosters, roster_features)
        
        return roster_features
    
    def _add_roster_continuity(self, rosters, roster_features):
        """Calculate roster continuity metrics (% of returning players) - vectorized."""
        print("  Calculating roster continuity (vectorized)...")
        
        # Select current players
        current_players = rosters[['teamId', 'season', 'playerId']]
        
        # Create a lookup of "players from last year"
        # We add 1 to their season to see if they *will* be on the next year's roster
        prior_players_lookup = rosters[['teamId', 'season', 'playerId']].copy()
        prior_players_lookup['season'] = prior_players_lookup['season'] + 1
        prior_players_lookup['was_on_prior_roster'] = 1  # Flag
        
        # Left-merge current players with the "prior players" lookup
        merged = current_players.merge(
            prior_players_lookup,
            on=['teamId', 'season', 'playerId'],
            how='left'
        )
        
        # Fill NaN with 0 (these players were not on the prior roster)
        merged['was_on_prior_roster'] = merged['was_on_prior_roster'].fillna(0)
        
        # Group by team/season and sum the flags
        continuity = merged.groupby(['teamId', 'season'])['was_on_prior_roster'].sum().reset_index()
        continuity.rename(columns={'was_on_prior_roster': 'returning_players'}, inplace=True)
        
        # Merge back to the main roster_features to get roster_size for pct calc
        roster_features = roster_features.merge(
            continuity,
            on=['teamId', 'season'],
            how='left'
        )
        
        # Calculate percentage
        roster_features['pct_returning'] = (
            roster_features['returning_players'] / roster_features['roster_size']
        ).fillna(0.0)
        
        return roster_features
    
    def _get_recruiting_features(self):
        """Calculate comprehensive recruiting statistics."""
        print("  Calculating recruiting features...")
        
        recruiting = self.db.query("""
            SELECT 
                rc.committedToId as teamId,
                rc.year as season,
                rc.stars,
                rc.rating,
                rc.ranking,
                rc.position,
                rc.heightInches,
                rc.weightPounds,
                rc.hometownState,
                rc.hometownCountry
            FROM recruiting rc
            WHERE rc.committedToId IS NOT NULL
        """)
        
        # Standardize positions
        recruiting['position_std'] = recruiting['position'].apply(self._standardize_position)
        recruiting['is_guard'] = (recruiting['position_std'] == 'G').astype(int)
        recruiting['is_forward'] = (recruiting['position_std'] == 'F').astype(int)
        recruiting['is_center'] = (recruiting['position_std'] == 'C').astype(int)
        
        # Parse physical attributes
        recruiting['height_inches'] = pd.to_numeric(recruiting['heightInches'], errors='coerce')
        recruiting['weight_pounds'] = pd.to_numeric(recruiting['weightPounds'], errors='coerce')
        
        # Create star tier flags
        recruiting['is_5star'] = (recruiting['stars'] == 5).astype(int)
        recruiting['is_4star'] = (recruiting['stars'] == 4).astype(int)
        recruiting['is_3star'] = (recruiting['stars'] == 3).astype(int)
        
        # Group by team and season
        recruiting_features = recruiting.groupby(['teamId', 'season']).agg({
            'stars': ['mean', 'std', 'max', 'min'],
            'rating': ['mean', 'std', 'max'],
            'ranking': ['mean', 'min'],  # Lower ranking is better
            'teamId': 'count',  # num_recruits
            'is_5star': 'sum',
            'is_4star': 'sum',
            'is_3star': 'sum',
            'is_guard': 'sum',
            'is_forward': 'sum',
            'is_center': 'sum',
            'height_inches': 'mean',
            'weight_pounds': 'mean',
            'hometownCountry': lambda x: (x == 'USA').sum(),
        }).reset_index()
        
        # Flatten column names
        recruiting_features.columns = [
            'teamId', 'season',
            'avg_recruit_stars', 'std_recruit_stars', 'max_recruit_stars', 'min_recruit_stars',
            'avg_recruit_rating', 'std_recruit_rating', 'max_recruit_rating',
            'avg_recruit_ranking', 'best_recruit_ranking',
            'num_recruits',
            'num_5star_recruits', 'num_4star_recruits', 'num_3star_recruits',
            'num_guard_recruits', 'num_forward_recruits', 'num_center_recruits',
            'avg_recruit_height', 'avg_recruit_weight',
            'num_us_recruits'
        ]
        
        # Derived features
        recruiting_features['pct_5star_recruits'] = (
            recruiting_features['num_5star_recruits'] / recruiting_features['num_recruits']
        )
        recruiting_features['pct_4star_plus'] = (
            (recruiting_features['num_5star_recruits'] + recruiting_features['num_4star_recruits']) / 
            recruiting_features['num_recruits']
        )
        recruiting_features['pct_guard_recruits'] = (
            recruiting_features['num_guard_recruits'] / recruiting_features['num_recruits']
        )
        recruiting_features['pct_us_recruits'] = (
            recruiting_features['num_us_recruits'] / recruiting_features['num_recruits']
        )
        
        return recruiting_features
    
    def _get_recruiting_trends(self, recruiting_features):
        """
        Calculate multi-year recruiting trends based on the aggregated features.
        No database query needed - operates on already-aggregated data.
        """
        print("  Calculating recruiting trends...")
        
        # Use a copy to avoid modifying the original df
        recruiting = recruiting_features.copy()
        
        # Sort by team and season
        recruiting = recruiting.sort_values(['teamId', 'season'])
        
        # Calculate rolling averages and trends
        # Use the column names from the recruiting_features DataFrame
        recruiting['avg_stars_L3'] = recruiting.groupby('teamId')['avg_recruit_stars'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
        recruiting['avg_rating_L3'] = recruiting.groupby('teamId')['avg_recruit_rating'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
        recruiting['num_recruits_L3'] = recruiting.groupby('teamId')['num_recruits'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
        
        # Calculate year-over-year change
        recruiting['recruit_stars_yoy_change'] = recruiting.groupby('teamId')['avg_recruit_stars'].diff()
        recruiting['recruit_rating_yoy_change'] = recruiting.groupby('teamId')['avg_recruit_rating'].diff()
        
        # Keep only trend columns
        trend_cols = [
            'teamId', 'season',
            'avg_stars_L3', 'avg_rating_L3', 'num_recruits_L3',
            'recruit_stars_yoy_change', 'recruit_rating_yoy_change'
        ]
        
        # Ensure we only return the columns that exist
        return recruiting[[col for col in trend_cols if col in recruiting.columns]]
    
    def _merge_all_features(self, df, roster_df, recruiting_df, recruiting_trends_df):
        """Efficiently merge all roster and recruiting features for home and away teams."""
        print("  Merging features to main dataframe...")
        
        # Combine all feature dataframes
        all_features = roster_df.merge(
            recruiting_df,
            on=['teamId', 'season'],
            how='outer'
        ).merge(
            recruiting_trends_df,
            on=['teamId', 'season'],
            how='outer'
        )
        
        # Get list of feature columns (exclude keys)
        feature_cols = [col for col in all_features.columns if col not in ['teamId', 'season']]
        
        # Merge for home team
        print("    Merging home team features...")
        home_features = all_features.copy()
        home_features.columns = ['homeTeamId', 'season'] + [f'home_{col}' for col in feature_cols]
        
        df = df.merge(
            home_features,
            on=['homeTeamId', 'season'],
            how='left'
        )
        
        # Merge for away team
        print("    Merging away team features...")
        away_features = all_features.copy()
        away_features.columns = ['awayTeamId', 'season'] + [f'away_{col}' for col in feature_cols]
        
        df = df.merge(
            away_features,
            on=['awayTeamId', 'season'],
            how='left'
        )
        
        return df