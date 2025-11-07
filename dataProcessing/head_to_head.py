"""
Calculates head-to-head historical features.
"""
import pandas as pd
import numpy as np
import logging

class HeadToHeadProcessor:
    def __init__(self, db_conn):
        self.db = db_conn
        
    def add_features(self, df):
        """
        Adds head-to-head features to the dataframe.
        Uses vectorized approach to avoid N+1 query problem.
        Pre-groups history by matchup for O(1) lookups instead of O(n) scans.
        """
        print("Adding head-to-head historical features...")
        
        # Fetch ALL historical games at once (single query)
        history_query = """
        SELECT g.gameId, g.homePoints, g.awayPoints, 
               g.homeTeamId, g.awayTeamId, g.startDate
        FROM games g
        WHERE g.status = 'Final'
        ORDER BY g.startDate DESC
        """
        
        try:
            all_history = self.db.query(history_query)
            
            if len(all_history) == 0:
                logging.warning("No historical games found in database")
                return self._add_default_features(df)
            
            # Pre-process: Create a matchup key and group history by it
            history_grouped = self._preprocess_history(all_history)
            
            # Calculate H2H features for all games at once
            h2h_features = self._calculate_all_h2h_features(df, history_grouped)
            
            # Merge back to main dataframe
            df = df.merge(h2h_features, on='gameId', how='left')
            
            # Fill any missing values with defaults
            df['h2h_games_played'] = df['h2h_games_played'].fillna(0).astype(int)
            df['h2h_home_win_pct'] = df['h2h_home_win_pct'].fillna(0.5)
            df['h2h_avg_margin'] = df['h2h_avg_margin'].fillna(np.nan)
            df['h2h_last_margin'] = df['h2h_last_margin'].fillna(np.nan)
            
            print(f"Added head-to-head features for {len(df)} games")
            return df
            
        except Exception as e:
            logging.error(f"Failed to add H2H features: {e}")
            return self._add_default_features(df)
    
    def _preprocess_history(self, history):
        """
        Pre-process history to enable O(1) matchup lookups.
        Creates a normalized matchup key (smaller_id, larger_id) and groups by it.
        """
        # Create normalized matchup key (always put smaller team ID first)
        history['team1'] = history[['homeTeamId', 'awayTeamId']].min(axis=1)
        history['team2'] = history[['homeTeamId', 'awayTeamId']].max(axis=1)
        history['matchup_key'] = history['team1'].astype(str) + '_' + history['team2'].astype(str)
        
        # Group by matchup and store in dictionary for O(1) lookup
        grouped = {}
        for matchup_key, group in history.groupby('matchup_key'):
            # Sort by date descending (most recent first) for each matchup
            grouped[matchup_key] = group.sort_values('startDate', ascending=False)
        
        return grouped
    
    def _calculate_all_h2h_features(self, df, history_grouped):
        """
        Vectorized calculation of H2H features for all games.
        Uses pre-grouped history for O(1) matchup lookups.
        """
        results = []
        
        for idx, game in df.iterrows():
            h2h_data = self._get_h2h_stats_from_grouped_history(
                game['homeTeamId'],
                game['awayTeamId'],
                game['startDate'],
                history_grouped
            )
            h2h_data['gameId'] = game['gameId']
            results.append(h2h_data)
        
        return pd.DataFrame(results)
    
    def _get_h2h_stats_from_grouped_history(self, home_id, away_id, game_date, history_grouped):
        """
        Extract H2H stats from pre-grouped history dictionary.
        O(1) lookup instead of O(n) scan.
        """
        # Create the same normalized matchup key
        team1 = min(home_id, away_id)
        team2 = max(home_id, away_id)
        matchup_key = f"{team1}_{team2}"
        
        # O(1) lookup in dictionary
        if matchup_key not in history_grouped:
            return {
                'h2h_games_played': 0,
                'h2h_home_win_pct': 0.5,
                'h2h_avg_margin': np.nan,
                'h2h_last_margin': np.nan
            }
        
        # Get all games for this matchup, then filter by date
        matchup_history = history_grouped[matchup_key]
        h2h_games = matchup_history[matchup_history['startDate'] < game_date]
        
        if len(h2h_games) > 0:
            return self._calculate_h2h_metrics(h2h_games, home_id)
        else:
            return {
                'h2h_games_played': 0,
                'h2h_home_win_pct': 0.5,
                'h2h_avg_margin': np.nan,
                'h2h_last_margin': np.nan
            }
    
    def _calculate_h2h_metrics(self, games, home_id):
        """
        Calculate H2H metrics from a set of historical games.
        Correctly handles margin calculation regardless of which team was home.
        """
        home_wins = 0
        total_margin = 0
        
        for _, game in games.iterrows():
            if game['homeTeamId'] == home_id:
                # Current home team was home in this historical game
                margin = game['homePoints'] - game['awayPoints']
                if margin > 0:
                    home_wins += 1
            else:
                # Current home team was away in this historical game
                margin = game['awayPoints'] - game['homePoints']
                if margin > 0:
                    home_wins += 1
            
            total_margin += margin
        
        # Fix for last_margin: must check which team was home
        last_game = games.iloc[0]
        if last_game['homeTeamId'] == home_id:
            last_margin = last_game['homePoints'] - last_game['awayPoints']
        else:
            # Current home team was away in the last game
            last_margin = last_game['awayPoints'] - last_game['homePoints']
        
        return {
            'h2h_games_played': len(games),
            'h2h_home_win_pct': home_wins / len(games),
            'h2h_avg_margin': total_margin / len(games),
            'h2h_last_margin': last_margin
        }
    
    def _add_default_features(self, df):
        """
        Add default H2H features when no history is available.
        """
        df['h2h_games_played'] = 0
        df['h2h_home_win_pct'] = 0.5
        df['h2h_avg_margin'] = np.nan
        df['h2h_last_margin'] = np.nan
        return df


class HeadToHeadProcessorParameterized(HeadToHeadProcessor):
    """
    Alternative implementation using parameterized queries (row-by-row).
    Use this if your dataset is small or if you can't load all history at once.
    Still much better than the original due to SQL injection protection.
    """
    
    def add_features(self, df):
        """
        Adds H2H features using parameterized queries (one per game).
        Slower than vectorized approach but protects against SQL injection.
        """
        print("Adding head-to-head historical features (parameterized)...")
        h2h_features = []
        
        for idx, game in df.iterrows():
            h2h_data = self._get_h2h_stats_parameterized(
                game['homeTeamId'], 
                game['awayTeamId'], 
                game['startDate']
            )
            h2h_data['gameId'] = game['gameId']
            h2h_features.append(h2h_data)
        
        h2h_df = pd.DataFrame(h2h_features)
        df = df.merge(h2h_df, on='gameId', how='left')
        
        print("Added head-to-head features")
        return df
    
    def _get_h2h_stats_parameterized(self, home_id, away_id, game_date):
        """
        Get H2H stats using parameterized query (SQL injection safe).
        """
        # Use '?' placeholders (adjust to '%s' or ':name' for your DB library)
        query = """
        SELECT g.homePoints, g.awayPoints, g.homeTeamId, g.awayTeamId
        FROM games g
        WHERE ((g.homeTeamId = ? AND g.awayTeamId = ?)
               OR (g.homeTeamId = ? AND g.awayTeamId = ?))
            AND g.startDate < ?
            AND g.status = 'Final'
        ORDER BY g.startDate DESC
        """
        
        params = [home_id, away_id, away_id, home_id, game_date]
        
        try:
            h2h_games = self.db.query(query, params)
            
            if len(h2h_games) > 0:
                stats = self._calculate_h2h_metrics(h2h_games, home_id)
                return stats
                
        except Exception as e:
            logging.error(f"H2H query failed for {home_id} vs {away_id} on {game_date}: {e}")
        
        return {
            'h2h_games_played': 0,
            'h2h_home_win_pct': 0.5,
            'h2h_avg_margin': np.nan,
            'h2h_last_margin': np.nan
        }