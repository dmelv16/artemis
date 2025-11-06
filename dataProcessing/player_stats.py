"""
Calculates player-level statistics and aggregates them to team level.
This is critical for capturing roster strength, star player impact, and depth.
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
        Main method to add all player-based features.
        """
        print("Calculating player statistics...")
        
        # Get all player game stats with team and date info
        player_games = self._get_player_games()
        
        # Add various player feature sets
        df = self._add_starter_stats(df, player_games, windows)
        df = self._add_bench_stats(df, player_games, windows)
        df = self._add_top_performer_stats(df, player_games, windows)
        df = self._add_roster_depth_metrics(df, player_games)
        df = self._add_key_player_availability(df, player_games)
        
        print(f"Added {len([c for c in df.columns if 'player_' in c or 'starter_' in c or 'bench_' in c])} player features")
        return df
    
    def _get_player_games(self):
        """Get all player game statistics."""
        query = """
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
        WHERE g.status = 'Final'
        ORDER BY g.startDate, pg.teamId, pg.minutes DESC
        """
        return self.db.query(query)
    
    def _add_starter_stats(self, df, player_games, windows):
        """
        Calculate rolling averages for starting lineup.
        Starters are key indicators of team strength.
        """
        print("  Processing starter statistics...")
        
        starters = player_games[player_games['starter'] == 1]
        
        for team_type in ['home', 'away']:
            team_id_col = f'{team_type}TeamId'
            
            for team_id in df[team_id_col].unique():
                if pd.isna(team_id):
                    continue
                
                team_starters = starters[starters['teamId'] == team_id].sort_values('startDate')
                
                if len(team_starters) == 0:
                    continue
                
                # Group by game and aggregate starter stats
                game_aggregates = team_starters.groupby(['gameId', 'startDate']).agg({
                    'points': ['sum', 'mean', 'max'],
                    'assists': ['sum', 'mean'],
                    'totalRebounds': ['sum', 'mean'],
                    'gameScore': ['mean', 'max'],
                    'usage': ['mean', 'max'],
                    'minutes': ['sum', 'mean'],
                    'offensiveRating': 'mean',
                    'defensiveRating': 'mean',
                    'effectiveFieldGoalPct': 'mean',
                    'trueShootingPct': 'mean'
                }).reset_index()
                
                # Flatten column names
                game_aggregates.columns = ['_'.join(col).strip('_') for col in game_aggregates.columns]
                
                # Calculate rolling averages for each window
                for window in windows:
                    for col in game_aggregates.columns:
                        if col not in ['gameId', 'startDate']:
                            if window == 'season':
                                game_aggregates[f'{col}_std'] = game_aggregates[col].expanding(min_periods=1).mean().shift(1)
                            else:
                                game_aggregates[f'{col}_L{window}'] = game_aggregates[col].rolling(
                                    window=window, min_periods=1
                                ).mean().shift(1)
                
                # Merge back to main dataframe
                self._merge_player_stats(df, game_aggregates, team_id, team_id_col, team_type, 'starter')
        
        return df
    
    def _add_bench_stats(self, df, player_games, windows):
        """
        Calculate bench strength metrics.
        Good bench depth is crucial for tournament success.
        """
        print("  Processing bench statistics...")
        
        bench = player_games[player_games['starter'] == 0]
        
        for team_type in ['home', 'away']:
            team_id_col = f'{team_type}TeamId'
            
            for team_id in df[team_id_col].unique():
                if pd.isna(team_id):
                    continue
                
                team_bench = bench[bench['teamId'] == team_id].sort_values('startDate')
                
                if len(team_bench) == 0:
                    continue
                
                # Aggregate bench contributions
                game_aggregates = team_bench.groupby(['gameId', 'startDate']).agg({
                    'points': 'sum',
                    'assists': 'sum',
                    'totalRebounds': 'sum',
                    'minutes': 'sum',
                    'gameScore': 'mean',
                    'athleteId': 'count'  # Number of bench players used
                }).reset_index()
                
                game_aggregates.rename(columns={'athleteId': 'bench_players_used'}, inplace=True)
                
                # Calculate bench scoring percentage
                total_team_points = self._get_team_points(team_id)
                game_aggregates = game_aggregates.merge(total_team_points, on='gameId', how='left')
                game_aggregates['bench_scoring_pct'] = game_aggregates['points'] / game_aggregates['team_points']
                
                # Rolling averages
                for window in windows:
                    for col in ['points', 'minutes', 'bench_players_used', 'bench_scoring_pct']:
                        if window == 'season':
                            game_aggregates[f'{col}_std'] = game_aggregates[col].expanding(min_periods=1).mean().shift(1)
                        else:
                            game_aggregates[f'{col}_L{window}'] = game_aggregates[col].rolling(
                                window=window, min_periods=1
                            ).mean().shift(1)
                
                self._merge_player_stats(df, game_aggregates, team_id, team_id_col, team_type, 'bench')
        
        return df
    
    def _add_top_performer_stats(self, df, player_games, windows):
        """
        Track the team's best players' performance.
        Star players often determine game outcomes.
        """
        print("  Processing top performer statistics...")
        
        for team_type in ['home', 'away']:
            team_id_col = f'{team_type}TeamId'
            
            for team_id in df[team_id_col].unique():
                if pd.isna(team_id):
                    continue
                
                team_players = player_games[player_games['teamId'] == team_id].sort_values('startDate')
                
                if len(team_players) == 0:
                    continue
                
                # Identify top 3 players by average gameScore for the season
                top_players = team_players.groupby('athleteId')['gameScore'].mean().nlargest(3).index
                
                # Get stats for top players
                top_player_games = team_players[team_players['athleteId'].isin(top_players)]
                
                # Aggregate top player stats by game
                game_aggregates = top_player_games.groupby(['gameId', 'startDate']).agg({
                    'points': ['sum', 'max'],
                    'assists': ['sum', 'max'],
                    'gameScore': ['mean', 'max'],
                    'usage': 'max',
                    'minutes': 'sum'
                }).reset_index()
                
                # Flatten columns
                game_aggregates.columns = ['_'.join(col).strip('_') for col in game_aggregates.columns]
                
                # Rolling averages
                for window in windows:
                    for col in game_aggregates.columns:
                        if col not in ['gameId', 'startDate']:
                            if window == 'season':
                                game_aggregates[f'{col}_std'] = game_aggregates[col].expanding(min_periods=1).mean().shift(1)
                            else:
                                game_aggregates[f'{col}_L{window}'] = game_aggregates[col].rolling(
                                    window=window, min_periods=1
                                ).mean().shift(1)
                
                self._merge_player_stats(df, game_aggregates, team_id, team_id_col, team_type, 'top3')
        
        return df
    
    def _add_roster_depth_metrics(self, df, player_games):
        """
        Calculate roster depth and rotation metrics.
        Teams with deeper rotations handle fatigue better.
        """
        print("  Processing roster depth metrics...")
        
        depth_metrics = []
        
        for idx, game in df.iterrows():
            home_id = game['homeTeamId']
            away_id = game['awayTeamId']
            game_date = game['startDate']
            
            # Calculate for each team
            for team_id, team_type in [(home_id, 'home'), (away_id, 'away')]:
                # Get recent games for this team
                recent_games = player_games[
                    (player_games['teamId'] == team_id) & 
                    (player_games['startDate'] < game_date)
                ].tail(50)  # Last ~10 games worth of player data
                
                if len(recent_games) > 0:
                    # Calculate depth metrics
                    games_by_player = recent_games.groupby('athleteId').agg({
                        'minutes': 'mean',
                        'gameScore': 'mean',
                        'gameId': 'count'
                    })
                    
                    # Players averaging 10+ minutes
                    rotation_size = len(games_by_player[games_by_player['minutes'] > 10])
                    
                    # Minutes concentration (how much do top players play)
                    top5_minutes = games_by_player.nlargest(5, 'minutes')['minutes'].sum()
                    total_minutes = games_by_player['minutes'].sum()
                    minutes_concentration = top5_minutes / total_minutes if total_minutes > 0 else 0
                    
                    # Quality depth (players with positive gameScore)
                    quality_depth = len(games_by_player[games_by_player['gameScore'] > 5])
                    
                    depth_metrics.append({
                        'gameId': game['gameId'],
                        f'{team_type}_rotation_size': rotation_size,
                        f'{team_type}_minutes_concentration': minutes_concentration,
                        f'{team_type}_quality_depth': quality_depth
                    })
                else:
                    depth_metrics.append({
                        'gameId': game['gameId'],
                        f'{team_type}_rotation_size': 8,
                        f'{team_type}_minutes_concentration': 0.5,
                        f'{team_type}_quality_depth': 5
                    })
        
        depth_df = pd.DataFrame(depth_metrics)
        # Remove duplicate gameId columns
        depth_df = depth_df.groupby('gameId').first().reset_index()
        df = df.merge(depth_df, on='gameId', how='left')
        
        return df
    
    def _add_key_player_availability(self, df, player_games):
        """
        Track if key players are available/injured.
        Missing key players significantly impacts performance.
        """
        print("  Processing key player availability...")
        
        availability_features = []
        
        for idx, game in df.iterrows():
            features = {'gameId': game['gameId']}
            
            for team_id, team_type in [(game['homeTeamId'], 'home'), (game['awayTeamId'], 'away')]:
                # Get this team's recent games
                recent = player_games[
                    (player_games['teamId'] == team_id) & 
                    (player_games['startDate'] < game['startDate'])
                ].sort_values('startDate')
                
                if len(recent) > 0:
                    # Identify key players (top 5 by average minutes in last 10 games)
                    last_10_games = recent['gameId'].unique()[-10:]
                    recent_10 = recent[recent['gameId'].isin(last_10_games)]
                    
                    key_players = recent_10.groupby('athleteId')['minutes'].mean().nlargest(5).index
                    
                    # Check if they played in the most recent game
                    last_game = recent['gameId'].unique()[-1]
                    last_game_players = recent[recent['gameId'] == last_game]['athleteId'].values
                    
                    key_players_available = sum([p in last_game_players for p in key_players])
                    key_player_availability_pct = key_players_available / 5
                    
                    # Get average minutes of missing key players
                    missing_players = [p for p in key_players if p not in last_game_players]
                    if missing_players:
                        missing_minutes = recent_10[recent_10['athleteId'].isin(missing_players)]['minutes'].mean()
                    else:
                        missing_minutes = 0
                    
                    features[f'{team_type}_key_players_available'] = key_players_available
                    features[f'{team_type}_key_player_availability_pct'] = key_player_availability_pct
                    features[f'{team_type}_missing_player_minutes'] = missing_minutes
                else:
                    features[f'{team_type}_key_players_available'] = 5
                    features[f'{team_type}_key_player_availability_pct'] = 1.0
                    features[f'{team_type}_missing_player_minutes'] = 0
            
            availability_features.append(features)
        
        availability_df = pd.DataFrame(availability_features)
        df = df.merge(availability_df, on='gameId', how='left')
        
        return df
    
    def _merge_player_stats(self, df, player_stats, team_id, team_id_col, team_type, prefix):
        """Helper to merge player stats back to main dataframe."""
        for idx, game in df[df[team_id_col] == team_id].iterrows():
            game_date = game['startDate']
            prior_games = player_stats[player_stats['startDate'] < game_date]
            
            if len(prior_games) > 0:
                latest_stats = prior_games.iloc[-1]
                
                for col in latest_stats.index:
                    if col not in ['gameId', 'startDate'] and ('_std' in col or '_L' in col):
                        feature_name = f'{team_type}_{prefix}_{col}'
                        df.at[idx, feature_name] = latest_stats[col]
    
    def _get_team_points(self, team_id):
        """Get total team points for each game."""
        query = f"""
        SELECT 
            gameId,
            points as team_points
        FROM team_games
        WHERE teamId = {team_id}
        """
        return self.db.query(query)