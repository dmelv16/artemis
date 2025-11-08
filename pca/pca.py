import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RollingWindowClusterPipeline:
    """
    Implements a rolling window approach for training and applying PCA/cluster models.
    Updated for team-by-team data format with team_* and opp_* prefixes.
    """
    
    def __init__(self, data_path, output_dir='rolling_clusters', 
                 min_training_days=365, retrain_interval_days=15):
        """
        Args:
            data_path: Path to the team-by-team parquet file
            output_dir: Directory to save models and results
            min_training_days: Minimum days of historical data needed before making predictions
            retrain_interval_days: How often to retrain the global models (default 15 days)
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.min_training_days = min_training_days
        self.retrain_interval_days = retrain_interval_days
        
        # Track cluster evolution over time
        self.cluster_evolution = {}
        
        # Define feature groups for team-by-team format
        self.feature_groups = self._define_feature_groups()
        
    def _define_feature_groups(self):
        """
        Define feature groups for clustering - comprehensive coverage
        These groups capture all aspects of team performance and game context
        """
        return {
            # Team offensive capabilities
            'team_offensive_efficiency': {
                'patterns': ['team_points_L', 'team_fieldGoalsPct_L', 'team_effectiveFieldGoalPct_L', 
                           'team_trueShootingPct_L', 'team_assists_L', 'team_freeThrowRate_L',
                           'team_oppTurnoverRatio_L'],  # Added: forcing turnovers
                'n_components': 4,
                'n_clusters': 8
            },
            
            # Team defensive capabilities
            'team_defensive_strength': {
                'patterns': ['team_oppPoints_L', 'team_oppFieldGoalsPct_L', 'team_oppEffectiveFieldGoalPct_L',
                           'team_steals_L', 'team_blocks_L', 'team_oppRating_L',
                           'team_fouls_L', 'team_oppFreeThrowRate_L'],  # Added: fouls and FT defense
                'n_components': 5,
                'n_clusters': 8
            },
            
            # Free throw performance (new group)
            'team_free_throw_profile': {
                'patterns': ['team_freeThrowsPct_L', 'team_freeThrowRate_L', 
                           'team_oppFreeThrowsPct_L', 'team_oppFreeThrowRate_L'],
                'n_components': 3,
                'n_clusters': 6
            },
            
            # Turnover differential (new group)
            'team_turnover_battle': {
                'patterns': ['team_turnovers_L', 'team_turnoverRatio_L', 
                           'team_oppTurnoverRatio_L'],  # Creating vs giving up turnovers
                'n_components': 2,
                'n_clusters': 6
            },
            
            # Discipline and fouls (new group)
            'team_discipline': {
                'patterns': ['team_fouls_L', 'team_oppFreeThrowRate_L', 
                           'team_freeThrowRate_L'],  # How often they foul vs draw fouls
                'n_components': 2,
                'n_clusters': 5
            },
            
            # Pace and tempo control
            'team_pace_tempo': {
                'patterns': ['team_pace_L', 'team_possessions_L', 'team_rating_L', 'team_oppRating_L'],
                'n_components': 3,
                'n_clusters': 6
            },
            
            # Rebounding and hustle stats
            'team_rebounding': {
                'patterns': ['team_offensiveReboundPct_L', 'team_oppOffensiveReboundPct_L', 
                           'team_totalRebounds_L', 'team_turnovers_L', 'team_turnoverRatio_L'],
                'n_components': 3,
                'n_clusters': 6
            },
            
            # Three-point shooting profile
            'team_three_point': {
                'patterns': ['team_threePointFGPct_L', 'team_oppThreePointFGPct_L', 
                           'team_threePointFGMade_L', 'team_threePointFGAttempted_L'],
                'n_components': 3,
                'n_clusters': 6
            },
            
            # Starter performance
            'team_starters': {
                'patterns': ['team_starter_points_', 'team_starter_assists_', 'team_starter_totalRebounds_',
                           'team_starter_gameScore_', 'team_starter_usage_', 'team_starter_netRating_',
                           'team_starter_effectiveFieldGoalPct_', 'team_starter_trueShootingPct_',
                           'team_starter_offensiveRating_mean_', 'team_starter_defensiveRating_mean_'],  # Added: off/def ratings
                'n_components': 5,
                'n_clusters': 8
            },
            
            # Star concentration (new group)
            'team_usage_concentration': {
                'patterns': ['team_starter_usage_max_', 'team_top3_usage_max_',
                           'team_starter_gameScore_std_', 'team_minutes_concentration'],  # How balanced is scoring/usage?
                'n_components': 3,
                'n_clusters': 6
            },
            
            # Bench depth and contribution
            'team_bench_depth': {
                'patterns': ['team_bench_points_L', 'team_bench_assists_L', 'team_bench_gameScore_L',
                           'team_bench_netRating_L', 'team_bench_bench_scoring_pct_L',
                           'team_bench_effectiveFieldGoalPct_L', 'team_bench_players_used'],  # Added: bench size
                'n_components': 4,
                'n_clusters': 6
            },
            
            # Top 3 players (star power)
            'team_star_power': {
                'patterns': ['team_top3_points_', 'team_top3_assists_', 'team_top3_usage_',
                           'team_top3_gameScore_', 'team_top3_effectiveFieldGoalPct_',
                           'team_top3_netRating_'],
                'n_components': 4,
                'n_clusters': 7
            },
            
            # Position-specific: Guards
            'team_guard_production': {
                'patterns': ['team_guard_points_', 'team_guard_assists_', 'team_guard_usage_',
                           'team_guard_netRating_', 'team_guard_twoPointFGPct_'],
                'n_components': 3,
                'n_clusters': 6
            },
            
            # Position-specific: Forwards
            'team_forward_production': {
                'patterns': ['team_forward_points_', 'team_forward_totalRebounds_',
                           'team_forward_netRating_', 'team_forward_offensiveReboundPct_'],
                'n_components': 3,
                'n_clusters': 6
            },
            
            # Position-specific: Centers
            'team_center_production': {
                'patterns': ['team_center_points_', 'team_center_totalRebounds_',
                           'team_center_twoPointFGPct_', 'team_center_offensiveReboundPct_'],
                'n_components': 3,
                'n_clusters': 5
            },
            
            # Rotation and depth management
            'team_rotation': {
                'patterns': ['team_rotation_size', 'team_minutes_concentration', 'team_quality_depth',
                           'team_key_minutes_availability_pct'],
                'n_components': 2,
                'n_clusters': 5
            },
            
            # Roster composition
            'team_roster_profile': {
                'patterns': ['team_avg_height', 'team_avg_weight', 'team_avg_experience',
                           'team_pct_freshmen', 'team_pct_seniors', 'team_pct_guards',
                           'team_pct_forwards', 'team_avg_recruit_rating',
                           'team_height_range', 'team_weight_range'],  # Added: physical variance
                'n_components': 5,
                'n_clusters': 7
            },
            
            # Recruiting quality (new group)
            'team_recruiting_strength': {
                'patterns': ['team_avg_recruit_rating', 'team_max_recruit_rating', 
                           'team_best_recruit_ranking', 'team_pct_4star_plus',
                           'team_num_4star_recruits', 'team_num_5star_recruits',
                           'team_recruit_rating_yoy_change', 'team_recruit_stars_yoy_change'],
                'n_components': 4,
                'n_clusters': 7
            },
            
            # Experience and returning talent (new group)
            'team_experience_continuity': {
                'patterns': ['team_avg_experience', 'team_pct_returning', 
                           'team_returning_players', 'team_pct_freshmen',
                           'team_pct_seniors', 'team_num_juniors', 'team_num_sophomores'],
                'n_components': 4,
                'n_clusters': 6
            },
            
            # Recent form and momentum (L5 only)
            'team_recent_form': {
                'patterns': ['team_points_L5', 'team_rating_L5', 'team_fieldGoalsPct_L5',
                           'team_effectiveFieldGoalPct_L5', 'team_assists_L5',
                           'team_oppPoints_L5'],  # Added: recent defense
                'n_components': 3,
                'n_clusters': 7
            },
            
            # Opponent strength (what they've faced)
            'opponent_quality': {
                'patterns': ['opp_rating_L', 'opp_points_L', 'opp_fieldGoalsPct_L',
                           'opp_effectiveFieldGoalPct_L', 'opp_pace_L'],
                'n_components': 3,
                'n_clusters': 6
            },
            
            # Matchup differentials - expanded
            'matchup_advantages': {
                'patterns': ['diff_rating_L', 'diff_points_L', 'diff_fieldGoalsPct_L',
                           'diff_pace_L', 'diff_effectiveFieldGoalPct_L',
                           'diff_offensiveReboundPct_L', 'diff_steals_L', 'diff_blocks_L',
                           'diff_turnoverRatio_L'],  # Added: rebounding, defense, turnovers
                'n_components': 4,
                'n_clusters': 8
            },
            
            # Matchup - offensive comparison (new group)
            'matchup_offensive_edge': {
                'patterns': ['diff_assists_L', 'diff_freeThrowRate_L', 
                           'diff_threePointFGPct_L', 'diff_effectiveFieldGoalPct_L',
                           'diff_freeThrowsPct_L'],
                'n_components': 3,
                'n_clusters': 7
            },
            
            # Matchup - defensive comparison (new group)
            'matchup_defensive_edge': {
                'patterns': ['diff_oppFieldGoalsPct_L', 'diff_oppEffectiveFieldGoalPct_L',
                           'diff_oppThreePointFGPct_L', 'diff_oppTurnoverRatio_L',
                           'diff_oppOffensiveReboundPct_L'],
                'n_components': 3,
                'n_clusters': 7
            },
            
            # Matchup - star power comparison (new group)
            'matchup_star_differential': {
                'patterns': ['diff_starter_points_', 'diff_top3_points_',
                           'diff_starter_assists_', 'diff_top3_assists_',
                           'diff_bench_points_'],
                'n_components': 3,
                'n_clusters': 7
            },
            
            # Matchup - position battles (new group)
            'matchup_position_advantages': {
                'patterns': ['diff_guard_points_', 'diff_forward_points_', 'diff_center_points_',
                           'diff_guard_assists_', 'diff_forward_totalRebounds_'],
                'n_components': 3,
                'n_clusters': 6
            },
            
            # Matchup - recruiting differential (new group)
            'matchup_talent_gap': {
                'patterns': ['diff_avg_recruit_rating', 'diff_max_recruit_rating',
                           'diff_AP_poll_points', 'diff_Coaches_poll_points'],
                'n_components': 2,
                'n_clusters': 6
            },
            
            # Rankings and prestige (new group)
            'team_rankings_prestige': {
                'patterns': ['team_AP_poll_points', 'team_AP_rank', 'team_AP_ranked', 'team_AP_tier',
                           'team_Coaches_poll_points', 'team_Coaches_rank', 'team_Coaches_ranked', 'team_Coaches_tier'],
                'n_components': 3,
                'n_clusters': 7
            },
            
            # Head-to-head history (new group)
            'h2h_history': {
                'patterns': ['h2h_games_played', 'h2h_home_win_pct', 
                           'h2h_avg_margin', 'h2h_last_margin'],
                'n_components': 2,
                'n_clusters': 5
            },
            
            # Rest and schedule context (new group)
            'rest_and_fatigue': {
                'patterns': ['rest_days', 'is_b2b', 'rest_days_diff', 'b2b_diff',
                           'opp_rest_days', 'opp_is_b2b'],
                'n_components': 3,
                'n_clusters': 6
            },
            
            # Momentum and streaks (new group)
            'momentum': {
                'patterns': ['streak', 'opp_streak', 'streak_diff'],
                'n_components': 2,
                'n_clusters': 7
            },
            
            # Game context (new group)
            'game_context': {
                'patterns': ['is_home', 'is_neutral_site', 'is_conference_game',
                           'neutralSite', 'conferenceGame', 'is_weekend',
                           'day_of_week', 'month', 'days_since_season_start'],
                'n_components': 3,
                'n_clusters': 6
            },
            
            # Betting market signals (new group)
            'betting_market': {
                'patterns': ['spread', 'overUnder', 'spread_movement', 'total_movement',
                           'homeMoneyline', 'awayMoneyline', 'home_implied_prob', 'away_implied_prob',
                           'line_movement_magnitude', 'significant_line_move',
                           'favorite_size', 'is_pick_em'],
                'n_components': 5,
                'n_clusters': 8
            },
            
            # Expected vs actual betting (new group)
            'betting_expectations': {
                'patterns': ['home_true_prob', 'away_true_prob', 'team_implied_prob', 'team_true_prob',
                           'expected_home_margin', 'expected_away_margin',
                           'implied_home_score', 'implied_away_score'],
                'n_components': 4,
                'n_clusters': 7
            },
            
            # Game favorites and expectations (new group)
            'favorite_analysis': {
                'patterns': ['team_favorite', 'opp_favorite', 'home_favorite',
                           'favorite_size', 'is_pick_em'],
                'n_components': 2,
                'n_clusters': 5
            },
            
            # Season timing (new group)
            'season_phase': {
                'patterns': ['days_since_season_start', 'month', 'seasonType', 'tournament'],
                'n_components': 2,
                'n_clusters': 6
            },
            
            # Venue and location (new group)
            'venue_context': {
                'patterns': ['is_home', 'is_neutral_site', 'venueId'],  # venueId for venue-specific effects
                'n_components': 2,
                'n_clusters': 5
            },
            
            # Conference matchup type (new group)
            'conference_context': {
                'patterns': ['is_conference_game', 'conferenceId', 'opp_conferenceId'],
                'n_components': 2,
                'n_clusters': 6
            },
            
            # Poll rankings matchup (new group)
            'rankings_matchup': {
                'patterns': ['home_AP_ranked', 'away_AP_ranked', 'AP_rank_diff', 'AP_matchup_type',
                           'home_Coaches_ranked', 'away_Coaches_ranked', 'Coaches_rank_diff', 'Coaches_matchup_type',
                           'AP_points_diff', 'Coaches_points_diff'],
                'n_components': 4,
                'n_clusters': 7
            },
        }
    
    def load_data(self):
        """Load and prepare team-by-team data"""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_parquet(self.data_path)
        
        # Ensure date column
        if 'startDate' not in df.columns:
            raise ValueError("startDate column not found in data")
        
        df['startDate'] = pd.to_datetime(df['startDate'])
        df = df.sort_values('startDate')
        
        print(f"Loaded {len(df)} team-game records")
        print(f"Date range: {df['startDate'].min()} to {df['startDate'].max()}")
        print(f"Number of unique teams: {df['team'].nunique() if 'team' in df.columns else 'N/A'}")
        
        return df
    
    def get_relevant_columns(self, df, patterns):
        """
        Get columns matching any of the patterns
        Updated to work with team_* and opp_* prefixes
        """
        cols = []
        for pattern in patterns:
            # Find columns that contain the pattern
            matching = [c for c in df.columns if pattern in c]
            cols.extend(matching)
        
        # Remove duplicates and filter out _season columns
        cols = list(set(cols))
        cols = [c for c in cols if not c.endswith('_season')]
        
        return cols
    
    def fit_models_for_date(self, train_data, feature_group, group_config):
        """
        Fit PCA and KMeans models on training data up to a specific date
        """
        cols = self.get_relevant_columns(train_data, group_config['patterns'])
        
        if not cols:
            print(f"  Warning: No columns found for {feature_group}")
            return None
        
        # Select only numeric columns
        numeric_cols = train_data[cols].select_dtypes(include=np.number).columns
        
        if not list(numeric_cols):
            print(f"  Warning: No NUMERIC columns found for {feature_group}")
            return None

        if len(train_data) < 100:
            print(f"  Warning: Not enough training data for {feature_group} ({len(train_data)} rows)")
            return None
        
        # Prepare data - use only numeric_cols
        data = train_data[numeric_cols].dropna(thresh=len(numeric_cols)*0.5)
        
        if len(data) < 50:
            print(f"  Warning: Too many missing values for {feature_group} (using {len(numeric_cols)} numeric features)")
            return None
        
        # Fill remaining NaNs with median
        median_vals = data.median()
        median_vals = median_vals.fillna(0)
        data_filled = data.fillna(median_vals)
        
        # --- START NEW FIX ---
        # Check for columns with zero variance (e.g., all 0s)
        # These will cause StandardScaler to produce NaNs
        variance = data_filled.var()
        constant_cols = variance[variance == 0].index
        
        if len(constant_cols) > 0:
            print(f"    Warning: Removing {len(constant_cols)} constant column(s) from {feature_group} (zero variance)")
            for col in constant_cols:
                print(f"      - {col}")
                
            # Drop them from the dataframe
            data_filled = data_filled.drop(columns=constant_cols)
            
            # Update numeric_cols to reflect the removed columns
            # This is critical for the 'apply_models' function
            numeric_cols = data_filled.columns
            
            # If we removed all columns, we can't proceed
            if len(numeric_cols) == 0:
                print(f"    Warning: No non-constant features left for {feature_group}")
                return None
        # --- END NEW FIX ---
        
        # Fit StandardScaler (now safe)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_filled)
        
        # Fit PCA
        n_features = len(numeric_cols)
        n_samples = len(data_filled)
        n_components = min(group_config['n_components'], n_features, n_samples - 1)
        
        if n_components <= 0:
             print(f"  Warning: Not enough samples or features for PCA in {feature_group}")
             return None

        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(data_scaled)
        
        # Fit KMeans
        n_clusters = min(group_config['n_clusters'], n_samples)
        if n_clusters <= 0:
            print(f"  Warning: Not enough samples for KMeans in {feature_group}")
            return None
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(pca_features)
        
        print(f"  {feature_group}: {len(numeric_cols)} features -> {n_components} PCs -> {n_clusters} clusters")
        print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        # IMPORTANT: Filter median_vals to only include columns we actually used
        final_median_vals = median_vals[numeric_cols]
        
        return {
            'scaler': scaler,
            'pca': pca,
            'kmeans': kmeans,
            'columns': numeric_cols, # This list is now clean
            'median_vals': final_median_vals, # This median dict now matches the columns
            'cluster_centers': kmeans.cluster_centers_,
            'explained_variance': pca.explained_variance_ratio_,
            'n_components': n_components,
            'n_clusters': n_clusters
        }
    
    def map_cluster_evolution(self, old_centers, new_centers):
        """
        Map old cluster IDs to new cluster IDs based on center similarity.
        This maintains cluster continuity across retraining.
        """
        if old_centers is None or len(old_centers) == 0:
            return {i: i for i in range(len(new_centers))}
        
        # Handle dimension mismatch
        if old_centers.shape[1] != new_centers.shape[1]:
            print(f"    Warning: Dimension mismatch in cluster centers, resetting mapping")
            return {i: i for i in range(len(new_centers))}
        
        # Calculate cosine similarity between old and new centers
        similarity = cosine_similarity(old_centers, new_centers)
        
        # Greedy matching to maintain cluster consistency
        mapping = {}
        used_new = set()
        
        for old_idx in range(len(old_centers)):
            similarities = similarity[old_idx].copy()
            for new_idx in used_new:
                similarities[new_idx] = -1
            
            best_new = np.argmax(similarities)
            mapping[old_idx] = best_new
            used_new.add(best_new)
        
        # Map any remaining new clusters
        for new_idx in range(len(new_centers)):
            if new_idx not in used_new:
                mapping[len(mapping)] = new_idx
        
        return mapping
    
    def apply_models(self, data, models, feature_group):
        """Apply fitted models to transform new data"""
        if models is None:
            return None
        
        cols = models['columns']
        
        # Ensure all required columns exist
        missing_cols = set(cols) - set(data.columns)
        if missing_cols:
            print(f"  Warning: {len(missing_cols)} columns missing for {feature_group}")
            for col in missing_cols:
                data[col] = np.nan
        
        data_subset = data[cols].copy()
        data_filled = data_subset.fillna(models['median_vals'])
        
        # Transform
        try:
            data_scaled = models['scaler'].transform(data_filled)
            pca_features = models['pca'].transform(data_scaled)
            clusters = models['kmeans'].predict(pca_features)
            
            return {
                'pca_features': pca_features,
                'clusters': clusters,
                'index': data.index
            }
        except Exception as e:
            print(f"  Error applying models for {feature_group}: {e}")
            return None
    
    def create_rolling_features(self, df):
        """
        Main pipeline: Create features using rolling window approach
        No data leakage - models only trained on past data
        """
        results = []
        model_history = {}
        last_retrain_date = None
        
        # Get unique dates
        dates = df['startDate'].unique()
        dates = pd.Series(dates).sort_values().values
        
        print(f"\nProcessing {len(dates)} unique game dates...")
        print(f"Retraining every {self.retrain_interval_days} days")
        print(f"Minimum training period: {self.min_training_days} days\n")
        
        for date_idx, current_date in enumerate(dates):
            
            # Skip if we don't have enough training data
            earliest_date = df['startDate'].min()
            days_of_data = (pd.Timestamp(current_date) - pd.Timestamp(earliest_date)).days
            
            if days_of_data < self.min_training_days:
                continue
            
            # Determine if we need to retrain
            needs_retrain = (
                last_retrain_date is None or 
                (pd.Timestamp(current_date) - pd.Timestamp(last_retrain_date)).days >= self.retrain_interval_days
            )
            
            if needs_retrain:
                print(f"\n{'='*70}")
                print(f"RETRAINING models for {pd.Timestamp(current_date).strftime('%Y-%m-%d')}")
                print(f"{'='*70}")
                
                # Get all data BEFORE current date for training (no data leakage)
                train_mask = df['startDate'] < current_date
                train_data = df[train_mask].copy()
                
                print(f"Training on {len(train_data)} team-game records")
                
                # Store previous models for cluster mapping
                prev_models = model_history.copy()
                
                # Fit new models for each feature group
                current_models = {}
                for group_name, group_config in self.feature_groups.items():
                    models = self.fit_models_for_date(train_data, group_name, group_config)
                    
                    if models is not None:
                        # Map cluster evolution if we have previous models
                        if group_name in prev_models and prev_models[group_name] is not None:
                            old_centers = prev_models[group_name]['cluster_centers']
                            new_centers = models['cluster_centers']
                            
                            cluster_mapping = self.map_cluster_evolution(old_centers, new_centers)
                            models['cluster_mapping'] = cluster_mapping
                            
                            # Track evolution
                            if group_name not in self.cluster_evolution:
                                self.cluster_evolution[group_name] = []
                            self.cluster_evolution[group_name].append({
                                'date': str(current_date),
                                'mapping': cluster_mapping,
                                'n_clusters': models['n_clusters']
                            })
                        
                        current_models[group_name] = models
                
                model_history = current_models
                last_retrain_date = current_date
                
                # Save models for this retraining point
                self._save_models(current_models, current_date)
                
                print(f"\nModels trained and saved for {len(current_models)} feature groups")
            
            # Apply models to games on current date
            current_games = df[df['startDate'] == current_date].copy()
            
            if len(current_games) > 0 and model_history:
                game_features = pd.DataFrame(index=current_games.index)
                
                # Copy important identifiers
                for id_col in ['gameId', 'team', 'teamId', 'opponent', 'is_home', 
                              'startDate', 'season', 'conference']:
                    if id_col in current_games.columns:
                        game_features[id_col] = current_games[id_col]
                
                # Apply each feature group model
                for group_name, models in model_history.items():
                    if models is None:
                        continue
                    
                    result = self.apply_models(current_games, models, group_name)
                    
                    if result is not None:
                        # Add PCA features
                        for i in range(result['pca_features'].shape[1]):
                            col_name = f'{group_name}_PC{i+1}'
                            game_features.loc[result['index'], col_name] = result['pca_features'][:, i]
                        
                        # Add cluster assignments with mapping
                        clusters = result['clusters']
                        if 'cluster_mapping' in models:
                            # Apply mapping to maintain consistent cluster IDs
                            reverse_map = {v: k for k, v in models['cluster_mapping'].items()}
                            clusters = np.array([reverse_map.get(c, c) for c in clusters])
                        
                        game_features.loc[result['index'], f'{group_name}_cluster'] = clusters
                
                # Add metadata
                game_features['model_date'] = last_retrain_date
                game_features['days_since_retrain'] = (pd.Timestamp(current_date) - pd.Timestamp(last_retrain_date)).days
                
                results.append(game_features)
            
            # Progress update
            if date_idx % 100 == 0 and date_idx > 0:
                pct_complete = (date_idx / len(dates)) * 100
                print(f"\nProgress: {date_idx}/{len(dates)} dates ({pct_complete:.1f}%)")
        
        # Combine all results
        if results:
            print(f"\n{'='*70}")
            print("COMBINING RESULTS")
            print(f"{'='*70}")
            final_features = pd.concat(results, axis=0)
            print(f"Created {len(final_features)} rows with {len(final_features.columns)} columns")
            return final_features
        else:
            print("No results generated")
            return pd.DataFrame()
    
    def _save_models(self, models, date):
        """Save models for a specific training date"""
        date_str = pd.Timestamp(date).strftime('%Y%m%d')
        model_dir = self.output_dir / f'models_{date_str}'
        model_dir.mkdir(exist_ok=True)
        
        # Save each feature group's models
        for group_name, model_data in models.items():
            if model_data is not None:
                filepath = model_dir / f'{group_name}.pkl'
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
        
        # Save metadata
        metadata = {
            'date': str(date),
            'n_groups': len(models),
            'groups': list(models.keys())
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run_full_pipeline(self):
        """
        Run the complete rolling window pipeline
        """
        print("="*70)
        print("ROLLING WINDOW CLUSTER PIPELINE")
        print("="*70)
        
        print("\n1. Loading data...")
        df = self.load_data()
        
        print("\n2. Creating rolling features...")
        features = self.create_rolling_features(df)
        
        if len(features) == 0:
            print("ERROR: No features generated!")
            return None
        
        print("\n3. Saving results...")
        output_file = self.output_dir / 'rolling_features.parquet'
        features.to_parquet(output_file, index=True)
        print(f"Features saved to: {output_file}")
        
        # Save cluster evolution tracking
        evolution_file = self.output_dir / 'cluster_evolution.json'
        with open(evolution_file, 'w') as f:
            json.dump(self.cluster_evolution, f, indent=2, default=str)
        print(f"Cluster evolution saved to: {evolution_file}")
        
        # Save feature group definitions
        groups_file = self.output_dir / 'feature_groups.json'
        with open(groups_file, 'w') as f:
            json.dump(self.feature_groups, f, indent=2)
        print(f"Feature group definitions saved to: {groups_file}")
        
        # Generate summary statistics
        self._generate_summary(features)
        
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*70}")
        
        return features
    
    def _generate_summary(self, features):
        """Generate and save summary statistics"""
        summary = {
            'n_rows': len(features),
            'n_columns': len(features.columns),
            'date_range': {
                'start': str(features['startDate'].min()) if 'startDate' in features.columns else None,
                'end': str(features['startDate'].max()) if 'startDate' in features.columns else None
            },
            'feature_groups': {}
        }
        
        # Count features by group
        for group_name in self.feature_groups.keys():
            pc_cols = [c for c in features.columns if c.startswith(f'{group_name}_PC')]
            cluster_col = f'{group_name}_cluster'
            
            if cluster_col in features.columns:
                n_clusters = features[cluster_col].nunique()
                summary['feature_groups'][group_name] = {
                    'n_pcs': len(pc_cols),
                    'n_clusters': int(n_clusters),
                    'cluster_distribution': features[cluster_col].value_counts().to_dict()
                }
        
        summary_file = self.output_dir / 'pipeline_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nSummary statistics saved to: {summary_file}")
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"  Total rows: {summary['n_rows']:,}")
        print(f"  Total columns: {summary['n_columns']}")
        print(f"  Feature groups processed: {len(summary['feature_groups'])}")


# Example usage
if __name__ == "__main__":
    # Create the rolling window pipeline
    pipeline = RollingWindowClusterPipeline(
        data_path='cbb_team_features.parquet',  # Updated to team-by-team format
        output_dir='rolling_clusters_output',
        min_training_days=365,       # Need at least 1 year of data before predictions
        retrain_interval_days=15     # Retrain models every 15 days
    )
    
    # Run the pipeline
    features = pipeline.run_full_pipeline()
    
    if features is not None:
        print(f"\nSample of output features:")
        print(features.head())
        
        print(f"\nCluster columns:")
        cluster_cols = [c for c in features.columns if '_cluster' in c]
        print(cluster_cols)
        
        print(f"\nPC columns:")
        pc_cols = [c for c in features.columns if '_PC' in c]
        print(f"Total PCs: {len(pc_cols)}")