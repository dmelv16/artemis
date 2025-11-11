import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CBBFeatureEngineer:
    """
    Advanced feature engineering pipeline for college basketball game prediction.
    
    Handles:
    - Domain-informed feature creation
    - Static PCA on historical data
    - Separate datasets for spread and total prediction
    """
    
    def __init__(self, parquet_path, output_dir='./output', pca_training_cutoff='2021-07-01'):
        """
        Args:
            parquet_path: Path to cbb_teams.parquet file
            output_dir: Directory for output files
            pca_training_cutoff: Date before which to train PCA (format: 'YYYY-MM-DD')
        """
        self.parquet_path = parquet_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.pca_training_cutoff = pd.to_datetime(pca_training_cutoff)
        
        # Storage for fitted transformers
        self.pca_transformers = {}
        self.scalers = {}
        
    def load_data(self):
        """Load parquet file and perform initial cleaning."""
        logger.info(f"Loading data from {self.parquet_path}")
        df = pd.read_parquet(self.parquet_path)
        
        # Convert startDate to datetime
        df['startDate'] = pd.to_datetime(df['startDate'])
        
        # Remove all *_season columns as requested
        season_cols = [col for col in df.columns if col.endswith('_season')]
        df = df.drop(columns=season_cols)
        logger.info(f"Removed {len(season_cols)} season-level columns")
        
        logger.info(f"Loaded {len(df)} rows spanning {df['startDate'].min()} to {df['startDate'].max()}")
        
        # Check spread data availability
        spread_data = df[df['spread'].notna()]
        if len(spread_data) > 0:
            logger.info(f"Spread data available from {spread_data['startDate'].min()} ({len(spread_data)} games)")
        
        total_data = df[df['overUnder'].notna()]
        if len(total_data) > 0:
            logger.info(f"Total data available from {total_data['startDate'].min()} ({len(total_data)} games)")
        
        return df
    
    def create_aggregated_features(self, df):
        """Create domain-informed aggregated features."""
        logger.info("Creating aggregated features...")
        
        df = df.copy()
        
        # === OFFENSIVE EFFICIENCY (Four Factors) ===
        # L10 version
        df['team_offensive_efficiency_L10'] = (
            0.4 * df['team_effectiveFieldGoalPct_L10'] +
            0.25 * (1 - df['team_turnoverRatio_L10']) +
            0.2 * df['team_offensiveReboundPct_L10'] +
            0.15 * df['team_freeThrowRate_L10']
        )
        
        df['opp_offensive_efficiency_L10'] = (
            0.4 * df['opp_effectiveFieldGoalPct_L10'] +
            0.25 * (1 - df['opp_turnoverRatio_L10']) +
            0.2 * df['opp_offensiveReboundPct_L10'] +
            0.15 * df['opp_freeThrowRate_L10']
        )
        
        # L5 version
        df['team_offensive_efficiency_L5'] = (
            0.4 * df['team_effectiveFieldGoalPct_L5'] +
            0.25 * (1 - df['team_turnoverRatio_L5']) +
            0.2 * df['team_offensiveReboundPct_L5'] +
            0.15 * df['team_freeThrowRate_L5']
        )
        
        df['opp_offensive_efficiency_L5'] = (
            0.4 * df['opp_effectiveFieldGoalPct_L5'] +
            0.25 * (1 - df['opp_turnoverRatio_L5']) +
            0.2 * df['opp_offensiveReboundPct_L5'] +
            0.15 * df['opp_freeThrowRate_L5']
        )
        
        # === DEFENSIVE EFFICIENCY ===
        df['team_defensive_efficiency_L10'] = (
            0.4 * (1 - df['team_oppEffectiveFieldGoalPct_L10']) +
            0.25 * df['team_oppTurnoverRatio_L10'] +
            0.2 * (1 - df['team_oppOffensiveReboundPct_L10']) +
            0.15 * (1 - df['team_oppFreeThrowRate_L10'])
        )
        
        df['opp_defensive_efficiency_L10'] = (
            0.4 * (1 - df['opp_oppEffectiveFieldGoalPct_L10']) +
            0.25 * df['opp_oppTurnoverRatio_L10'] +
            0.2 * (1 - df['opp_oppOffensiveReboundPct_L10']) +
            0.15 * (1 - df['opp_oppFreeThrowRate_L10'])
        )
        
        df['team_defensive_efficiency_L5'] = (
            0.4 * (1 - df['team_oppEffectiveFieldGoalPct_L5']) +
            0.25 * df['team_oppTurnoverRatio_L5'] +
            0.2 * (1 - df['team_oppOffensiveReboundPct_L5']) +
            0.15 * (1 - df['team_oppFreeThrowRate_L5'])
        )
        
        df['opp_defensive_efficiency_L5'] = (
            0.4 * (1 - df['opp_oppEffectiveFieldGoalPct_L5']) +
            0.25 * df['opp_oppTurnoverRatio_L5'] +
            0.2 * (1 - df['opp_oppOffensiveReboundPct_L5']) +
            0.15 * (1 - df['opp_oppFreeThrowRate_L5'])
        )
        
        # === MOMENTUM/FORM ===
        # Recent form (higher weight on L5)
        df['team_recent_form'] = (
            0.6 * df['team_rating_L5'] + 0.4 * df['team_rating_L10']
        )
        df['opp_recent_form'] = (
            0.6 * df['opp_rating_L5'] + 0.4 * df['opp_rating_L10']
        )
        
        # Trend (improving vs declining)
        df['team_trend'] = df['team_rating_L5'] - df['team_rating_L10']
        df['opp_trend'] = df['opp_rating_L5'] - df['opp_rating_L10']
        
        # === PACE/TEMPO MATCHUP ===
        df['pace_matchup'] = df['team_pace_L10'] + df['opp_pace_L10']
        df['style_clash'] = np.abs(df['team_pace_L10'] - df['opp_pace_L10'])
        
        # === POSITION GROUP ADVANTAGES ===
        # Backcourt (guards)
        df['backcourt_scoring_advantage'] = (
            df['team_guard_points_sum_L10'] - df['opp_guard_points_sum_L10']
        )
        
        # Frontcourt (forwards + centers combined)
        df['frontcourt_rebounding_advantage'] = (
            (df['team_forward_totalRebounds_sum_L10'] + df['team_center_totalRebounds_sum_L10']) -
            (df['opp_forward_totalRebounds_sum_L10'] + df['opp_center_totalRebounds_sum_L10'])
        )
        
        # === BENCH QUALITY ===
        df['bench_quality_diff'] = df['team_bench_netRating_L10'] - df['opp_bench_netRating_L10']
        
        logger.info("Created aggregated features")
        return df
    
    def fit_pca_groups(self, df):
        """
        Fit PCA transformers on pre-cutoff data for specific feature groups.
        These transformers will be frozen and applied to all data.
        """
        logger.info(f"Fitting PCA transformers on data before {self.pca_training_cutoff}")
        
        # Filter to training period
        train_df = df[df['startDate'] < self.pca_training_cutoff].copy()
        logger.info(f"Using {len(train_df)} games for PCA training")
        
        # Define feature groups for PCA
        pca_groups = {
            'team_four_factors': [
                'team_effectiveFieldGoalPct_L10', 'team_turnoverRatio_L10',
                'team_offensiveReboundPct_L10', 'team_freeThrowRate_L10',
                'team_effectiveFieldGoalPct_L5', 'team_turnoverRatio_L5',
                'team_offensiveReboundPct_L5', 'team_freeThrowRate_L5'
            ],
            'team_defensive_factors': [
                'team_oppEffectiveFieldGoalPct_L10', 'team_oppTurnoverRatio_L10',
                'team_oppOffensiveReboundPct_L10', 'team_oppFreeThrowRate_L10',
                'team_oppEffectiveFieldGoalPct_L5', 'team_oppTurnoverRatio_L5',
                'team_oppOffensiveReboundPct_L5', 'team_oppFreeThrowRate_L5'
            ],
            'opp_four_factors': [
                'opp_effectiveFieldGoalPct_L10', 'opp_turnoverRatio_L10',
                'opp_offensiveReboundPct_L10', 'opp_freeThrowRate_L10',
                'opp_effectiveFieldGoalPct_L5', 'opp_turnoverRatio_L5',
                'opp_offensiveReboundPct_L5', 'opp_freeThrowRate_L5'
            ],
            'opp_defensive_factors': [
                'opp_oppEffectiveFieldGoalPct_L10', 'opp_oppTurnoverRatio_L10',
                'opp_oppOffensiveReboundPct_L10', 'opp_oppFreeThrowRate_L10',
                'opp_oppEffectiveFieldGoalPct_L5', 'opp_oppTurnoverRatio_L5',
                'opp_oppOffensiveReboundPct_L5', 'opp_oppFreeThrowRate_L5'
            ],
            'team_shooting': [
                'team_threePointFGPct_L10', 
                'team_freeThrowsPct_L10', 'team_fieldGoalsPct_L10',
                'team_threePointFGPct_L5', 
                'team_freeThrowsPct_L5', 'team_fieldGoalsPct_L5'
            ],
            'opp_shooting': [
                'opp_threePointFGPct_L10', 
                'opp_freeThrowsPct_L10', 'opp_fieldGoalsPct_L10',
                'opp_threePointFGPct_L5',
                'opp_freeThrowsPct_L5', 'opp_fieldGoalsPct_L5'
            ],
            'team_rotation': [
                'team_starter_points_mean_L10', 'team_starter_minutes_mean_L10',
                'team_bench_points_L10', 'team_bench_minutes_L10',
                'team_starter_gameScore_mean_L10', 'team_bench_netRating_L10',
                'team_starter_points_mean_L5', 'team_bench_points_L5'
            ]
        }
        
        # Fit PCA for each group
        for group_name, features in pca_groups.items():
            logger.info(f"Fitting PCA for {group_name} ({len(features)} features)")
            
            # Extract features, handle missing values
            X = train_df[features].copy()
            
            # Fill NaN with median (only for PCA training)
            X_filled = X.fillna(X.median())
            
            # Check for remaining issues
            if X_filled.isnull().any().any():
                logger.warning(f"NaN values remain in {group_name}, filling with 0")
                X_filled = X_filled.fillna(0)
            
            if np.isinf(X_filled).any().any():
                logger.warning(f"Inf values in {group_name}, clipping")
                X_filled = X_filled.replace([np.inf, -np.inf], [X_filled[~np.isinf(X_filled)].max().max(), 
                                                                   X_filled[~np.isinf(X_filled)].min().min()])
            
            # Standardize first
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_filled)
            
            # Fit PCA (2-3 components per group)
            n_components = 3 if len(features) >= 6 else 2
            pca = PCA(n_components=n_components)
            pca.fit(X_scaled)
            
            # Store transformers
            self.pca_transformers[group_name] = pca
            self.scalers[group_name] = scaler
            
            var_explained = pca.explained_variance_ratio_.sum()
            logger.info(f"  → {n_components} components explain {var_explained:.1%} variance")
        
        # Save transformers
        transformer_path = self.output_dir / 'pca_transformers.pkl'
        joblib.dump({
            'pca': self.pca_transformers,
            'scalers': self.scalers,
            'training_cutoff': self.pca_training_cutoff
        }, transformer_path)
        logger.info(f"Saved PCA transformers to {transformer_path}")
        
        return self
    
    def transform_pca_features(self, df):
        """Apply frozen PCA transformers to create latent features."""
        logger.info("Transforming features with frozen PCA transformers")
        
        df = df.copy()
        
        pca_groups = {
            'team_four_factors': [
                'team_effectiveFieldGoalPct_L10', 'team_turnoverRatio_L10',
                'team_offensiveReboundPct_L10', 'team_freeThrowRate_L10',
                'team_effectiveFieldGoalPct_L5', 'team_turnoverRatio_L5',
                'team_offensiveReboundPct_L5', 'team_freeThrowRate_L5'
            ],
            'team_defensive_factors': [
                'team_oppEffectiveFieldGoalPct_L10', 'team_oppTurnoverRatio_L10',
                'team_oppOffensiveReboundPct_L10', 'team_oppFreeThrowRate_L10',
                'team_oppEffectiveFieldGoalPct_L5', 'team_oppTurnoverRatio_L5',
                'team_oppOffensiveReboundPct_L5', 'team_oppFreeThrowRate_L5'
            ],
            'opp_four_factors': [
                'opp_effectiveFieldGoalPct_L10', 'opp_turnoverRatio_L10',
                'opp_offensiveReboundPct_L10', 'opp_freeThrowRate_L10',
                'opp_effectiveFieldGoalPct_L5', 'opp_turnoverRatio_L5',
                'opp_offensiveReboundPct_L5', 'opp_freeThrowRate_L5'
            ],
            'opp_defensive_factors': [
                'opp_oppEffectiveFieldGoalPct_L10', 'opp_oppTurnoverRatio_L10',
                'opp_oppOffensiveReboundPct_L10', 'opp_oppFreeThrowRate_L10',
                'opp_oppEffectiveFieldGoalPct_L5', 'opp_oppTurnoverRatio_L5',
                'opp_oppOffensiveReboundPct_L5', 'opp_oppFreeThrowRate_L5'
            ],
            'team_shooting': [
                'team_threePointFGPct_L10',
                'team_freeThrowsPct_L10', 'team_fieldGoalsPct_L10',
                'team_threePointFGPct_L5', 
                'team_freeThrowsPct_L5', 'team_fieldGoalsPct_L5'
            ],
            'opp_shooting': [
                'opp_threePointFGPct_L10', 
                'opp_freeThrowsPct_L10', 'opp_fieldGoalsPct_L10',
                'opp_threePointFGPct_L5', 
                'opp_freeThrowsPct_L5', 'opp_fieldGoalsPct_L5'
            ],
            'team_rotation': [
                'team_starter_points_mean_L10', 'team_starter_minutes_mean_L10',
                'team_bench_points_L10', 'team_bench_minutes_L10',
                'team_starter_gameScore_mean_L10', 'team_bench_netRating_L10',
                'team_starter_points_mean_L5', 'team_bench_points_L5'
            ]
        }
        
        for group_name, features in pca_groups.items():
            # Extract and fill
            X = df[features].copy()
            X_filled = X.fillna(X.median()).fillna(0)
            X_filled = X_filled.replace([np.inf, -np.inf], 0)
            
            # Transform
            scaler = self.scalers[group_name]
            pca = self.pca_transformers[group_name]
            
            X_scaled = scaler.transform(X_filled)
            X_pca = pca.transform(X_scaled)
            
            # Add to dataframe
            for i in range(X_pca.shape[1]):
                df[f'{group_name}_PC{i+1}'] = X_pca[:, i]
        
        logger.info("PCA transformation complete")
        return df
    
    def create_final_feature_set(self, df):
        """Select final features for modeling."""
        logger.info("Creating final feature set")
        
        final_features = [
            # === MARKET SIGNALS (10) ===
            'spread', 'overUnder', 'spread_movement', 'total_movement',
            'home_implied_prob', 'away_implied_prob',
            'homeMoneyline', 'awayMoneyline', 
            'line_movement_magnitude', 'favorite_size',
            
            # === GAME CONTEXT (13) ===
            'is_home', 'is_neutral_site', 
            'rest_days', 'opp_rest_days', 'rest_days_diff',
            'is_b2b', 'opp_is_b2b', 'b2b_diff',
            'streak', 'streak_diff',
            'day_of_week', 'is_weekend', 'days_since_season_start',
            
            # === RANKINGS (10) ===
            'team_AP_rank', 'opp_AP_rank', 'AP_rank_diff',
            'team_AP_tier', 'opp_AP_tier',
            'team_Coaches_rank', 'opp_Coaches_rank', 'Coaches_rank_diff',
            'team_Coaches_tier', 'opp_Coaches_tier',
            
            # === AGGREGATED FEATURES (16) ===
            'team_offensive_efficiency_L10', 'team_offensive_efficiency_L5',
            'opp_offensive_efficiency_L10', 'opp_offensive_efficiency_L5',
            'team_defensive_efficiency_L10', 'team_defensive_efficiency_L5',
            'opp_defensive_efficiency_L10', 'opp_defensive_efficiency_L5',
            'team_recent_form', 'opp_recent_form',
            'team_trend', 'opp_trend',
            'pace_matchup', 'style_clash',
            'backcourt_scoring_advantage', 'frontcourt_rebounding_advantage',
            'bench_quality_diff',
            'team_key_minutes_availability_pct', 'opp_key_minutes_availability_pct',
            'team_shooting_balance_L10', 'opp_shooting_balance_L10',
            
            # === PCA FEATURES (will be added dynamically)
        ]
        
        # Add all PCA features
        pca_features = [col for col in df.columns if '_PC' in col and col.endswith(tuple(f'_PC{i}' for i in range(1, 4)))]
        final_features.extend(pca_features)
        
        # Filter to available columns
        available_features = [f for f in final_features if f in df.columns]
        missing_features = set(final_features) - set(available_features)
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features}")
        
        logger.info(f"Final feature set: {len(available_features)} features")
        
        # Return only features and basic metadata
        # Target columns will be added in create_target_datasets()
        return df[available_features + ['gameId', 'startDate', 'team', 'opponent', 
                                         'points', 'opp_points', 'won', 'is_home']]
    
    def create_target_datasets(self, df):
        """Create separate datasets for spread and total prediction."""
        logger.info("Creating target-specific datasets")
        
        # === SPREAD DATASET ===
        spread_df = df[df['spread'].notna()].copy()
        
        # CRITICAL: Remove any duplicate columns that might exist
        if spread_df.columns.duplicated().any():
            logger.warning(f"Found {spread_df.columns.duplicated().sum()} duplicate columns, removing...")
            spread_df = spread_df.loc[:, ~spread_df.columns.duplicated()]
        
        if 'is_home' not in spread_df.columns:
            logger.error("'is_home' column not found - cannot properly calculate spread target")
            raise ValueError("Missing 'is_home' column")
        
        # CRITICAL FIX: Calculate actual game margin from team's perspective
        spread_df['actual_margin'] = spread_df['points'] - spread_df['opp_points']
        
        # CRITICAL FIX: Adjust spread to team's perspective
        # The 'spread' column is ALWAYS from HOME team's perspective
        # If our 'team' is away (is_home = 0), we must flip the sign
        logger.info("Adjusting spread to team's perspective...")
        
        # Use numpy where with explicit array conversion
        is_home_array = spread_df['is_home'].to_numpy()
        spread_array = spread_df['spread'].to_numpy()
        
        # Create team_spread: if home, keep as-is; if away, flip sign
        team_spread_array = np.where(is_home_array == 1, spread_array, -spread_array)
        
        spread_df['team_spread'] = team_spread_array
        
        # Calculate spread target using team-perspective spread
        # Positive target = team beat the spread (covered)
        spread_df['spread_target'] = spread_df['actual_margin'] - spread_df['team_spread']
        spread_df['spread_cover'] = (spread_df['actual_margin'] + spread_df['team_spread'] > 0).astype(int)
        
        # VALIDATION: Check for data leakage
        logger.info("=== SPREAD TARGET VALIDATION ===")
        
        # First, check for duplicate games
        dup_check = spread_df.groupby('gameId').size()
        unique_games = len(dup_check)
        total_rows = len(spread_df)
        
        if (dup_check > 1).any():
            num_duplicates = (dup_check > 1).sum()
            logger.warning(f"⚠️  DUPLICATE GAMES DETECTED!")
            logger.warning(f"  Total rows: {total_rows:,}")
            logger.warning(f"  Unique games: {unique_games:,}")
            logger.warning(f"  Games with 2+ rows: {num_duplicates:,}")
            logger.warning(f"  Most common row count per game: {dup_check.mode()[0]}")
            logger.warning("")
            logger.warning("This explains the unusual cover rate. Each game appears from both team perspectives.")
            logger.warning("The training script will automatically keep only one row per game.")
        
        logger.info(f"Spread dataset: {len(spread_df)} rows ({unique_games:,} unique games)")
        logger.info(f"  Date range: {spread_df['startDate'].min()} to {spread_df['startDate'].max()}")
        logger.info(f"  Cover rate (all rows): {spread_df['spread_cover'].mean():.1%}")
        logger.info(f"  Mean actual margin: {spread_df['actual_margin'].mean():.2f}")
        logger.info(f"  Mean team_spread: {spread_df['team_spread'].mean():.2f}")
        logger.info(f"  Mean spread_target: {spread_df['spread_target'].mean():.2f}")
        
        # Show what cover rate would be with deduplicated data
        if (dup_check > 1).any():
            deduped = spread_df.drop_duplicates(subset=['gameId'], keep='first')
            logger.info(f"  Cover rate (after dedup): {deduped['spread_cover'].mean():.1%} ← This is the real rate")
            
            # Validate it's now close to 50%
            cover_rate = deduped['spread_cover'].mean()
            if 0.48 <= cover_rate <= 0.52:
                logger.info(f"  ✓ Cover rate looks healthy!")
            else:
                logger.warning(f"  ⚠️  Cover rate still unusual. Expected ~50%, got {cover_rate:.1%}")
                logger.warning(f"  Possible remaining issues:")
                logger.warning(f"    - Time-based data leakage")
                logger.warning(f"    - Spread perspective still incorrect")
                logger.warning(f"    - Line movement not accounted for")
        
        # === TOTAL DATASET ===
        total_df = df[df['overUnder'].notna()].copy()
        
        # CRITICAL: Remove any duplicate columns that might exist
        if total_df.columns.duplicated().any():
            logger.warning(f"Found {total_df.columns.duplicated().sum()} duplicate columns, removing...")
            total_df = total_df.loc[:, ~total_df.columns.duplicated()]
        
        # Calculate total
        if 'total_actual' not in total_df.columns:
            total_df['total_actual'] = total_df['points'] + total_df['opp_points']
        
        total_df['total_target'] = total_df['total_actual'] - total_df['overUnder']
        total_df['total_over'] = (total_df['total_target'] > 0).astype(int)
        
        logger.info("=== TOTAL TARGET VALIDATION ===")
        
        # Check for duplicate games
        dup_check = total_df.groupby('gameId').size()
        unique_games = len(dup_check)
        total_rows = len(total_df)
        
        if (dup_check > 1).any():
            num_duplicates = (dup_check > 1).sum()
            logger.warning(f"⚠️  DUPLICATE GAMES DETECTED!")
            logger.warning(f"  Total rows: {total_rows:,}")
            logger.warning(f"  Unique games: {unique_games:,}")
            logger.warning(f"  Games with 2+ rows: {num_duplicates:,}")
        
        logger.info(f"Total dataset: {len(total_df)} rows ({unique_games:,} unique games)")
        logger.info(f"  Date range: {total_df['startDate'].min()} to {total_df['startDate'].max()}")
        logger.info(f"  Over rate (all rows): {total_df['total_over'].mean():.1%}")
        logger.info(f"  Mean O/U line: {total_df['overUnder'].mean():.1f}")
        logger.info(f"  Mean actual total: {total_df['total_actual'].mean():.1f}")
        logger.info(f"  Mean total_target: {total_df['total_target'].mean():.2f}")
        
        # Show deduplicated rate
        if (dup_check > 1).any():
            deduped = total_df.drop_duplicates(subset=['gameId'], keep='first')
            logger.info(f"  Over rate (after dedup): {deduped['total_over'].mean():.1%} ← This is the real rate")
        
        return spread_df, total_df
    
    def run_pipeline(self):
        """Execute full feature engineering pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 80)
        
        # Load data
        df = self.load_data()
        
        # Create aggregated features
        df = self.create_aggregated_features(df)
        
        # Fit PCA transformers
        self.fit_pca_groups(df)
        
        # Transform all data
        df = self.transform_pca_features(df)
        
        # Create final feature set
        df = self.create_final_feature_set(df)
        
        # Create target datasets
        spread_df, total_df = self.create_target_datasets(df)
        
        # Save outputs
        spread_path = self.output_dir / 'spread_training_data.parquet'
        total_path = self.output_dir / 'total_training_data.parquet'
        
        spread_df.to_parquet(spread_path, index=False)
        total_df.to_parquet(total_path, index=False)
        
        logger.info(f"Saved spread dataset to {spread_path}")
        logger.info(f"Saved total dataset to {total_path}")
        
        # Summary statistics
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total features: {len([c for c in spread_df.columns if c not in ['gameId', 'startDate', 'team', 'opponent', 'points', 'opp_points', 'won', 'spread_actual', 'total_actual', 'spread_target', 'spread_cover']])}")
        logger.info(f"Spread games: {len(spread_df):,}")
        logger.info(f"Total games: {len(total_df):,}")
        logger.info(f"PCA groups: {len(self.pca_transformers)}")
        
        return spread_df, total_df


# === EXECUTION ===
if __name__ == "__main__":
    # Configuration
    PARQUET_PATH = r"C:\Users\DMelv\Documents\artemis\cbb_team_features.parquet"
    OUTPUT_DIR = 'feature_output'
    PCA_TRAINING_CUTOFF = '2021-07-01'
    
    # Run pipeline
    engineer = CBBFeatureEngineer(
        parquet_path=PARQUET_PATH,
        output_dir=OUTPUT_DIR,
        pca_training_cutoff=PCA_TRAINING_CUTOFF
    )
    
    spread_data, total_data = engineer.run_pipeline()
    
    print("\n" + "=" * 80)
    print("READY FOR MODEL TRAINING")
    print("=" * 80)
    print(f"Spread model features: {spread_data.shape[1] - 9}")
    print(f"Total model features: {total_data.shape[1] - 9}")
    print("\nNext steps:")
    print("1. Load 'spread_training_data.parquet' for spread model")
    print("2. Load 'total_training_data.parquet' for total model")
    print("3. Train models using 'spread_target' and 'total_target' as targets")