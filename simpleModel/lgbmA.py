import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LightGBMSpreadModel:
    """
    LightGBM baseline model for spread prediction with walk-forward validation.
    """
    
    def __init__(self, data_path, output_dir='./lgbm_output'):
        """
        Args:
            data_path: Path to spread_training_data.parquet
            output_dir: Directory for outputs
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.predictions = []
        
    def load_and_prepare_data(self):
        """Load data and prepare for training."""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        # Deduplicate games (keep one row per game)
        logger.info(f"Original rows: {len(df):,}")
        df = df.drop_duplicates(subset=['gameId'], keep='first')
        logger.info(f"After deduplication: {len(df):,}")
        
        # Extract season from startDate
        df['season'] = pd.to_datetime(df['startDate']).dt.year
        # Adjust for seasons that span calendar years (games after July = next season)
        df.loc[pd.to_datetime(df['startDate']).dt.month >= 7, 'season'] += 1
        
        logger.info(f"Data spans seasons: {df['season'].min()} to {df['season'].max()}")
        logger.info(f"Cover rate: {df['spread_cover'].mean():.1%}")
        logger.info(f"Mean spread_target: {df['spread_target'].mean():.2f}")
        
        return df
    
    def get_feature_columns(self, df):
        """Identify feature columns (exclude metadata and targets)."""
        exclude_cols = [
            'gameId', 'startDate', 'team', 'opponent', 
            'points', 'opp_points', 'won', 'season',
            'actual_margin', 'team_spread', 'spread_target', 'spread_cover'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Check for any remaining NaN or inf
        nan_cols = df[feature_cols].columns[df[feature_cols].isna().any()].tolist()
        inf_cols = df[feature_cols].columns[np.isinf(df[feature_cols]).any()].tolist()
        
        if nan_cols:
            logger.warning(f"NaN values in {len(nan_cols)} columns: {nan_cols[:5]}")
        if inf_cols:
            logger.warning(f"Inf values in {len(inf_cols)} columns: {inf_cols[:5]}")
        
        logger.info(f"Using {len(feature_cols)} features for training")
        
        return feature_cols
    
    def walk_forward_validation(self, df, feature_cols, test_seasons=None):
        """
        Perform walk-forward validation.
        Train on all data before each season, test on that season.
        """
        if test_seasons is None:
            # Default: test on last 3 seasons
            all_seasons = sorted(df['season'].unique())
            test_seasons = all_seasons[-3:]
        
        logger.info("=" * 80)
        logger.info("STARTING WALK-FORWARD VALIDATION")
        logger.info("=" * 80)
        logger.info(f"Test seasons: {test_seasons}")
        
        all_predictions = []
        results_by_season = {}
        
        for test_season in test_seasons:
            logger.info(f"\n{'='*80}")
            logger.info(f"TESTING ON SEASON {test_season}")
            logger.info(f"{'='*80}")
            
            # Split data
            train_df = df[df['season'] < test_season].copy()
            test_df = df[df['season'] == test_season].copy()
            
            logger.info(f"Training games: {len(train_df):,} (seasons < {test_season})")
            logger.info(f"Test games: {len(test_df):,} (season {test_season})")
            
            if len(test_df) == 0:
                logger.warning(f"No test data for season {test_season}, skipping")
                continue
            
            # Prepare features and targets
            X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            y_train = train_df['spread_target']
            
            X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            y_test = test_df['spread_target']
            
            # Train LightGBM model
            logger.info("Training LightGBM...")
            
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[train_data],
                valid_names=['train'],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            # Save model
            model_path = self.output_dir / f'lgbm_season_{test_season}.txt'
            model.save_model(str(model_path))
            self.models[test_season] = model
            
            # Predict on test set
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Classification metrics (spread cover)
            pred_cover = (y_pred > 0).astype(int)
            actual_cover = test_df['spread_cover'].values
            accuracy = (pred_cover == actual_cover).mean()
            
            logger.info(f"\n{'='*40}")
            logger.info(f"SEASON {test_season} RESULTS")
            logger.info(f"{'='*40}")
            logger.info(f"MAE:      {mae:.3f} points")
            logger.info(f"RMSE:     {rmse:.3f} points")
            logger.info(f"Accuracy: {accuracy:.1%}")
            logger.info(f"{'='*40}")
            
            # Store results
            season_results = test_df[['gameId', 'startDate', 'team', 'opponent']].copy()
            season_results['actual_margin'] = test_df['actual_margin'].values
            season_results['team_spread'] = test_df['team_spread'].values
            season_results['spread_target'] = y_test.values
            season_results['spread_target_pred'] = y_pred
            season_results['spread_cover_actual'] = actual_cover
            season_results['spread_cover_pred'] = pred_cover
            season_results['season'] = test_season
            
            all_predictions.append(season_results)
            results_by_season[test_season] = {
                'mae': mae,
                'rmse': rmse,
                'accuracy': accuracy,
                'n_games': len(test_df)
            }
        
        # Combine all predictions
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Calculate overall metrics
        overall_mae = mean_absolute_error(
            predictions_df['spread_target'], 
            predictions_df['spread_target_pred']
        )
        overall_rmse = np.sqrt(mean_squared_error(
            predictions_df['spread_target'], 
            predictions_df['spread_target_pred']
        ))
        overall_accuracy = (
            predictions_df['spread_cover_actual'] == predictions_df['spread_cover_pred']
        ).mean()
        
        logger.info("\n" + "=" * 80)
        logger.info("OVERALL WALK-FORWARD VALIDATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total test games: {len(predictions_df):,}")
        logger.info(f"Overall MAE:      {overall_mae:.3f} points ⭐")
        logger.info(f"Overall RMSE:     {overall_rmse:.3f} points")
        logger.info(f"Overall Accuracy: {overall_accuracy:.1%}")
        logger.info("=" * 80)
        
        # Save predictions
        pred_path = self.output_dir / 'predictions.parquet'
        predictions_df.to_parquet(pred_path, index=False)
        logger.info(f"Saved predictions to {pred_path}")
        
        # Save summary
        summary = {
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'overall_accuracy': overall_accuracy,
            'by_season': results_by_season,
            'total_games': len(predictions_df)
        }
        
        summary_path = self.output_dir / 'results_summary.joblib'
        joblib.dump(summary, summary_path)
        logger.info(f"Saved summary to {summary_path}")
        
        return predictions_df, summary
    
    def analyze_feature_importance(self, season=None):
        """Analyze and plot feature importance from trained models."""
        if not self.models:
            logger.warning("No models available. Run training first.")
            return
        
        # Use the most recent model if season not specified
        if season is None:
            season = max(self.models.keys())
        
        model = self.models[season]
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TOP 20 FEATURES (Season {season})")
        logger.info(f"{'='*60}")
        for idx, row in importance_df.head(20).iterrows():
            logger.info(f"{row['feature']:40s} {row['importance']:10.0f}")
        
        # Plot top 30 features
        fig, ax = plt.subplots(figsize=(10, 12))
        top_n = 30
        top_features = importance_df.head(top_n)
        
        ax.barh(range(top_n), top_features['importance'].values)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance (Gain)')
        ax.set_title(f'Top {top_n} Features - Season {season}')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'feature_importance_season_{season}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {plot_path}")
        
        # Save feature importance
        importance_path = self.output_dir / f'feature_importance_season_{season}.csv'
        importance_df.to_csv(importance_path, index=False)
        
        plt.close()
        
        return importance_df
    
    def plot_predictions(self, predictions_df):
        """Create diagnostic plots for predictions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(
            predictions_df['spread_target'], 
            predictions_df['spread_target_pred'],
            alpha=0.3, s=10
        )
        min_val = min(predictions_df['spread_target'].min(), 
                     predictions_df['spread_target_pred'].min())
        max_val = max(predictions_df['spread_target'].max(), 
                     predictions_df['spread_target_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Spread Target')
        ax.set_ylabel('Predicted Spread Target')
        ax.set_title('Actual vs Predicted')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Residuals
        ax = axes[0, 1]
        residuals = predictions_df['spread_target'] - predictions_df['spread_target_pred']
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Residual (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Residual Distribution (MAE={residuals.abs().mean():.2f})')
        ax.grid(alpha=0.3)
        
        # 3. MAE by Season
        ax = axes[1, 0]
        mae_by_season = predictions_df.groupby('season').apply(
            lambda x: mean_absolute_error(x['spread_target'], x['spread_target_pred'])
        )
        ax.bar(mae_by_season.index, mae_by_season.values, edgecolor='black')
        ax.set_xlabel('Season')
        ax.set_ylabel('MAE')
        ax.set_title('MAE by Season')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Accuracy by Season
        ax = axes[1, 1]
        acc_by_season = predictions_df.groupby('season').apply(
            lambda x: (x['spread_cover_actual'] == x['spread_cover_pred']).mean()
        )
        ax.bar(acc_by_season.index, acc_by_season.values, edgecolor='black', color='green', alpha=0.7)
        ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Random Guess')
        ax.set_xlabel('Season')
        ax.set_ylabel('Accuracy')
        ax.set_title('Cover Prediction Accuracy by Season')
        ax.set_ylim([0.4, 0.6])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'prediction_diagnostics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved diagnostic plots to {plot_path}")
        
        plt.close()
    
    def run(self, test_seasons=None):
        """Execute full training and evaluation pipeline."""
        logger.info("=" * 80)
        logger.info("LIGHTGBM BASELINE MODEL - SPREAD PREDICTION")
        logger.info("=" * 80)
        
        # Load data
        df = self.load_and_prepare_data()
        
        # Get features
        feature_cols = self.get_feature_columns(df)
        
        # Run walk-forward validation
        predictions_df, summary = self.walk_forward_validation(
            df, feature_cols, test_seasons
        )
        
        # Analyze feature importance
        importance_df = self.analyze_feature_importance()
        
        # Create diagnostic plots
        self.plot_predictions(predictions_df)
        
        logger.info("\n" + "=" * 80)
        logger.info("🎯 BASELINE ESTABLISHED")
        logger.info("=" * 80)
        logger.info(f"Overall MAE: {summary['overall_mae']:.3f} points")
        logger.info("This is your score to beat with GPU models!")
        logger.info("=" * 80)
        
        return summary


# === EXECUTION ===
if __name__ == "__main__":
    # Configuration
    DATA_PATH = 'feature_output/spread_training_data.parquet'
    OUTPUT_DIR = 'lgbm_output'
    
    # Test on last 3 seasons (or specify your own)
    TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
    
    # Run model
    model = LightGBMSpreadModel(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR
    )
    
    summary = model.run(test_seasons=TEST_SEASONS)
    
    print("\n" + "=" * 80)
    print("✅ BASELINE COMPLETE")
    print("=" * 80)
    print(f"MAE to beat: {summary['overall_mae']:.3f} points")
    print("\nReady for Stage 2: GPU Deep Learning Models")
    print("=" * 80)