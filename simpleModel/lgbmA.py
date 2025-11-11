import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedSpreadModel:
    """
    FIXED model with correct betting logic:
    - Uses ALL rows (both team perspectives) - no deduplication!
    - Predicts spread_target directly (cover margin)
    - Simple betting logic: positive = bet team, negative = bet opponent
    - Now includes feature importance analysis to detect data leakage
    """
    
    def __init__(self, data_path, output_dir='./improved_lgbm_output'):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.models = {}
        self.feature_importance_history = []
        
    def load_and_prepare_data(self):
        """Load and prepare data - KEEPING ALL ROWS!"""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        logger.info(f"Total rows: {len(df):,}")
        logger.info("✅ Keeping ALL rows (both team perspectives per game)")
        
        # NO DEDUPLICATION - we want both perspectives!
        # This was Bug #1: removing half the data created a biased dataset
        
        # Extract season
        df['season'] = pd.to_datetime(df['startDate']).dt.year
        df.loc[pd.to_datetime(df['startDate']).dt.month >= 7, 'season'] += 1
        
        # Verify data balance
        logger.info(f"Data spans seasons: {df['season'].min()} to {df['season'].max()}")
        logger.info(f"Team cover rate: {(df['spread_target'] > 0).mean():.1%}")
        logger.info(f"Mean spread_target: {df['spread_target'].mean():.2f}")
        
        # This should be close to 50% if data is balanced
        cover_rate = (df['spread_target'] > 0).mean()
        if abs(cover_rate - 0.5) > 0.05:
            logger.warning(f"⚠️ Cover rate is {cover_rate:.1%} - expected ~50%")
        else:
            logger.info(f"✅ Cover rate is balanced at {cover_rate:.1%}")
        
        return df
    
    def get_feature_columns(self, df):
        """Identify feature columns."""
        exclude_cols = [
            'gameId', 'startDate', 'team', 'opponent', 
            'points', 'opp_points', 'won', 'season',
            'actual_margin', 'team_spread', 'spread_target', 'spread_cover',
            'spread', 'favorite_size', 'is_home',
            'bench_quality_diff', 'away_implied_prob'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Using {len(feature_cols)} features for training")
        logger.info(f"  - Numeric: {len(numeric_cols)}")
        logger.info(f"  - Categorical: {len(feature_cols) - len(numeric_cols)}")
        
        return feature_cols
    
    def analyze_feature_importance(self, model, feature_names, season, top_n=20):
        """
        Analyze and log feature importance to detect potential data leakage.
        
        RED FLAGS for data leakage:
        - Features with names containing: 'actual', 'result', 'final', 'outcome'
        - Features related to future events or game results
        - Suspiciously high importance (>10% of total)
        - Perfect or near-perfect correlations with target
        """
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Calculate percentage
        total_importance = feature_importance['importance'].sum()
        feature_importance['importance_pct'] = (feature_importance['importance'] / total_importance * 100)
        
        # Store for later analysis
        feature_importance['season'] = season
        self.feature_importance_history.append(feature_importance.copy())
        
        # Log top features
        logger.info(f"\n{'='*80}")
        logger.info(f"TOP {top_n} FEATURES - SEASON {season}")
        logger.info(f"{'='*80}")
        logger.info(f"{'Rank':<6} {'Feature':<40} {'Importance %':>12}")
        logger.info("-" * 80)
        
        red_flags = []
        for idx, row in feature_importance.head(top_n).iterrows():
            feature_name = row['feature']
            importance_pct = row['importance_pct']
            
            # Check for red flags
            suspicious_keywords = ['actual', 'result', 'final', 'outcome', 'won', 'points', 'score']
            is_suspicious = any(keyword in feature_name.lower() for keyword in suspicious_keywords)
            is_too_important = importance_pct > 10
            
            flag = ""
            if is_suspicious:
                flag = " 🚨 SUSPICIOUS NAME"
                red_flags.append(f"{feature_name}: Contains suspicious keyword")
            elif is_too_important:
                flag = " ⚠️ VERY HIGH"
                red_flags.append(f"{feature_name}: Unusually high importance ({importance_pct:.1f}%)")
            
            logger.info(f"{idx+1:<6} {feature_name:<40} {importance_pct:>11.2f}%{flag}")
        
        logger.info("=" * 80)
        
        # Report red flags
        if red_flags:
            logger.warning(f"\n⚠️ POTENTIAL DATA LEAKAGE DETECTED:")
            for flag in red_flags:
                logger.warning(f"  - {flag}")
            logger.warning(f"\nReview these features carefully! They may contain information from the future.")
        else:
            logger.info(f"\n✅ No obvious red flags detected in top {top_n} features")
        
        return feature_importance
    
    def plot_feature_importance(self, top_n=30):
        """Create comprehensive feature importance visualizations."""
        if not self.feature_importance_history:
            logger.warning("No feature importance data to plot!")
            return
        
        # Combine all seasons
        all_importance = pd.concat(self.feature_importance_history, ignore_index=True)
        
        # Average importance across seasons
        avg_importance = all_importance.groupby('feature').agg({
            'importance_pct': 'mean',
            'importance': 'mean'
        }).sort_values('importance_pct', ascending=False).head(top_n)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # 1. Top features (average across seasons)
        ax = axes[0]
        y_pos = np.arange(len(avg_importance))
        ax.barh(y_pos, avg_importance['importance_pct'], alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(avg_importance.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Importance (%)', fontsize=11)
        ax.set_title(f'Top {top_n} Most Important Features (Average Across Seasons)', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Highlight suspicious features
        suspicious_keywords = ['actual', 'result', 'final', 'outcome', 'won', 'points', 'score']
        for i, feature in enumerate(avg_importance.index):
            if any(keyword in feature.lower() for keyword in suspicious_keywords):
                ax.get_yticklabels()[i].set_color('red')
                ax.get_yticklabels()[i].set_weight('bold')
        
        # 2. Feature importance stability across seasons
        ax = axes[1]
        top_features = avg_importance.head(10).index
        for feature in top_features:
            feature_data = all_importance[all_importance['feature'] == feature]
            seasons = feature_data['season'].values
            importance = feature_data['importance_pct'].values
            ax.plot(seasons, importance, marker='o', label=feature[:30], linewidth=2)
        
        ax.set_xlabel('Season', fontsize=11)
        ax.set_ylabel('Importance (%)', fontsize=11)
        ax.set_title('Feature Importance Stability Across Seasons', fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'feature_importance_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"\nSaved feature importance plots to {plot_path}")
        plt.close()
        
        # Save detailed feature importance to CSV
        csv_path = self.output_dir / 'feature_importance_detailed.csv'
        avg_importance.to_csv(csv_path)
        logger.info(f"Saved detailed feature importance to {csv_path}")
    
    def calculate_betting_decision(self, predicted_margin, confidence_threshold=2.0):
        """
        FIXED betting logic - no double subtraction!
        
        Args:
            predicted_margin: Model's prediction of spread_target (actual_margin - team_spread)
                             Positive = team covers, Negative = opponent covers
            confidence_threshold: Minimum edge needed to bet (in points)
        
        Returns:
            'bet_team', 'bet_opponent', or 'no_bet'
        """
        if predicted_margin > confidence_threshold:
            # Model predicts team will cover by > threshold. Bet the team.
            return 'bet_team'
        elif predicted_margin < -confidence_threshold:
            # Model predicts opponent will cover by > threshold. Bet opponent.
            return 'bet_opponent'
        else:
            # Edge is too small, no bet.
            return 'no_bet'
    
    def evaluate_betting_performance(self, predictions_df, confidence_threshold=2.0):
        """
        Evaluate betting performance with CORRECT logic.
        """
        results = []
        
        for _, row in predictions_df.iterrows():
            decision = self.calculate_betting_decision(
                row['predicted_spread_target'], 
                confidence_threshold
            )
            
            if decision == 'no_bet':
                results.append({
                    'decision': 'no_bet',
                    'correct': None,
                    'profit': 0
                })
                continue
            
            # Actual result: did the team cover?
            team_covered = row['spread_target'] > 0
            
            # Did our bet win?
            if decision == 'bet_team':
                bet_won = team_covered
            else:  # bet_opponent
                bet_won = not team_covered
            
            # Standard -110 odds (risk $110 to win $100)
            profit = 100 if bet_won else -110
            
            results.append({
                'decision': decision,
                'correct': bet_won,
                'profit': profit
            })
        
        results_df = pd.DataFrame(results)
        predictions_df['betting_decision'] = results_df['decision']
        predictions_df['bet_correct'] = results_df['correct']
        predictions_df['bet_profit'] = results_df['profit']
        
        # Calculate metrics
        bets_made = results_df[results_df['decision'] != 'no_bet']
        
        if len(bets_made) > 0:
            win_rate = bets_made['correct'].mean()
            total_profit = bets_made['profit'].sum()
            roi = (total_profit / (len(bets_made) * 110)) * 100
            profit_per_game = total_profit / len(bets_made)
            
            logger.info(f"\n{'='*60}")
            logger.info("BETTING PERFORMANCE")
            logger.info(f"{'='*60}")
            logger.info(f"Confidence threshold: {confidence_threshold} points")
            logger.info(f"Total games: {len(predictions_df):,}")
            logger.info(f"Bets made: {len(bets_made):,} ({len(bets_made)/len(predictions_df):.1%})")
            logger.info(f"Win rate: {win_rate:.1%} (need 52.4% to break even)")
            logger.info(f"Total profit/loss: ${total_profit:,.0f}")
            logger.info(f"ROI: {roi:.2f}%")
            logger.info(f"Profit per bet: ${profit_per_game:.2f}")
            
            # Show performance vs random
            expected_random = len(bets_made) * 0.5 * 100 - len(bets_made) * 0.5 * 110
            beat_random = total_profit - expected_random
            logger.info(f"Beat random betting by: ${beat_random:,.0f}")
            logger.info(f"{'='*60}")
        else:
            logger.warning("No bets made!")
            win_rate = 0
            total_profit = 0
            roi = 0
            profit_per_game = 0
        
        return {
            'total_games': len(predictions_df),
            'bets_made': len(bets_made),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi': roi,
            'profit_per_game': profit_per_game,
            'confidence_threshold': confidence_threshold
        }
    
    def optimize_confidence_threshold(self, predictions_df):
        """
        Test different confidence thresholds to find optimal betting strategy.
        """
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZING CONFIDENCE THRESHOLD")
        logger.info("="*80)
        
        thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        results = []
        
        for threshold in thresholds:
            perf = self.evaluate_betting_performance(
                predictions_df.copy(), 
                confidence_threshold=threshold
            )
            results.append(perf)
            
            logger.info(f"Threshold {threshold:.1f}: "
                       f"Win Rate={perf['win_rate']:.1%}, "
                       f"ROI={perf['roi']:.2f}%, "
                       f"Bets={perf['bets_made']:,}")
        
        # Find best ROI
        best_result = max(results, key=lambda x: x['roi'])
        logger.info(f"\n🎯 Best threshold: {best_result['confidence_threshold']:.1f}")
        logger.info(f"   ROI: {best_result['roi']:.2f}%")
        logger.info(f"   Win Rate: {best_result['win_rate']:.1%}")
        logger.info(f"   Bets: {best_result['bets_made']:,}")
        
        return results, best_result
    
    def walk_forward_validation(self, df, feature_cols, test_seasons=None):
        """Walk-forward validation with feature importance analysis."""
        if test_seasons is None:
            all_seasons = sorted(df['season'].unique())
            test_seasons = all_seasons[-5:]
        
        logger.info("="*80)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("="*80)
        logger.info(f"Test seasons: {test_seasons}")
        
        all_predictions = []
        
        for test_season in test_seasons:
            logger.info(f"\n{'='*80}")
            logger.info(f"TESTING ON SEASON {test_season}")
            logger.info(f"{'='*80}")
            
            train_df = df[df['season'] < test_season].copy()
            test_df = df[df['season'] == test_season].copy()
            
            logger.info(f"Training rows: {len(train_df):,}")
            logger.info(f"Test rows: {len(test_df):,}")
            
            if len(test_df) == 0:
                continue
            
            # Prepare features
            numeric_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = train_df[feature_cols].select_dtypes(include=['category', 'object']).columns.tolist()
            
            X_train_numeric = train_df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
            X_test_numeric = test_df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
            
            if categorical_cols:
                X_train_cat = train_df[categorical_cols].apply(lambda x: x.cat.codes if x.dtype.name == 'category' else x)
                X_test_cat = test_df[categorical_cols].apply(lambda x: x.cat.codes if x.dtype.name == 'category' else x)
                X_train = pd.concat([X_train_numeric, X_train_cat], axis=1)
                X_test = pd.concat([X_test_numeric, X_test_cat], axis=1)
            else:
                X_train = X_train_numeric
                X_test = X_test_numeric
            
            y_train = train_df['spread_target']
            y_test = test_df['spread_target']
            
            # Train model
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
                params, train_data, num_boost_round=500,
                valid_sets=[train_data], valid_names=['train'],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            self.models[test_season] = model
            
            # 🔍 ANALYZE FEATURE IMPORTANCE (New!)
            self.analyze_feature_importance(model, X_train.columns.tolist(), test_season, top_n=25)
            
            # Predict
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            
            # Calculate prediction accuracy
            mae = mean_absolute_error(y_test, y_pred)
            logger.info(f"Prediction MAE: {mae:.2f} points")
            
            # Store predictions
            season_results = test_df[['gameId', 'startDate', 'team', 'opponent', 'team_spread']].copy()
            season_results['spread_target'] = y_test.values
            season_results['predicted_spread_target'] = y_pred
            season_results['season'] = test_season
            
            all_predictions.append(season_results)
        
        # Combine all predictions
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Save predictions
        pred_path = self.output_dir / 'predictions_with_bets.parquet'
        predictions_df.to_parquet(pred_path, index=False)
        logger.info(f"\nSaved predictions to {pred_path}")
        
        return predictions_df
    
    def plot_betting_results(self, predictions_df):
        """Create visualizations of betting performance."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        bets_only = predictions_df[predictions_df['betting_decision'] != 'no_bet'].copy()
        
        if len(bets_only) == 0:
            logger.warning("No bets to plot!")
            return
        
        bets_only = bets_only.sort_values('startDate')
        bets_only['cumulative_profit'] = bets_only['bet_profit'].cumsum()
        
        # 1. Cumulative profit
        ax = axes[0, 0]
        ax.plot(range(len(bets_only)), bets_only['cumulative_profit'], linewidth=2)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Bet Number')
        ax.set_ylabel('Cumulative Profit ($)')
        ax.set_title('Cumulative Betting Profit')
        ax.grid(alpha=0.3)
        
        # 2. Win rate by season
        ax = axes[0, 1]
        bets_by_season = bets_only.groupby('season').agg({
            'bet_correct': 'mean',
            'bet_profit': 'sum'
        })
        
        ax.bar(bets_by_season.index, bets_by_season['bet_correct'], 
               edgecolor='black', alpha=0.7, color='green')
        ax.axhline(0.524, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax.set_xlabel('Season')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate by Season')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Profit by season
        ax = axes[1, 0]
        ax.bar(bets_by_season.index, bets_by_season['bet_profit'], 
               edgecolor='black', alpha=0.7)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Season')
        ax.set_ylabel('Profit ($)')
        ax.set_title('Profit by Season')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Prediction accuracy distribution
        ax = axes[1, 1]
        errors = predictions_df['spread_target'] - predictions_df['predicted_spread_target']
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Prediction Error (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.set_title('Prediction Error Distribution')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'betting_performance.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved betting performance plot to {plot_path}")
        plt.close()
    
    def run(self, test_seasons=None):
        """Execute full pipeline."""
        logger.info("="*80)
        logger.info("FIXED SPREAD MODEL WITH CORRECT BETTING LOGIC")
        logger.info("="*80)
        
        df = self.load_and_prepare_data()
        feature_cols = self.get_feature_columns(df)
        predictions_df = self.walk_forward_validation(df, feature_cols, test_seasons)
        
        # 🔍 Plot feature importance analysis (New!)
        self.plot_feature_importance(top_n=30)
        
        # Optimize threshold
        threshold_results, best_result = self.optimize_confidence_threshold(predictions_df)
        
        # Apply best threshold for final evaluation
        final_predictions = predictions_df.copy()
        betting_results = self.evaluate_betting_performance(
            final_predictions, 
            confidence_threshold=best_result['confidence_threshold']
        )
        
        # Plot results
        self.plot_betting_results(final_predictions)
        
        logger.info("\n" + "="*80)
        logger.info("🎯 FINAL RESULTS (FIXED MODEL)")
        logger.info("="*80)
        logger.info(f"Best Confidence Threshold: {best_result['confidence_threshold']:.1f} points")
        logger.info(f"Win Rate: {best_result['win_rate']:.1%}")
        logger.info(f"ROI: {best_result['roi']:.2f}%")
        logger.info(f"Total Profit: ${best_result['total_profit']:,.0f}")
        logger.info(f"Bets Made: {best_result['bets_made']:,} / {len(predictions_df):,}")
        logger.info("="*80)
        
        return betting_results, threshold_results


if __name__ == "__main__":
    DATA_PATH = 'feature_output/spread_training_data.parquet'
    OUTPUT_DIR = 'improved_lgbm_output'
    TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
    
    model = ImprovedSpreadModel(data_path=DATA_PATH, output_dir=OUTPUT_DIR)
    betting_results, threshold_results = model.run(test_seasons=TEST_SEASONS)
    
    print("\n" + "="*80)
    print("✅ FIXED MODEL COMPLETE")
    print("="*80)
    print(f"ROI: {betting_results['roi']:.2f}%")
    print(f"Win Rate: {betting_results['win_rate']:.1%}")
    print("="*80)