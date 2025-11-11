import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score
import joblib
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpreadBettingModel:
    """
    Predicts which TEAM will cover the spread.
    
    KEY CONCEPTS:
    - spread_cover = 1: This TEAM covered (actual_margin > team_spread)
    - spread_cover = 0: This TEAM did NOT cover (opponent covered)
    - We predict probability that THIS TEAM covers
    - Only bet when we're highly confident (> threshold like 0.65)
    
    ANTI-CHEATING MEASURES:
    - Feature importance analysis with red flags
    - Strict validation: no future data leakage
    - Threshold optimized on validation set only
    """
    
    def __init__(self, data_path, output_dir='./spread_model_output'):
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
        
        # Extract season
        df['season'] = pd.to_datetime(df['startDate']).dt.year
        df.loc[pd.to_datetime(df['startDate']).dt.month >= 7, 'season'] += 1
        
        # Verify data integrity
        logger.info(f"\n{'='*80}")
        logger.info("DATA INTEGRITY CHECKS")
        logger.info(f"{'='*80}")
        logger.info(f"Data spans seasons: {df['season'].min()} to {df['season'].max()}")
        logger.info(f"Total games (row pairs): {len(df) // 2:,}")
        
        # Check spread_cover distribution (should be ~50%)
        cover_rate = df['spread_cover'].mean()
        logger.info(f"Cover rate: {cover_rate:.1%} (expect ~50%)")
        if abs(cover_rate - 0.5) > 0.05:
            logger.warning(f"⚠️ Cover rate is {cover_rate:.1%} - should be ~50%!")
        else:
            logger.info(f"✅ Cover rate is balanced")
        
        # Verify spread_target relationship
        logger.info(f"Mean spread_target: {df['spread_target'].mean():.2f} (expect ~0)")
        logger.info(f"Spread range: {df['team_spread'].min():.1f} to {df['team_spread'].max():.1f}")
        
        return df
    
    def get_feature_columns(self, df):
        """
        Identify feature columns with STRICT exclusions to prevent cheating.
        
        EXCLUDED (would be cheating):
        - actual_margin: This is the actual game result!
        - points, opp_points: Actual final scores
        - won: Actual game outcome
        - spread_target: Derived from actual_margin
        - spread_cover: This is what we're trying to predict!
        
        INCLUDED (legit info available before game):
        - team_spread: The line we're betting against (MUST include!)
        - All team statistics (efficiency, form, rankings, etc.)
        - Game context (rest days, home/away, etc.)
        """
        
        # These columns would leak future information
        forbidden_cols = [
            'gameId', 'startDate', 'team', 'opponent', 'season',
            # ACTUAL GAME RESULTS (would be cheating!)
            'points', 'opp_points', 'won', 'actual_margin',
            'spread_target', 'spread_cover',
        ]
        
        # Columns that might seem like cheating but aren't
        allowed_cols = [
            'team_spread',  # The line - we NEED this!
            'spread',       # Same as team_spread
            'favorite_size', # Absolute value of spread
        ]
        
        feature_cols = [col for col in df.columns if col not in forbidden_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"\n{'='*80}")
        logger.info("FEATURE SELECTION")
        logger.info(f"{'='*80}")
        logger.info(f"Total features: {len(feature_cols)}")
        logger.info(f"  - Numeric: {len(numeric_cols)}")
        logger.info(f"  - Categorical: {len(feature_cols) - len(numeric_cols)}")
        
        # Verify critical features
        if 'team_spread' in feature_cols or 'spread' in feature_cols:
            logger.info("  ✅ Spread included (essential for betting!)")
        else:
            logger.error("  🚨 SPREAD NOT INCLUDED - Model will fail!")
            
        # Check for leaked features
        leaked = [col for col in feature_cols if any(word in col.lower() 
                  for word in ['actual', 'final', 'result', 'points', 'score', 'won'])]
        if leaked:
            logger.error(f"  🚨 POTENTIAL LEAKAGE DETECTED: {leaked}")
        else:
            logger.info("  ✅ No obvious leakage in feature names")
        
        return feature_cols
    
    def detect_data_leakage(self, model, feature_names, X_train, y_train, season):
        """
        Comprehensive data leakage detection.
        
        RED FLAGS:
        1. Feature has suspicious name (actual, result, final, etc.)
        2. Feature has >10% importance
        3. Feature has perfect/near-perfect correlation with target
        4. Feature has implausible predictive power
        """
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        total_importance = feature_importance['importance'].sum()
        feature_importance['importance_pct'] = (
            feature_importance['importance'] / total_importance * 100
        )
        
        # Store history
        feature_importance['season'] = season
        self.feature_importance_history.append(feature_importance.copy())
        
        # RED FLAG CHECKS
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 DATA LEAKAGE DETECTION - SEASON {season}")
        logger.info(f"{'='*80}")
        
        red_flags = []
        
        # Check top features
        top_n = 30
        logger.info(f"\n{'Rank':<6} {'Feature':<40} {'Importance %':>12} {'Status':<20}")
        logger.info("-" * 80)
        
        for idx, row in feature_importance.head(top_n).iterrows():
            feature_name = row['feature']
            importance_pct = row['importance_pct']
            
            # Check 1: Suspicious names
            suspicious_keywords = ['actual', 'result', 'final', 'outcome', 'won', 
                                  'points', 'score', 'margin']
            is_suspicious_name = any(keyword in feature_name.lower() 
                                    for keyword in suspicious_keywords)
            
            # Check 2: Too important
            is_too_important = importance_pct > 10
            
            # Check 3: Perfect correlation (if we can calculate it)
            if feature_name in X_train.columns:
                try:
                    correlation = abs(X_train[feature_name].corr(y_train))
                    is_perfect_corr = correlation > 0.95
                except:
                    is_perfect_corr = False
            else:
                is_perfect_corr = False
            
            # Determine status
            status = "✅ OK"
            if is_suspicious_name:
                status = "🚨 SUSPICIOUS NAME"
                red_flags.append(f"{feature_name}: Contains forbidden keyword")
            elif is_too_important:
                status = "⚠️ VERY HIGH"
                red_flags.append(f"{feature_name}: Suspiciously high ({importance_pct:.1f}%)")
            elif is_perfect_corr:
                status = "🚨 PERFECT CORR"
                red_flags.append(f"{feature_name}: Near-perfect correlation with target")
            
            logger.info(f"{idx+1:<6} {feature_name:<40} {importance_pct:>11.2f}% {status:<20}")
        
        logger.info("=" * 80)
        
        if red_flags:
            logger.error(f"\n🚨 CRITICAL: POTENTIAL DATA LEAKAGE DETECTED!")
            logger.error("=" * 80)
            for flag in red_flags:
                logger.error(f"  ❌ {flag}")
            logger.error("\nTHESE FEATURES MAY CONTAIN FUTURE INFORMATION!")
            logger.error("Model results are NOT TRUSTWORTHY if leakage exists!")
            logger.error("=" * 80)
        else:
            logger.info(f"\n✅ No obvious data leakage detected in top {top_n} features")
        
        return feature_importance, red_flags
    
    def calculate_betting_decision(self, predicted_prob, confidence_threshold=0.65):
        """
        Betting decision based on predicted probability.
        
        Args:
            predicted_prob: Model's predicted probability that THIS TEAM covers
            confidence_threshold: Minimum probability to bet (default 0.65 = 65%)
        
        Returns:
            'bet_this_team', 'bet_opponent', or 'no_bet'
        
        Logic:
        - If P(team covers) > threshold: Bet this team
        - If P(team covers) < (1 - threshold): Bet opponent  
        - Otherwise: No bet (not confident enough)
        """
        if predicted_prob > confidence_threshold:
            return 'bet_this_team'
        elif predicted_prob < (1 - confidence_threshold):
            return 'bet_opponent'
        else:
            return 'no_bet'
    
    def evaluate_betting_performance(self, predictions_df, confidence_threshold=0.65, 
                                    verbose=True):
        """
        Evaluate betting performance.
        
        predictions_df must have:
        - predicted_prob: Model's probability that this team covers
        - spread_cover: Actual result (1 = team covered, 0 = didn't)
        """
        results = []
        
        for _, row in predictions_df.iterrows():
            decision = self.calculate_betting_decision(
                row['predicted_prob'], 
                confidence_threshold
            )
            
            if decision == 'no_bet':
                results.append({
                    'decision': 'no_bet',
                    'correct': None,
                    'profit': 0
                })
                continue
            
            # Actual result
            team_covered = row['spread_cover'] == 1
            
            # Did our bet win?
            if decision == 'bet_this_team':
                bet_won = team_covered
            else:  # bet_opponent
                bet_won = not team_covered
            
            # Standard -110 odds
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
            
            if verbose:
                logger.info(f"\n{'='*80}")
                logger.info("📊 BETTING PERFORMANCE")
                logger.info(f"{'='*80}")
                logger.info(f"Confidence threshold: {confidence_threshold:.1%}")
                logger.info(f"Total games: {len(predictions_df):,}")
                logger.info(f"Bets made: {len(bets_made):,} ({len(bets_made)/len(predictions_df):.1%})")
                logger.info(f"Win rate: {win_rate:.1%} (need 52.4% to profit)")
                logger.info(f"Total profit: ${total_profit:,.0f}")
                logger.info(f"ROI: {roi:+.2f}%")
                logger.info(f"Profit per bet: ${total_profit/len(bets_made):.2f}")
                logger.info(f"{'='*80}")
        else:
            win_rate = 0
            total_profit = 0
            roi = 0
        
        return {
            'total_games': len(predictions_df),
            'bets_made': len(bets_made),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi': roi,
            'confidence_threshold': confidence_threshold
        }
    
    def walk_forward_validation(self, df, feature_cols, test_seasons=None):
        """
        Walk-forward validation with NO LOOK-AHEAD BIAS.
        
        For each test season N:
        1. Train on: seasons < N-1
        2. Validate on: season N-1 (optimize threshold here)
        3. Test on: season N (apply threshold, never seen before)
        """
        if test_seasons is None:
            all_seasons = sorted(df['season'].unique())
            # Use last 5 seasons as test, but need at least 3 years of training
            if len(all_seasons) >= 8:
                test_seasons = all_seasons[-5:]
            else:
                logger.warning(f"Only {len(all_seasons)} seasons available")
                test_seasons = all_seasons[-min(5, len(all_seasons)):]
        
        logger.info("\n" + "="*80)
        logger.info("🔄 WALK-FORWARD VALIDATION (NO CHEATING)")
        logger.info("="*80)
        logger.info(f"Test seasons: {test_seasons}")
        logger.info(f"This ensures NO future data leaks into predictions!")
        
        all_predictions = []
        threshold_history = []
        all_red_flags = []
        
        for test_season in test_seasons:
            logger.info(f"\n{'='*80}")
            logger.info(f"TESTING SEASON {test_season}")
            logger.info(f"{'='*80}")
            
            val_season = test_season - 1
            train_df = df[df['season'] < val_season].copy()
            val_df = df[df['season'] == val_season].copy()
            test_df = df[df['season'] == test_season].copy()
            
            logger.info(f"Train: seasons < {val_season} ({len(train_df):,} rows)")
            logger.info(f"Val:   season {val_season} ({len(val_df):,} rows)")
            logger.info(f"Test:  season {test_season} ({len(test_df):,} rows)")
            
            if len(test_df) == 0:
                logger.warning(f"No test data for {test_season}, skipping")
                continue
            
            if len(train_df) < 1000:
                logger.warning(f"Very little training data ({len(train_df)} rows)")
            
            # Prepare features
            numeric_cols = train_df[feature_cols].select_dtypes(
                include=[np.number]).columns.tolist()
            categorical_cols = train_df[feature_cols].select_dtypes(
                include=['category', 'object']).columns.tolist()
            
            # Training data
            X_train_numeric = train_df[numeric_cols].fillna(0).replace(
                [np.inf, -np.inf], 0)
            y_train = train_df['spread_cover'].astype(int)
            
            # Test data (always needed)
            X_test_numeric = test_df[numeric_cols].fillna(0).replace(
                [np.inf, -np.inf], 0)
            y_test = test_df['spread_cover'].astype(int)
            
            # Validation data (if available)
            has_validation = len(val_df) > 0
            if has_validation:
                X_val_numeric = val_df[numeric_cols].fillna(0).replace(
                    [np.inf, -np.inf], 0)
                y_val = val_df['spread_cover'].astype(int)
            
            # Handle categorical features
            if categorical_cols:
                X_train_cat = train_df[categorical_cols].apply(
                    lambda x: x.cat.codes if x.dtype.name == 'category' else x)
                X_test_cat = test_df[categorical_cols].apply(
                    lambda x: x.cat.codes if x.dtype.name == 'category' else x)
                X_train = pd.concat([X_train_numeric, X_train_cat], axis=1)
                X_test = pd.concat([X_test_numeric, X_test_cat], axis=1)
                
                if has_validation:
                    X_val_cat = val_df[categorical_cols].apply(
                        lambda x: x.cat.codes if x.dtype.name == 'category' else x)
                    X_val = pd.concat([X_val_numeric, X_val_cat], axis=1)
            else:
                X_train = X_train_numeric
                X_test = X_test_numeric
                if has_validation:
                    X_val = X_val_numeric
            
            # Train model
            logger.info("\n🎯 Training LightGBM model...")
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
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
            
            if has_validation:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                model = lgb.train(
                    params, train_data, num_boost_round=500,
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'valid'],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            else:
                logger.warning("No validation set - using train for early stopping")
                model = lgb.train(
                    params, train_data, num_boost_round=300,
                    valid_sets=[train_data],
                    valid_names=['train'],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            
            self.models[test_season] = model
            
            # 🔍 DETECT DATA LEAKAGE
            feature_importance, red_flags = self.detect_data_leakage(
                model, X_train.columns.tolist(), X_train, y_train, test_season
            )
            if red_flags:
                all_red_flags.extend([(test_season, flag) for flag in red_flags])
            
            # Optimize threshold on VALIDATION set
            if has_validation:
                logger.info("\n📈 Optimizing threshold on VALIDATION data...")
                y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
                
                val_results = val_df[[
                    'gameId', 'startDate', 'team', 'opponent', 'team_spread'
                ]].copy()
                val_results['spread_cover'] = y_val.values
                val_results['predicted_prob'] = y_val_pred
                
                # Try different thresholds
                thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
                best_roi = -float('inf')
                optimal_threshold = 0.65
                
                for threshold in thresholds:
                    perf = self.evaluate_betting_performance(
                        val_results.copy(), threshold, verbose=False
                    )
                    if perf['bets_made'] >= 20 and perf['roi'] > best_roi:
                        best_roi = perf['roi']
                        optimal_threshold = threshold
                
                logger.info(f"✅ Optimal threshold: {optimal_threshold:.1%}")
                logger.info(f"   Validation ROI: {best_roi:+.2f}%")
                logger.info(f"   (This was optimized on season {val_season} data only)")
                
                threshold_history.append({
                    'test_season': test_season,
                    'val_season': val_season,
                    'optimal_threshold': optimal_threshold,
                    'val_roi': best_roi
                })
            else:
                optimal_threshold = 0.65
                logger.warning(f"Using default threshold: {optimal_threshold:.1%}")
            
            # Apply to TEST set (never seen before!)
            logger.info(f"\n🎲 Applying model to TEST season {test_season}...")
            y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
            
            # Store predictions
            season_results = test_df[[
                'gameId', 'startDate', 'team', 'opponent', 'team_spread'
            ]].copy()
            season_results['spread_cover'] = y_test.values
            season_results['predicted_prob'] = y_test_pred
            season_results['season'] = test_season
            season_results['optimal_threshold'] = optimal_threshold
            
            # Evaluate
            test_perf = self.evaluate_betting_performance(
                season_results.copy(), optimal_threshold, verbose=True
            )
            
            all_predictions.append(season_results)
        
        # Final check for data leakage
        if all_red_flags:
            logger.error("\n" + "="*80)
            logger.error("🚨 CRITICAL: DATA LEAKAGE DETECTED ACROSS SEASONS")
            logger.error("="*80)
            for season, flag in all_red_flags[:10]:  # Show first 10
                logger.error(f"Season {season}: {flag}")
            logger.error("\n⚠️ MODEL RESULTS ARE NOT TRUSTWORTHY!")
            logger.error("⚠️ These features contain future information!")
            logger.error("="*80)
        
        # Combine predictions
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Save
        pred_path = self.output_dir / 'predictions.parquet'
        predictions_df.to_parquet(pred_path, index=False)
        logger.info(f"\n💾 Saved predictions to {pred_path}")
        
        if threshold_history:
            threshold_df = pd.DataFrame(threshold_history)
            threshold_path = self.output_dir / 'threshold_history.csv'
            threshold_df.to_csv(threshold_path, index=False)
            logger.info(f"💾 Saved threshold history to {threshold_path}")
        else:
            threshold_df = pd.DataFrame()
        
        return predictions_df, threshold_df, all_red_flags
    
    def run(self, test_seasons=None):
        """Execute full pipeline."""
        logger.info("\n" + "="*80)
        logger.info("🎰 SPREAD BETTING MODEL - ZERO TOLERANCE FOR CHEATING")
        logger.info("="*80)
        
        df = self.load_and_prepare_data()
        feature_cols = self.get_feature_columns(df)
        
        # Run validation
        predictions_df, threshold_df, red_flags = self.walk_forward_validation(
            df, feature_cols, test_seasons
        )
        
        # Overall results
        logger.info("\n" + "="*80)
        logger.info("📊 OVERALL RESULTS")
        logger.info("="*80)
        
        season_results = []
        for season in sorted(predictions_df['season'].unique()):
            season_df = predictions_df[predictions_df['season'] == season].copy()
            threshold = season_df['optimal_threshold'].iloc[0]
            
            perf = self.evaluate_betting_performance(
                season_df, threshold, verbose=False
            )
            perf['season'] = season
            season_results.append(perf)
            
            logger.info(f"\nSeason {season} (threshold={threshold:.1%}):")
            logger.info(f"  Win Rate: {perf['win_rate']:.1%}")
            logger.info(f"  ROI: {perf['roi']:+.2f}%")
            logger.info(f"  Profit: ${perf['total_profit']:,.0f}")
            logger.info(f"  Bets: {perf['bets_made']:,}")
        
        # Aggregate
        total_profit = sum(r['total_profit'] for r in season_results)
        total_bets = sum(r['bets_made'] for r in season_results)
        total_wins = sum(r['bets_made'] * r['win_rate'] for r in season_results)
        overall_win_rate = total_wins / total_bets if total_bets > 0 else 0
        overall_roi = (total_profit / (total_bets * 110)) * 100 if total_bets > 0 else 0
        
        logger.info("\n" + "="*80)
        logger.info("🎯 AGGREGATE PERFORMANCE")
        logger.info("="*80)
        logger.info(f"Total Bets: {total_bets:,}")
        logger.info(f"Win Rate: {overall_win_rate:.1%}")
        logger.info(f"ROI: {overall_roi:+.2f}%")
        logger.info(f"Total Profit: ${total_profit:,.0f}")
        
        if overall_roi > 5:
            logger.info("\n✅ STRONG RESULTS - But verify no data leakage above!")
        elif overall_roi > 0:
            logger.info("\n✅ PROFITABLE - Modest edge detected")
        else:
            logger.info("\n⚠️ NOT PROFITABLE - Model needs improvement")
        
        logger.info("="*80)
        
        # Final warning about data leakage
        if red_flags:
            logger.error("\n🚨 WARNING: Data leakage was detected!")
            logger.error("Review the leakage detection output above.")
            logger.error("Results may be artificially inflated.")
        
        return {
            'overall_roi': overall_roi,
            'overall_win_rate': overall_win_rate,
            'total_profit': total_profit,
            'total_bets': total_bets,
            'season_results': season_results,
            'threshold_history': threshold_df,
            'data_leakage_detected': len(red_flags) > 0,
            'red_flags': red_flags
        }


if __name__ == "__main__":
    DATA_PATH = 'feature_output/spread_training_data.parquet'
    OUTPUT_DIR = 'spread_model_output'
    
    model = SpreadBettingModel(data_path=DATA_PATH, output_dir=OUTPUT_DIR)
    results = model.run()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"Overall ROI: {results['overall_roi']:+.2f}%")
    print(f"Win Rate: {results['overall_win_rate']:.1%}")
    print(f"Total Profit: ${results['total_profit']:,.0f}")
    
    if results['data_leakage_detected']:
        print("\n🚨 DATA LEAKAGE DETECTED - Results not trustworthy!")
    else:
        print("\n✅ No obvious data leakage detected")
    
    print("="*80)