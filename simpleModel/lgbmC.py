import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedSpreadBettingModel:
    """
    Fixed spread betting model with critical improvements:
    1. Conservative bankroll management (5% max Kelly)
    2. Minimum edge threshold (2.5% required to bet)
    3. Predicts spread ERROR instead of raw margin
    4. Better volatility calibration
    """
    
    def __init__(self, data_path, output_dir='./spread_model_output', 
                initial_bankroll=100, max_kelly_fraction=0.05,
                min_edge_threshold=0.05):  # INCREASED from 0.025 to 0.045
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Models
        self.mean_models = {}
        self.q16_models = {}
        self.q84_models = {}
        self.conformal_quantiles = {}  # NEW: Store conformal quantiles per season
        
        # Bankroll management
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_kelly_fraction = max_kelly_fraction
        self.min_edge_threshold = min_edge_threshold
        
        # Performance tracking
        self.betting_history = []
        self.push_history = []
        self.volatility_calibration = {}
        self.recent_performance = []  # NEW: Track recent bet outcomes


    def get_conformal_interval(self, calibration_residuals, alpha=0.1):
        """
        Calculate conformal prediction quantile from calibration residuals.
        Guarantees (1-alpha) coverage on exchangeable data.
        
        Args:
            calibration_residuals: Array of absolute residuals from calibration set
            alpha: Miscoverage rate (0.1 = 90% coverage intervals)
        
        Returns:
            quantile: The residual quantile for prediction intervals
        """
        n = len(calibration_residuals)
        quantile_level = np.ceil((1 - alpha) * (n + 1)) / n
        quantile = np.quantile(calibration_residuals, quantile_level)
        return quantile


    def _calibrate_conformal_on_holdout(self, calibration_df, mean_model, q16_model, q84_model, 
                                        feature_cols, numeric_cols, categorical_cols):
        """
        Use separate calibration set for conformal prediction intervals.
        This provides distribution-free coverage guarantees.
        """
        if len(calibration_df) < 30:
            logger.warning("Insufficient calibration data - skipping conformal calibration")
            return None
        
        # Prepare calibration features
        X_cal = self._prepare_features(calibration_df, numeric_cols, categorical_cols)
        y_cal_error = calibration_df['mov_target'].values
        
        # Get predictions
        cal_pred_error = mean_model.predict(X_cal, num_iteration=mean_model.best_iteration)
        
        # Calculate absolute residuals
        cal_residuals = np.abs(y_cal_error - cal_pred_error)
        
        # Get conformal quantile (90% coverage)
        conformal_q90 = self.get_conformal_interval(cal_residuals, alpha=0.10)
        
        logger.info(f"  Conformal calibration: 90% quantile = {conformal_q90:.2f} points")
        logger.info(f"  Calibration residuals: mean={cal_residuals.mean():.2f}, "
                f"median={np.median(cal_residuals):.2f}, max={cal_residuals.max():.2f}")
        
        return conformal_q90


    def _combine_quantile_and_conformal(self, quantile_std, conformal_quantile, blend_factor=0.6):
        """
        Combine quantile-based std with conformal prediction for robust uncertainty estimates.
        
        Args:
            quantile_std: Std from Q84-Q16 spread
            conformal_quantile: 90% residual quantile from conformal prediction
            blend_factor: Weight for quantile method (0=all conformal, 1=all quantile)
        
        Returns:
            combined_std: Blended uncertainty estimate
        """
        # Convert conformal quantile (90% coverage) to std equivalent
        # For normal distribution, 90% = ±1.645 std
        conformal_std = conformal_quantile / 1.645
        
        # Blend the two estimates
        combined_std = blend_factor * quantile_std + (1 - blend_factor) * conformal_std
        
        return combined_std
        
    def load_and_prepare_data(self):
        """Load data and engineer features focused on spread ERROR prediction"""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        df = df.drop_duplicates(subset=['gameId'], keep='first')
        logger.info(f"Total rows: {len(df):,}")
        
        # Core temporal features
        df['date'] = pd.to_datetime(df['startDate'])
        df['season'] = df['date'].dt.year
        df.loc[df['date'].dt.month >= 7, 'season'] += 1
        df['season_numeric'] = df['season'] - df['season'].min()
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['week_of_season'] = ((df['date'] - df.groupby('season')['date'].transform('min')).dt.days / 7).astype(int)
        
        # CRITICAL CHANGE: Target is spread ERROR, not raw margin
        df['spread_error'] = df['actual_margin'] - df['team_spread']
        df['mov_target'] = df['spread_error']  # Predict the market's error
        
        df['is_push'] = (np.abs(df['actual_margin'] - df['team_spread']) < 0.01).astype(int)
        push_rate = df['is_push'].mean()
        logger.info(f"Historical push rate: {push_rate:.2%}")
        
        # Market efficiency indicators
        if 'homeMoneyline' in df.columns and 'awayMoneyline' in df.columns:
            df['home_implied_prob'] = self._moneyline_to_prob(df['homeMoneyline'])
            df['away_implied_prob'] = self._moneyline_to_prob(df['awayMoneyline'])
            df['market_inefficiency'] = abs(df['home_implied_prob'] + df['away_implied_prob'] - 1.0)
            df['value_indicator'] = df['market_inefficiency'] * abs(df['team_spread'])
        # Enhanced features
        df = self._add_market_aware_features(df)
        df = self._add_volatility_features(df)
        
        logger.info(f"Features engineered. Total: {len(df.columns)}")
        
        return df
    
    def _moneyline_to_prob(self, moneyline):
        """Convert moneyline odds to implied probability"""
        return np.where(moneyline < 0, 
                       -moneyline / (-moneyline + 100),
                       100 / (moneyline + 100))
    
    def _add_market_aware_features(self, df):
        """Add features that respect market efficiency"""
        df = df.sort_values(['team', 'date'])
        
        for window in [5, 10, 20]:
            # --- FIX HERE: Add 'season' to groupby ---
            df[f'team_spread_bias_L{window}'] = (
                df.groupby(['season', 'team'])['spread_error']
                .transform(lambda x: x.rolling(window, min_periods=3).mean().shift(1))
            ).fillna(0)
            
            df[f'team_cover_rate_L{window}'] = (
                df.groupby(['season', 'team'])
                .apply(lambda x: ((x['spread_error'] > 0).rolling(window, min_periods=3).mean().shift(1)))
                .reset_index(level=[0,1], drop=True) # Reset multi-index
            ).fillna(0.5)
            
            df[f'opp_spread_bias_L{window}'] = (
                df.groupby(['season', 'opponent'])['spread_error']
                .transform(lambda x: -x.rolling(window, min_periods=3).mean().shift(1))
            ).fillna(0)
        
        # Spread magnitude features
        df['spread_magnitude'] = abs(df['team_spread'])
        df['is_close_spread'] = (df['spread_magnitude'] < 3).astype(int)
        df['is_large_spread'] = (df['spread_magnitude'] > 10).astype(int)
        
        # Historical performance in similar spreads
        df['spread_bucket'] = pd.cut(df['team_spread'], 
                                     bins=[-100, -14, -7, -3, 3, 7, 14, 100],
                                     labels=['huge_dog', 'big_dog', 'dog', 'close', 'fav', 'big_fav', 'huge_fav'])
        
        df['team_performance_in_bucket'] = (
            df.groupby(['team', 'spread_bucket'])['spread_error']
            .transform(lambda x: x.rolling(20, min_periods=3).mean().shift(1))
        ).fillna(0)
        
        return df
    
    def _add_volatility_features(self, df):
        """Add features predictive of game volatility"""
        # Recent volatility
        for window in [10, 20]:
            # --- FIX HERE: Add 'season' to groupby ---
            df[f'team_spread_error_std_L{window}'] = (
                df.groupby(['season', 'team'])['spread_error'] # <-- ADD 'season'
                .transform(lambda x: x.rolling(window, min_periods=3).std().shift(1))
            ).fillna(7)
        
        # Matchup volatility
        df['is_primetime'] = df['day_of_week'].isin([4, 5, 6, 0]).astype(int)
        
        # Situational volatility
        # --- FIX HERE: Add 'season' to groupby ---
        df['situational_volatility'] = df.groupby(['season', 'team', 'is_home'])['spread_error'].transform( # <-- ADD 'season'
            lambda x: x.rolling(20, min_periods=5).std().shift(1)
        ).fillna(7)
        
        return df

    def _calibrate_volatility_on_validation(self, val_predictions_df):
        """
        Learn a volatility inflation factor from validation set.
        
        We compare predicted probabilities to actual outcomes and find
        the multiplier that makes our confidence intervals match reality.
        """
        val_bets = val_predictions_df[val_predictions_df['bet_made']].copy()
        
        if len(val_bets) == 0:
            logger.warning("No validation bets for calibration")
            return 1.0  # No adjustment
        
        # For each bet, we predicted an edge. Did we win at that rate?
        decided = val_bets[val_bets['bet_correct'].notna()].copy()
        
        if len(decided) < 50:
            logger.warning(f"Only {len(decided)} validation bets - using default calibration")
            return 1.5  # Conservative default: inflate volatility by 50%
        
        # Calculate calibration error across edge buckets
        decided['edge_bin'] = pd.cut(
            decided['edge'] * 100,
            bins=[0, 3, 5, 7, 10, 100],
            labels=['0-3%', '3-5%', '5-7%', '7-10%', '10%+']
        )
        
        gaps = []
        for edge_bin in ['0-3%', '3-5%', '5-7%', '7-10%', '10%+']:
            bin_data = decided[decided['edge_bin'] == edge_bin]
            if len(bin_data) > 10:  # Need reasonable sample
                wins = bin_data['bet_correct'].astype(bool).sum()
                actual_wr = wins / len(bin_data)
                expected_wr = 0.5238 + bin_data['edge'].mean()
                gap = expected_wr - actual_wr  # Positive = we're overconfident
                gaps.append(gap)
        
        if not gaps:
            return 1.5
        
        avg_gap = np.mean(gaps)
        
        # Convert gap to volatility multiplier
        # If we're 5% overconfident, we need ~1.4x more volatility
        # If we're 10% overconfident, we need ~2.0x more volatility
        if avg_gap < 0.01:
            multiplier = 1.0  # Well calibrated
        elif avg_gap < 0.03:
            multiplier = 1.2
        elif avg_gap < 0.05:
            multiplier = 1.4
        elif avg_gap < 0.08:
            multiplier = 1.6
        else:
            multiplier = 1.8
        
        logger.info(f"  Calibration: avg gap = {avg_gap:.1%}, multiplier = {multiplier:.2f}x")
        
        return multiplier
    
    def _train_quantile_model(self, X_train, y_train, X_val=None, y_val=None, quantile=0.5, sample_weights=None):
        """
        Train a model to predict a specific quantile of the spread error distribution.
        
        This replaces the MAE-based volatility model with direct quantile prediction.
        """
        params = {
            'objective': 'quantile',
            'metric': 'quantile',
            'alpha': quantile,  # The quantile to predict
            'boosting_type': 'gbdt',
            'learning_rate': 0.02,
            'num_leaves': 31,
            'max_depth': 5,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'verbose': -1,
            'seed': 42 + int(quantile * 100)  # Different seed per quantile
        }
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        
        if X_val is not None and len(X_val) > 0:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            model = lgb.train(
                params, train_data, num_boost_round=500,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
        else:
            model = lgb.train(
                params, train_data, num_boost_round=300
            )
        
        return model
        
    def get_feature_columns(self, df):
        """Get valid feature columns for modeling"""
        forbidden_cols = [
            'gameId', 'startDate', 'date', 'team', 'opponent', 'season',
            'points', 'opp_points', 'won', 'actual_margin', 'mov_target',
            'spread_target', 'spread_cover', 'is_push', 'spread_error',
            'spread_bucket'  # categorical that needs special handling
        ]
        
        feature_cols = [col for col in df.columns if col not in forbidden_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df[feature_cols].select_dtypes(include=['category', 'object']).columns.tolist()
        
        logger.info(f"\n{'='*80}")
        logger.info("SPREAD ERROR MODEL FEATURES")
        logger.info(f"{'='*80}")
        logger.info(f"Total features: {len(feature_cols)}")
        logger.info(f"  - Numeric: {len(numeric_cols)}")
        logger.info(f"  - Categorical: {len(categorical_cols)}")
        
        return feature_cols, numeric_cols, categorical_cols
    
    def _train_mean_model(self, X_train, y_train, X_val=None, y_val=None, sample_weights=None):
        """Train spread error prediction model with optional sample weights"""
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'learning_rate': 0.02,
            'num_leaves': 31,
            'max_depth': 5,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'verbose': -1,
            'seed': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        
        if X_val is not None and len(X_val) > 0:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            model = lgb.train(
                params, train_data, num_boost_round=500,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
        else:
            model = lgb.train(
                params, train_data, num_boost_round=300
            )
        
        return model
    
    def _train_volatility_model(self, X_train, y_train_residuals, X_val=None, y_val_residuals=None):
        """Train volatility prediction model"""
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'learning_rate': 0.02,
            'num_leaves': 31,
            'max_depth': 5,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.6,
            'bagging_freq': 5,
            'lambda_l1': 2.0,
            'lambda_l2': 2.0,
            'verbose': -1,
            'seed': 123
        }
        
        train_data = lgb.Dataset(X_train, label=y_train_residuals)
        
        if X_val is not None and len(X_val) > 0:
            val_data = lgb.Dataset(X_val, label=y_val_residuals, reference=train_data)
            model = lgb.train(
                params, train_data, num_boost_round=300,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
        else:
            model = lgb.train(
                params, train_data, num_boost_round=200
            )
        
        return model
    
    def make_betting_decision(self, mov_mean, mov_std, spread, current_bankroll=None, 
                          edge_percentile=None):
        """
        IMPROVED: More conservative betting with adaptive Kelly and performance-based adjustments
        """
        if current_bankroll is None:
            current_bankroll = self.current_bankroll
        
        # Safety checks
        if mov_std <= 0 or current_bankroll < 1.0:
            return self._no_bet_decision(mov_mean, mov_std, 'invalid_inputs')
        
        # Calculate probabilities
        team_cover_prob = 1 - norm.cdf(-spread, loc=mov_mean, scale=mov_std)
        opp_cover_prob = norm.cdf(-spread, loc=mov_mean, scale=mov_std)
        
        # Push probability
        push_window = 0.5
        push_prob = norm.cdf(-spread + push_window, loc=mov_mean, scale=mov_std) - \
                norm.cdf(-spread - push_window, loc=mov_mean, scale=mov_std)
        
        # Fair probability at -110 odds
        fair_prob = 110/210  # ~0.5238
        
        # Calculate edges
        team_edge = team_cover_prob - fair_prob
        opp_edge = opp_cover_prob - fair_prob
        
        # Determine best bet
        if team_edge > self.min_edge_threshold and team_edge > opp_edge:
            edge = team_edge
            cover_prob = team_cover_prob
            decision_type = 'bet_team_cover'
        elif opp_edge > self.min_edge_threshold:
            edge = opp_edge
            cover_prob = opp_cover_prob
            decision_type = 'bet_opponent_cover'
        else:
            return self._no_bet_decision(mov_mean, mov_std, 'edge_below_threshold',
                                        team_edge, opp_edge, max(team_edge, opp_edge),
                                        team_cover_prob, push_prob)
        
        # Calculate base Kelly fraction
        p = cover_prob * (1 - push_prob)
        q = (1 - cover_prob) * (1 - push_prob)
        b = 10/11  # Payout ratio for -110 odds
        
        kelly_fraction = (p * b - q) / b
        
        if kelly_fraction <= 0:
            return self._no_bet_decision(mov_mean, mov_std, 'negative_kelly',
                                        team_edge, opp_edge, edge, cover_prob, push_prob)
        
        # IMPROVED: Multi-factor Kelly adjustments
        kelly_multiplier = 1.0
        
        # 1. Edge-based confidence scaling (most important)
        if edge < 0.055:  # Edges under 5.5%
            kelly_multiplier *= 0.25  # Very conservative
        elif edge < 0.075:  # Edges 5.5-7.5%
            kelly_multiplier *= 0.40
        elif edge < 0.10:  # Edges 7.5-10%
            kelly_multiplier *= 0.60
        # else: edges 10%+ use full multiplier
        
        # 2. Volatility-based adjustment
        if mov_std > 18:  # Very high volatility
            kelly_multiplier *= 0.4
        elif mov_std > 15:  # High volatility
            kelly_multiplier *= 0.6
        elif mov_std < 8:  # Suspiciously low volatility - likely underestimated
            kelly_multiplier *= 0.25
        elif mov_std < 11:
            kelly_multiplier *= 0.5
        
        # 3. Edge percentile adjustment (if provided)
        # This measures how unusual this edge is compared to historical edges
        if edge_percentile is not None:
            if edge_percentile < 0.70:  # Bottom 70% of edges
                kelly_multiplier *= 0.5  # Be more conservative
            elif edge_percentile > 0.90:  # Top 10% of edges
                kelly_multiplier *= 1.2  # Slightly more aggressive (but still capped)
        
        # 4. Recent performance adjustment
        if len(self.recent_performance) >= 20:
            recent_roi = sum([x['profit'] for x in self.recent_performance[-20:]]) / \
                        sum([x['bet_amount'] for x in self.recent_performance[-20:]])
            
            if recent_roi < -0.15:  # 15%+ drawdown
                kelly_multiplier *= 0.3
            elif recent_roi < -0.05:  # 5-15% drawdown
                kelly_multiplier *= 0.6
            elif recent_roi > 0.10:  # 10%+ hot streak - be careful of variance
                kelly_multiplier *= 0.8
        
        # Apply all adjustments
        kelly_fraction *= kelly_multiplier
        
        # Cap at maximum
        kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
        
        # Calculate bet amount
        bet_amount = kelly_fraction * current_bankroll
        
        # Minimum bet threshold - scale with bankroll
        min_bet = max(current_bankroll * 0.01, 1.0)
        if bet_amount < min_bet:
            return self._no_bet_decision(mov_mean, mov_std, 'bet_too_small',
                                        team_edge, opp_edge, edge, cover_prob, push_prob)
        
        return {
            'decision': decision_type,
            'reason': 'sufficient_edge',
            'bet_amount': bet_amount,
            'kelly_fraction': kelly_fraction,
            'kelly_multiplier': kelly_multiplier,
            'cover_prob': cover_prob,
            'push_prob': push_prob,
            'mov_mean': mov_mean,
            'mov_std': mov_std,
            'edge': edge,
            'team_edge': team_edge,
            'opp_edge': opp_edge
        }


    def _no_bet_decision(self, mov_mean, mov_std, reason, team_edge=0, opp_edge=0, 
                        edge=0, cover_prob=0, push_prob=0):
        """Helper to return consistent no-bet decision"""
        return {
            'decision': 'no_bet',
            'reason': reason,
            'bet_amount': 0,
            'kelly_fraction': 0,
            'kelly_multiplier': 0,
            'cover_prob': cover_prob,
            'push_prob': push_prob,
            'mov_mean': mov_mean,
            'mov_std': mov_std,
            'edge': edge,
            'team_edge': team_edge,
            'opp_edge': opp_edge
        }


    def _update_recent_performance(self, bet_amount, profit):
        """Track recent performance for adaptive betting"""
        self.recent_performance.append({
            'bet_amount': bet_amount,
            'profit': profit
        })
        # Keep last 50 bets
        if len(self.recent_performance) > 50:
            self.recent_performance.pop(0)

    
    def calculate_bet_outcome(self, decision, actual_margin, spread):
        """Calculate if bet won"""
        if abs(actual_margin - spread) < 0.01:
            return None, 0  # Push
        
        team_covered = (actual_margin + spread) > 0
        
        if decision['decision'] == 'bet_team_cover':
            bet_correct = team_covered
        elif decision['decision'] == 'bet_opponent_cover':
            bet_correct = not team_covered
        else:
            return None, 0  # No bet
        
        if bet_correct:
            profit = decision['bet_amount'] * (100/110)
        else:
            profit = -decision['bet_amount']
        
        return bet_correct, profit

    def _calculate_sample_weights(self, df, test_season):
        """
        Calculate sample weights that emphasize recent data.
        
        Data from recent seasons is more predictive of future performance
        due to rule changes, strategy evolution, and roster turnover.
        """
        # Years before test season
        years_ago = test_season - df['season']
        
        # Exponential decay: half-life of 3 seasons
        # Season 1 year ago: 0.89x weight
        # Season 3 years ago: 0.50x weight
        # Season 5 years ago: 0.25x weight
        weights = np.exp(-0.231 * years_ago)  # ln(2)/3 ≈ 0.231
        
        return weights
    
    def _calibrate_volatility_on_validation(self, val_predictions_df):
        """
        Learn a volatility inflation factor from validation set using continuous optimization.
        
        We compare predicted probabilities to actual outcomes and find
        the multiplier that makes our confidence intervals match reality.
        
        Key improvements:
        - Continuous formula instead of hardcoded thresholds
        - Weighted by sample size (more data = more reliable)
        - Considers both over- and under-confidence
        - Bounded to prevent extreme adjustments
        """
        val_bets = val_predictions_df[val_predictions_df['bet_made']].copy()
        
        if len(val_bets) == 0:
            logger.warning("No validation bets for calibration")
            return 1.0  # No adjustment
        
        # For each bet, we predicted an edge. Did we win at that rate?
        decided = val_bets[val_bets['bet_correct'].notna()].copy()
        
        if len(decided) < 30:
            logger.warning(f"Only {len(decided)} validation bets - using conservative default")
            return 1.5  # Conservative default: inflate volatility by 50%
        
        # Calculate calibration error across edge buckets
        decided['edge_bin'] = pd.cut(
            decided['edge'] * 100,
            bins=[0, 2.5, 3.5, 4.5, 6, 8, 100],
            labels=['0-2.5%', '2.5-3.5%', '3.5-4.5%', '4.5-6%', '6-8%', '8%+']
        )
        
        # Collect weighted gaps
        weighted_gaps = []
        bin_sizes = []
        
        for edge_bin in ['0-2.5%', '2.5-3.5%', '3.5-4.5%', '4.5-6%', '6-8%', '8%+']:
            bin_data = decided[decided['edge_bin'] == edge_bin]
            if len(bin_data) >= 5:  # Need at least 5 samples
                wins = bin_data['bet_correct'].astype(bool).sum()
                actual_wr = wins / len(bin_data)
                expected_wr = 0.5238 + bin_data['edge'].mean()
                gap = expected_wr - actual_wr  # Positive = overconfident
                
                # Weight by sqrt(sample_size) - standard error weighting
                weight = np.sqrt(len(bin_data))
                weighted_gaps.append(gap * weight)
                bin_sizes.append(weight)
        
        if not weighted_gaps or sum(bin_sizes) == 0:
            logger.warning("Insufficient data across bins - using default")
            return 1.5
        
        # Weighted average gap
        avg_gap = sum(weighted_gaps) / sum(bin_sizes)
        
        # IMPROVED: Continuous formula for multiplier
        # Theory: If we're X% overconfident, we underestimated std by factor of ~(1 + k*X)
        # where k is calibrated empirically
        
        if avg_gap < 0:
            # Underconfident - reduce multiplier, but cautiously
            # Don't go below 0.9x (assume we're not THAT good)
            multiplier = max(1.8, min(2.5, 1.0 + avg_gap * 10.0))
        else:
            # Overconfident - increase multiplier more aggressively
            # Map gap to multiplier: 0% → 1.0x, 3% → 1.3x, 5% → 1.5x, 8% → 1.8x, 10%+ → 2.0x
            # Formula: multiplier = 1.0 + gap * 10.0, capped at 2.0
            multiplier = min(2.0, 1.0 + avg_gap * 10.0)
        
        # Additional safety: if sample size is small, be more conservative
        # Shrink toward 1.5 based on confidence
        total_samples = len(decided)
        confidence = min(1.0, total_samples / 100)  # Full confidence at 100+ samples
        multiplier = confidence * multiplier + (1 - confidence) * 1.5
        
        # Log detailed diagnostics
        logger.info(f"  Calibration: avg gap = {avg_gap:.1%} (weighted), "
                   f"n={len(decided)}, multiplier = {multiplier:.2f}x")
        
        return multiplier


    def _evaluate_calibration(self, predictions_df):
        """
        Enhanced calibration evaluation with per-bucket diagnostics.
        
        Shows not just gaps but also confidence intervals to assess statistical significance.
        """
        bets_made = predictions_df[predictions_df['bet_made']].copy()
        
        if len(bets_made) == 0:
            return
        
        # Finer-grained bins for better diagnostics
        bets_made['edge_bin'] = pd.cut(
            bets_made['edge'] * 100,
            bins=[0, 2.5, 3.5, 4.5, 6, 8, 100],
            labels=['0-2.5%', '2.5-3.5%', '3.5-4.5%', '4.5-6%', '6-8%', '8%+']
        )
        
        logger.info("\n📊 CALIBRATION ANALYSIS")
        logger.info("="*80)
        logger.info(f"{'Edge Bucket':<14} {'Count':<8} {'Win Rate':<10} {'Expected':<10} "
                   f"{'Gap':<10} {'95% CI':<12}")
        logger.info("-"*80)
        
        for edge_bin in ['0-2.5%', '2.5-3.5%', '3.5-4.5%', '4.5-6%', '6-8%', '8%+']:
            bin_data = bets_made[bets_made['edge_bin'] == edge_bin]
            decided = bin_data[bin_data['bet_correct'].notna()]
            
            if len(decided) > 0:
                wins = decided['bet_correct'].astype(bool).sum()
                actual_wr = wins / len(decided)
                avg_edge = bin_data['edge'].mean()
                expected_wr = 0.5238 + avg_edge
                gap = actual_wr - expected_wr
                
                # Calculate 95% confidence interval for win rate
                # Using Wilson score interval (more accurate than normal approximation)
                from scipy import stats as scipy_stats
                if len(decided) >= 5:
                    ci_low, ci_high = scipy_stats.binom.interval(
                        0.95, len(decided), actual_wr
                    )
                    ci_low /= len(decided)
                    ci_high /= len(decided)
                    ci_range = ci_high - ci_low
                    ci_str = f"±{ci_range/2:.1%}"
                else:
                    ci_str = "N/A"
                
                logger.info(
                    f"{edge_bin:<14} {len(decided):<8} "
                    f"{actual_wr:>8.1%}  {expected_wr:>8.1%}  "
                    f"{gap:>+8.1%}  {ci_str:>12}"
                )
        
        # Overall calibration metric
        decided_all = bets_made[bets_made['bet_correct'].notna()]
        if len(decided_all) > 0:
            overall_wr = decided_all['bet_correct'].astype(bool).sum() / len(decided_all)
            overall_expected = 0.5238 + decided_all['edge'].mean()
            overall_gap = overall_wr - overall_expected
            
            logger.info("-"*80)
            logger.info(f"{'OVERALL':<14} {len(decided_all):<8} "
                       f"{overall_wr:>8.1%}  {overall_expected:>8.1%}  "
                       f"{overall_gap:>+8.1%}")
            
            # Statistical significance test
            # Null hypothesis: actual WR = expected WR
            if overall_gap != 0:
                z_score = overall_gap / np.sqrt(overall_wr * (1-overall_wr) / len(decided_all))
                p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))
                
                if p_value < 0.05:
                    logger.info(f"\n⚠️  Calibration gap is statistically significant (p={p_value:.3f})")
                    if overall_gap < 0:
                        logger.info("   Model is overconfident - consider increasing min_edge_threshold")
                    else:
                        logger.info("   Model is underconfident - consider decreasing multiplier")
                else:
                    logger.info(f"\n✅ Calibration gap not statistically significant (p={p_value:.3f})")
        
    def walk_forward_validation(self, df, feature_cols, numeric_cols, categorical_cols, 
                           test_seasons=None):
        """
        IMPROVED: Walk-forward validation with separate calibration holdout set
        """
        if test_seasons is None:
            all_seasons = sorted(df['season'].unique())
            test_seasons = all_seasons[-5:] if len(all_seasons) >= 8 else all_seasons[-3:]
        
        logger.info("\n" + "="*80)
        logger.info("🎯 IMPROVED SPREAD BETTING MODEL v2")
        logger.info("="*80)
        logger.info(f"Test seasons: {test_seasons}")
        logger.info(f"Max Kelly: {self.max_kelly_fraction:.1%}")
        logger.info(f"Min Edge: {self.min_edge_threshold:.1%}")
        logger.info("New features:")
        logger.info("  - Separate calibration holdout set")
        logger.info("  - Conformal + Quantile uncertainty blending")
        logger.info("  - Adaptive Kelly based on recent performance")
        logger.info("  - Edge-based confidence scaling")
        
        all_predictions = []
        season_bankrolls = {test_seasons[0]: self.initial_bankroll}
        
        for i, test_season in enumerate(test_seasons):
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 SEASON {test_season}")
            logger.info(f"{'='*80}")
            
            self.current_bankroll = season_bankrolls.get(test_season, self.initial_bankroll)
            logger.info(f"Starting bankroll: ${self.current_bankroll:,.2f}")
            
            # IMPROVED: Three-way split with separate calibration set
            val_season = test_season - 1
            cal_season = test_season - 2  # NEW: Separate calibration season
            
            train_df = df[df['season'] < cal_season].copy()
            cal_df = df[df['season'] == cal_season].copy()  # NEW: Calibration set
            val_df = df[df['season'] == val_season].copy()
            test_df = df[df['season'] == test_season].copy()
            
            logger.info(f"Train: < {cal_season} ({len(train_df):,} rows)")
            logger.info(f"Calibration: {cal_season} ({len(cal_df):,} rows)")  # NEW
            logger.info(f"Val: {val_season} ({len(val_df):,} rows)")
            logger.info(f"Test: {test_season} ({len(test_df):,} rows)")
            
            if len(test_df) == 0:
                continue
            
            # Calculate sample weights (emphasize recent data)
            train_weights = self._calculate_sample_weights(train_df, test_season)
            logger.info(f"Sample weights: {train_weights.min():.3f} to {train_weights.max():.3f} "
                    f"(mean: {train_weights.mean():.3f})")
            
            # Prepare features
            X_train = self._prepare_features(train_df, numeric_cols, categorical_cols)
            X_val = self._prepare_features(val_df, numeric_cols, categorical_cols) if len(val_df) > 0 else None
            X_test = self._prepare_features(test_df, numeric_cols, categorical_cols)
            
            # Target is spread ERROR
            y_train_error = train_df['mov_target'].values
            y_val_error = val_df['mov_target'].values if len(val_df) > 0 else None
            y_test_error = test_df['mov_target'].values
            
            # Train models
            logger.info("\n📈 Training spread error model (with recency weighting)...")
            mean_model = self._train_mean_model(
                X_train, y_train_error, X_val, y_val_error, 
                sample_weights=train_weights.values
            )
            self.mean_models[test_season] = mean_model
            
            test_pred_error = mean_model.predict(X_test, num_iteration=mean_model.best_iteration)
            
            logger.info("📊 Training quantile models (with recency weighting)...")
            
            logger.info("  Training Q16 model...")
            q16_model = self._train_quantile_model(
                X_train, y_train_error, X_val, y_val_error, 
                quantile=0.16, sample_weights=train_weights.values
            )
            self.q16_models[test_season] = q16_model
            
            logger.info("  Training Q84 model...")
            q84_model = self._train_quantile_model(
                X_train, y_train_error, X_val, y_val_error, 
                quantile=0.84, sample_weights=train_weights.values
            )
            self.q84_models[test_season] = q84_model
            
            # Get quantile predictions
            test_pred_q16 = q16_model.predict(X_test, num_iteration=q16_model.best_iteration)
            test_pred_q84 = q84_model.predict(X_test, num_iteration=q84_model.best_iteration)
            
            # Calculate raw quantile-based std
            raw_quantile_std = (test_pred_q84 - test_pred_q16) / 2.0
            raw_quantile_std = np.clip(raw_quantile_std, 3.0, 20.0)
            
            # NEW: Calibrate conformal prediction on separate holdout set
            logger.info("🎯 Calibrating conformal prediction on holdout set...")
            conformal_quantile = None
            if len(cal_df) >= 30:
                conformal_quantile = self._calibrate_conformal_on_holdout(
                    cal_df, mean_model, q16_model, q84_model,
                    feature_cols, numeric_cols, categorical_cols
                )
                self.conformal_quantiles[test_season] = conformal_quantile
            
            # IMPROVED: Blend quantile and conformal estimates
            if conformal_quantile is not None:
                # Combine both methods for robustness
                test_pred_std_base = np.array([
                    self._combine_quantile_and_conformal(q_std, conformal_quantile, blend_factor=0.6)
                    for q_std in raw_quantile_std
                ])
                logger.info(f"  Using blended quantile + conformal uncertainty")
            else:
                test_pred_std_base = raw_quantile_std
                logger.info(f"  Using quantile-only uncertainty (no conformal data)")
            
            # CRITICAL: Calibrate on validation set for additional multiplier
            volatility_multiplier = 1.0
            if len(val_df) > 0:
                val_pred_error = mean_model.predict(X_val, num_iteration=mean_model.best_iteration)
                val_pred_q16 = q16_model.predict(X_val, num_iteration=q16_model.best_iteration)
                val_pred_q84 = q84_model.predict(X_val, num_iteration=q84_model.best_iteration)
                val_raw_std = (val_pred_q84 - val_pred_q16) / 2.0
                val_raw_std = np.clip(val_raw_std, 3.0, 20.0)
                
                # Blend with conformal if available
                if conformal_quantile is not None:
                    val_pred_std = np.array([
                        self._combine_quantile_and_conformal(v_std, conformal_quantile, blend_factor=0.6)
                        for v_std in val_raw_std
                    ])
                else:
                    val_pred_std = val_raw_std
                
                val_spreads = val_df['team_spread'].values
                val_pred_mov = val_spreads + val_pred_error
                
                # Make validation betting decisions
                val_results = []
                for idx in range(len(val_df)):
                    decision = self.make_betting_decision(
                        val_pred_mov[idx],
                        val_pred_std[idx],
                        val_spreads[idx],
                        self.current_bankroll
                    )
                    
                    if decision['decision'] != 'no_bet':
                        bet_correct, _ = self.calculate_bet_outcome(
                            decision,
                            val_df['actual_margin'].iloc[idx],
                            val_spreads[idx]
                        )
                        decision['bet_correct'] = bet_correct
                        decision['bet_made'] = True
                    else:
                        decision['bet_made'] = False
                    
                    val_results.append(decision)
                
                val_results_df = pd.DataFrame(val_results)
                volatility_multiplier = self._calibrate_volatility_on_validation(val_results_df)
            else:
                volatility_multiplier = 1.8
                logger.info("  No validation set - using conservative 1.8x multiplier")
            
            # Apply final calibration
            test_pred_std = test_pred_std_base * volatility_multiplier
            test_pred_std = np.clip(test_pred_std, 3.0, 28.0)
            
            # Log metrics
            historical_std = np.std(y_train_error)
            logger.info(f"  Historical std: {historical_std:.2f}")
            logger.info(f"  Final predicted std range: [{test_pred_std.min():.2f}, {test_pred_std.max():.2f}]")
            logger.info(f"  Final predicted std mean: {test_pred_std.mean():.2f}")
            
            # Convert to MOV
            test_spreads = test_df['team_spread'].values
            test_pred_mov = test_spreads + test_pred_error
            
            # Evaluate
            test_mae = np.mean(np.abs(test_pred_error - y_test_error))
            logger.info(f"\n📊 Model Performance:")
            logger.info(f"  Spread Error MAE: {test_mae:.2f} points")
            logger.info(f"  Avg predicted volatility: {test_pred_std.mean():.2f} points")
            
            # Make betting decisions with edge percentiles
            season_results = test_df[[
                'gameId', 'startDate', 'team', 'opponent', 'team_spread', 'actual_margin'
            ]].copy()
            
            season_results['is_push'] = test_df['is_push'].values
            season_results['pred_spread_error'] = test_pred_error
            season_results['mov_pred_mean'] = test_pred_mov
            season_results['mov_pred_std'] = test_pred_std
            season_results['season'] = test_season
            
            # Calculate edge percentiles for adaptive betting
            temp_decisions = []
            for idx in range(len(season_results)):
                row = season_results.iloc[idx]
                temp_dec = self.make_betting_decision(
                    row['mov_pred_mean'], row['mov_pred_std'], 
                    row['team_spread'], self.current_bankroll
                )
                temp_decisions.append(temp_dec['edge'])
            
            edge_percentiles = np.array([
                (np.array(temp_decisions) < edge).mean() 
                for edge in temp_decisions
            ])
            
            # Process each game with edge percentiles
            betting_decisions = []
            running_bankroll = self.current_bankroll
            
            for idx in range(len(season_results)):
                row = season_results.iloc[idx]
                
                # Make decision with edge percentile
                decision = self.make_betting_decision(
                    row['mov_pred_mean'],
                    row['mov_pred_std'],
                    row['team_spread'],
                    running_bankroll,
                    edge_percentile=edge_percentiles[idx]
                )
                
                # Calculate outcome
                if decision['decision'] != 'no_bet':
                    bet_correct, profit = self.calculate_bet_outcome(
                        decision,
                        row['actual_margin'],
                        row['team_spread']
                    )
                    
                    if bet_correct is None:  # Push
                        decision['bet_correct'] = None
                        decision['bet_profit'] = 0
                    else:
                        decision['bet_correct'] = bet_correct
                        decision['bet_profit'] = profit
                        running_bankroll += profit
                        
                        # Update recent performance tracker
                        self._update_recent_performance(decision['bet_amount'], profit)
                        
                        if running_bankroll <= 0:
                            logger.warning(f"Bankrupt at game {idx+1}")
                            running_bankroll = 0
                else:
                    decision['bet_correct'] = None
                    decision['bet_profit'] = 0
                
                decision['running_bankroll'] = running_bankroll
                decision['edge_percentile'] = edge_percentiles[idx]
                betting_decisions.append(decision)
            
            # Add to results
            for key in betting_decisions[0].keys():
                season_results[key] = [d[key] for d in betting_decisions]
            
            season_results['bet_made'] = season_results['decision'] != 'no_bet'
            
            # Evaluate
            self._evaluate_season_performance(season_results, test_season, self.current_bankroll)
            
            # Store next season's bankroll
            if i < len(test_seasons) - 1:
                season_bankrolls[test_seasons[i + 1]] = running_bankroll
            
            all_predictions.append(season_results)
        
        # Combine and save
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        pred_path = self.output_dir / 'improved_predictions_v2.parquet'
        predictions_df.to_parquet(pred_path, index=False)
        logger.info(f"\n💾 Saved predictions to {pred_path}")
        
        return predictions_df
    
    def _prepare_features(self, df, numeric_cols, categorical_cols):
        """Prepare features for modeling"""
        existing_numeric = [col for col in numeric_cols if col in df.columns]
        existing_categorical = [col for col in categorical_cols if col in df.columns]
        
        X_numeric = df[existing_numeric].fillna(0).replace([np.inf, -np.inf], 0)
        
        if existing_categorical:
            X_categorical = df[existing_categorical].apply(
                lambda x: x.cat.codes if x.dtype.name == 'category' else 
                pd.Categorical(x).codes
            )
            X = pd.concat([X_numeric, X_categorical], axis=1)
        else:
            X = X_numeric
        
        return X
    
    def _evaluate_season_performance(self, results_df, season, initial_bankroll):
        """Evaluate season performance"""
        bets_made = results_df[results_df['bet_made']].copy()
        
        if len(bets_made) == 0:
            logger.warning(f"No bets made in season {season}")
            return
        
        # Count outcomes
        wins = bets_made[bets_made['bet_correct'] == True].shape[0]
        losses = bets_made[bets_made['bet_correct'] == False].shape[0]
        pushes = bets_made[bets_made['bet_correct'].isna()].shape[0]
        
        total_bets = len(bets_made)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # Financial metrics
        total_wagered = bets_made['bet_amount'].sum()
        total_profit = bets_made['bet_profit'].sum()
        roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        
        final_bankroll = results_df['running_bankroll'].iloc[-1]
        bankroll_growth = ((final_bankroll - initial_bankroll) / initial_bankroll) * 100
        
        # Average edge of bets made
        avg_edge = bets_made['edge'].mean() * 100
        
        logger.info(f"\n💰 SEASON {season} RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Bets: {total_bets}/{len(results_df)} ({total_bets/len(results_df)*100:.1f}%)")
        logger.info(f"Record: {wins}-{losses}-{pushes}")
        logger.info(f"Win rate: {win_rate:.1%} (need 52.38%)")
        logger.info(f"Avg edge: {avg_edge:.2f}%")
        logger.info(f"ROI: {roi:+.2f}%")
        logger.info(f"Profit: ${total_profit:,.2f}")
        logger.info(f"Bankroll: ${final_bankroll:,.2f} ({bankroll_growth:+.1f}%)")

    def save_models(self, filepath=None):
        """
        Save all trained models and calibration data for future use
        
        Args:
            filepath: Path to save models (default: output_dir/spread_betting_models.pkl)
        """
        if filepath is None:
            filepath = self.output_dir / 'spread_betting_models.pkl'
        
        model_data = {
            'mean_models': self.mean_models,
            'q16_models': self.q16_models,
            'q84_models': self.q84_models,
            'conformal_quantiles': self.conformal_quantiles,
            'volatility_calibration': self.volatility_calibration,
            'initial_bankroll': self.initial_bankroll,
            'max_kelly_fraction': self.max_kelly_fraction,
            'min_edge_threshold': self.min_edge_threshold,
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'seasons_trained': list(self.mean_models.keys())
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"\n💾 Models saved to {filepath}")
        logger.info(f"   Seasons: {model_data['metadata']['seasons_trained']}")
        
        return filepath
        
    def run(self, test_seasons=None, save_models=True):
        """Execute the improved betting pipeline"""
        logger.info("\n" + "="*80)
        logger.info("🚀 IMPROVED SPREAD BETTING MODEL")
        logger.info("="*80)
        logger.info("Key improvements:")
        logger.info("  - Conservative bankroll (5% max Kelly)")
        logger.info("  - Minimum edge threshold (2.5%)")
        logger.info("  - Predicting spread ERROR, not raw margin")
        logger.info("  - Better volatility calibration")
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        feature_cols, numeric_cols, categorical_cols = self.get_feature_columns(df)
        
        # Run validation
        predictions_df = self.walk_forward_validation(
            df, feature_cols, numeric_cols, categorical_cols, test_seasons
        )
        
        # Summary
        self._summarize_overall_performance(predictions_df)
        
        # Save models
        if save_models:
            self.save_models()
        
        return predictions_df

    def _summarize_overall_performance(self, predictions_df):
        """Overall performance summary"""
        bets_made = predictions_df[predictions_df['bet_made']].copy()
        self._evaluate_calibration(predictions_df)
        if len(bets_made) == 0:
            logger.warning("No bets made")
            return
        
        # Metrics
        total_games = len(predictions_df)
        total_bets = len(bets_made)
        
        decided_bets = bets_made[bets_made['bet_correct'].notna()]
        wins = decided_bets['bet_correct'].astype(bool).sum()
        losses = len(decided_bets) - wins
        pushes = total_bets - len(decided_bets)
        win_rate = wins / len(decided_bets) if len(decided_bets) > 0 else 0
        
        total_wagered = bets_made['bet_amount'].sum()
        total_profit = bets_made['bet_profit'].sum()
        roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        
        final_bankroll = predictions_df['running_bankroll'].iloc[-1]
        total_return = ((final_bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
        
        logger.info("\n" + "="*80)
        logger.info("🏆 OVERALL PERFORMANCE")
        logger.info("="*80)
        logger.info(f"Initial: ${self.initial_bankroll:,.2f}")
        logger.info(f"Final: ${final_bankroll:,.2f}")
        logger.info(f"Return: {total_return:+.1f}%")
        
        logger.info(f"\nBetting Summary:")
        logger.info(f"  Games: {total_games:,}")
        logger.info(f"  Bets: {total_bets:,} ({total_bets/total_games*100:.1f}%)")
        logger.info(f"  Record: {wins}-{losses}-{pushes}")
        logger.info(f"  Win rate: {win_rate:.1%} (need 52.38%)")
        logger.info(f"  ROI: {roi:+.2f}%")
        
        # Sharpe ratio
        if len(decided_bets) > 10:
            bet_returns = decided_bets['bet_profit'] / decided_bets['bet_amount']
            sharpe = bet_returns.mean() / bet_returns.std() * np.sqrt(252)
            logger.info(f"  Sharpe: {sharpe:.2f}")
        
        logger.info("\n" + "="*80)
        if win_rate > 0.55:
            logger.info("✅ STRONG - Genuine edge detected")
        elif win_rate > 0.5238:
            logger.info("✅ PROFITABLE - Beating the vig")
        else:
            logger.info("📉 UNPROFITABLE - Need better features")


if __name__ == "__main__":
    DATA_PATH = 'feature_output/spread_training_data.parquet'
    OUTPUT_DIR = 'spread_model_output'
    
    # Initialize improved model with conservative parameters
    model = ImprovedSpreadBettingModel(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        initial_bankroll=100,
        max_kelly_fraction=0.25,  # 5% max (was 25%!)
        min_edge_threshold=0.045  # 2.5% minimum edge required
    )
    
    # Run the model
    results = model.run()
    
    print("\n" + "="*80)
    print("🎯 IMPROVED MODEL COMPLETE")
    print("="*80)
    print("Key changes implemented:")
    print("  1. Conservative betting (5% max Kelly vs 25%)")
    print("  2. Minimum edge requirement (2.5%)")
    print("  3. Predicting spread ERROR instead of raw margin")
    print("  4. Better model regularization")
    print("="*80)