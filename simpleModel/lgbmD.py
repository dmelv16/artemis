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
import optuna
import json
import sys
from optuna.visualization import plot_optimization_history, plot_param_importances

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizableSpreadBettingModel:
    """
    Spread betting model with hyperparameter optimization via Optuna.
    All critical parameters can now be optimized instead of hard-coded.
    """
    
    def __init__(self, data_path, output_dir='./spread_model_output', 
                 initial_bankroll=100, 
                 # Hyperparameters to optimize
                 max_kelly_fraction=0.05,
                 min_edge_threshold=0.05,
                 blend_factor=0.6,
                 conformal_alpha=0.10,
                 num_leaves=31,
                 learning_rate=0.02,
                 feature_windows=None,
                 volatility_multiplier_base=1.5,
                 kelly_edge_threshold_1=0.055,
                 kelly_edge_threshold_2=0.075,
                 kelly_edge_threshold_3=0.10,
                 kelly_multiplier_1=0.25,
                 kelly_multiplier_2=0.40,
                 kelly_multiplier_3=0.60):
        
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Models
        self.mean_models = {}
        self.q16_models = {}
        self.q84_models = {}
        self.conformal_quantiles = {}
        
        # Bankroll management - OPTIMIZABLE
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_kelly_fraction = max_kelly_fraction
        self.min_edge_threshold = min_edge_threshold
        
        # Uncertainty estimation - OPTIMIZABLE
        self.blend_factor = blend_factor
        self.conformal_alpha = conformal_alpha
        self.volatility_multiplier_base = volatility_multiplier_base
        
        # Model hyperparameters - OPTIMIZABLE
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        
        # Feature engineering - OPTIMIZABLE
        self.feature_windows = feature_windows or [5, 10, 20]
        
        # Kelly adjustment thresholds - OPTIMIZABLE
        self.kelly_edge_threshold_1 = kelly_edge_threshold_1
        self.kelly_edge_threshold_2 = kelly_edge_threshold_2
        self.kelly_edge_threshold_3 = kelly_edge_threshold_3
        self.kelly_multiplier_1 = kelly_multiplier_1
        self.kelly_multiplier_2 = kelly_multiplier_2
        self.kelly_multiplier_3 = kelly_multiplier_3
        
        # Performance tracking
        self.betting_history = []
        self.push_history = []
        self.volatility_calibration = {}
        self.recent_performance = []
        
        # Quiet mode for optimization
        self.quiet_mode = False

    def set_quiet_mode(self, quiet=True):
        """Enable/disable logging for optimization runs"""
        self.quiet_mode = quiet
        if quiet:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)

    def get_conformal_interval(self, calibration_residuals):
        """Calculate conformal prediction quantile - uses self.conformal_alpha"""
        n = len(calibration_residuals)
        quantile_level = np.ceil((1 - self.conformal_alpha) * (n + 1)) / n
        quantile = np.quantile(calibration_residuals, quantile_level)
        return quantile

    def _calibrate_conformal_on_holdout(self, calibration_df, mean_model, q16_model, q84_model, 
                                        feature_cols, numeric_cols, categorical_cols):
        """Use separate calibration set for conformal prediction intervals"""
        if len(calibration_df) < 30:
            if not self.quiet_mode:
                logger.warning("Insufficient calibration data - skipping conformal calibration")
            return None
        
        X_cal = self._prepare_features(calibration_df, numeric_cols, categorical_cols)
        y_cal_error = calibration_df['mov_target'].values
        
        cal_pred_error = mean_model.predict(X_cal, num_iteration=mean_model.best_iteration)
        cal_residuals = np.abs(y_cal_error - cal_pred_error)
        
        conformal_q = self.get_conformal_interval(cal_residuals)
        
        if not self.quiet_mode:
            logger.info(f"  Conformal calibration: {(1-self.conformal_alpha)*100:.0f}% quantile = {conformal_q:.2f} points")
        
        return conformal_q

    def _combine_quantile_and_conformal(self, quantile_std, conformal_quantile):
        """Combine using self.blend_factor"""
        conformal_std = conformal_quantile / 1.645
        combined_std = self.blend_factor * quantile_std + (1 - self.blend_factor) * conformal_std
        return combined_std
        
    def load_and_prepare_data(self):
        """Load data and engineer features"""
        if not self.quiet_mode:
            logger.info(f"Loading data from {self.data_path}")
        
        df = pd.read_parquet(self.data_path)
        df = df.drop_duplicates(subset=['gameId'], keep='first')
        
        # Core temporal features
        df['date'] = pd.to_datetime(df['startDate'])
        df['season'] = df['date'].dt.year
        df.loc[df['date'].dt.month >= 7, 'season'] += 1
        df['season_numeric'] = df['season'] - df['season'].min()
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['week_of_season'] = ((df['date'] - df.groupby('season')['date'].transform('min')).dt.days / 7).astype(int)
        
        # Target is spread ERROR
        df['spread_error'] = df['actual_margin'] - df['team_spread']
        df['mov_target'] = df['spread_error']
        df['is_push'] = (np.abs(df['actual_margin'] - df['team_spread']) < 0.01).astype(int)
        
        # Market efficiency indicators
        if 'homeMoneyline' in df.columns and 'awayMoneyline' in df.columns:
            df['home_implied_prob'] = self._moneyline_to_prob(df['homeMoneyline'])
            df['away_implied_prob'] = self._moneyline_to_prob(df['awayMoneyline'])
            df['market_inefficiency'] = abs(df['home_implied_prob'] + df['away_implied_prob'] - 1.0)
            df['value_indicator'] = df['market_inefficiency'] * abs(df['team_spread'])
        
        # Enhanced features with OPTIMIZABLE windows
        df = self._add_market_aware_features(df)
        df = self._add_volatility_features(df)
        
        return df
    
    def _moneyline_to_prob(self, moneyline):
        """Convert moneyline odds to implied probability"""
        return np.where(moneyline < 0, 
                       -moneyline / (-moneyline + 100),
                       100 / (moneyline + 100))
    
    def _add_market_aware_features(self, df):
        """Add features using OPTIMIZABLE windows"""
        df = df.sort_values(['team', 'date'])
        
        # Use self.feature_windows instead of hard-coded [5, 10, 20]
        for window in self.feature_windows:
            df[f'team_spread_bias_L{window}'] = (
                df.groupby(['season', 'team'])['spread_error']
                .transform(lambda x: x.rolling(window, min_periods=3).mean().shift(1))
            ).fillna(0)
            
            df[f'team_cover_rate_L{window}'] = (
                df.groupby(['season', 'team'])
                .apply(lambda x: ((x['spread_error'] > 0).rolling(window, min_periods=3).mean().shift(1)))
                .reset_index(level=[0,1], drop=True)
            ).fillna(0.5)
            
            df[f'opp_spread_bias_L{window}'] = (
                df.groupby(['season', 'opponent'])['spread_error']
                .transform(lambda x: -x.rolling(window, min_periods=3).mean().shift(1))
            ).fillna(0)
        
        # Spread magnitude features
        df['spread_magnitude'] = abs(df['team_spread'])
        df['is_close_spread'] = (df['spread_magnitude'] < 3).astype(int)
        df['is_large_spread'] = (df['spread_magnitude'] > 10).astype(int)
        
        df['spread_bucket'] = pd.cut(df['team_spread'], 
                                     bins=[-100, -14, -7, -3, 3, 7, 14, 100],
                                     labels=['huge_dog', 'big_dog', 'dog', 'close', 'fav', 'big_fav', 'huge_fav'])
        
        df['team_performance_in_bucket'] = (
            df.groupby(['team', 'spread_bucket'])['spread_error']
            .transform(lambda x: x.rolling(20, min_periods=3).mean().shift(1))
        ).fillna(0)
        
        return df
    
    def _add_volatility_features(self, df):
        """Add volatility features using OPTIMIZABLE windows"""
        for window in [10, 20]:
            df[f'team_spread_error_std_L{window}'] = (
                df.groupby(['season', 'team'])['spread_error']
                .transform(lambda x: x.rolling(window, min_periods=3).std().shift(1))
            ).fillna(7)
        
        df['is_primetime'] = df['day_of_week'].isin([4, 5, 6, 0]).astype(int)
        
        df['situational_volatility'] = df.groupby(['season', 'team', 'is_home'])['spread_error'].transform(
            lambda x: x.rolling(20, min_periods=5).std().shift(1)
        ).fillna(7)
        
        return df

    def _calibrate_volatility_on_validation(self, val_predictions_df):
        """Learn volatility multiplier - uses self.volatility_multiplier_base as starting point"""
        val_bets = val_predictions_df[val_predictions_df['bet_made']].copy()
        
        if len(val_bets) == 0:
            return self.volatility_multiplier_base
        
        decided = val_bets[val_bets['bet_correct'].notna()].copy()
        
        if len(decided) < 30:
            return self.volatility_multiplier_base
        
        decided['edge_bin'] = pd.cut(
            decided['edge'] * 100,
            bins=[0, 2.5, 3.5, 4.5, 6, 8, 100],
            labels=['0-2.5%', '2.5-3.5%', '3.5-4.5%', '4.5-6%', '6-8%', '8%+']
        )
        
        weighted_gaps = []
        bin_sizes = []
        
        for edge_bin in ['0-2.5%', '2.5-3.5%', '3.5-4.5%', '4.5-6%', '6-8%', '8%+']:
            bin_data = decided[decided['edge_bin'] == edge_bin]
            if len(bin_data) >= 5:
                wins = bin_data['bet_correct'].astype(bool).sum()
                actual_wr = wins / len(bin_data)
                expected_wr = 0.5238 + bin_data['edge'].mean()
                gap = expected_wr - actual_wr
                
                weight = np.sqrt(len(bin_data))
                weighted_gaps.append(gap * weight)
                bin_sizes.append(weight)
        
        if not weighted_gaps or sum(bin_sizes) == 0:
            return self.volatility_multiplier_base
        
        avg_gap = sum(weighted_gaps) / sum(bin_sizes)
        
        if avg_gap < 0:
            multiplier = max(1.8, min(2.5, 1.0 + avg_gap * 10.0))
        else:
            multiplier = min(2.0, 1.0 + avg_gap * 10.0)
        
        total_samples = len(decided)
        confidence = min(1.0, total_samples / 100)
        multiplier = confidence * multiplier + (1 - confidence) * self.volatility_multiplier_base
        
        return multiplier
    
    def _train_quantile_model(self, X_train, y_train, X_val=None, y_val=None, quantile=0.5, sample_weights=None):
        """Train quantile model with OPTIMIZABLE hyperparameters"""
        params = {
            'objective': 'quantile',
            'metric': 'quantile',
            'alpha': quantile,
            'boosting_type': 'gbdt',
            'learning_rate': self.learning_rate,  # OPTIMIZABLE
            'num_leaves': self.num_leaves,  # OPTIMIZABLE
            'max_depth': 5,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'verbose': -1,
            'seed': 42 + int(quantile * 100)
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
            model = lgb.train(params, train_data, num_boost_round=300)
        
        return model
        
    def get_feature_columns(self, df):
        """Get valid feature columns for modeling"""
        forbidden_cols = [
            'gameId', 'startDate', 'date', 'team', 'opponent', 'season',
            'points', 'opp_points', 'won', 'actual_margin', 'mov_target',
            'spread_target', 'spread_cover', 'is_push', 'spread_error',
            'spread_bucket'
        ]
        
        feature_cols = [col for col in df.columns if col not in forbidden_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df[feature_cols].select_dtypes(include=['category', 'object']).columns.tolist()
        
        return feature_cols, numeric_cols, categorical_cols
    
    def _train_mean_model(self, X_train, y_train, X_val=None, y_val=None, sample_weights=None):
        """Train mean model with OPTIMIZABLE hyperparameters"""
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'learning_rate': self.learning_rate,  # OPTIMIZABLE
            'num_leaves': self.num_leaves,  # OPTIMIZABLE
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
            model = lgb.train(params, train_data, num_boost_round=300)
        
        return model
    
    def make_betting_decision(self, mov_mean, mov_std, spread, current_bankroll=None, 
                              edge_percentile=None):
        """
        Betting decision with OPTIMIZABLE Kelly adjustments
        """
        if current_bankroll is None:
            current_bankroll = self.current_bankroll
        
        if mov_std <= 0 or current_bankroll < 1.0:
            return self._no_bet_decision(mov_mean, mov_std, 'invalid_inputs')
        
        team_cover_prob = 1 - norm.cdf(-spread, loc=mov_mean, scale=mov_std)
        opp_cover_prob = norm.cdf(-spread, loc=mov_mean, scale=mov_std)
        
        push_window = 0.5
        push_prob = norm.cdf(-spread + push_window, loc=mov_mean, scale=mov_std) - \
                    norm.cdf(-spread - push_window, loc=mov_mean, scale=mov_std)
        
        fair_prob = 110/210
        
        team_edge = team_cover_prob - fair_prob
        opp_edge = opp_cover_prob - fair_prob
        
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
        
        p = cover_prob * (1 - push_prob)
        q = (1 - cover_prob) * (1 - push_prob)
        b = 10/11
        
        kelly_fraction = (p * b - q) / b
        
        if kelly_fraction <= 0:
            return self._no_bet_decision(mov_mean, mov_std, 'negative_kelly',
                                        team_edge, opp_edge, edge, cover_prob, push_prob)
        
        # OPTIMIZABLE Kelly adjustments
        kelly_multiplier = 1.0
        
        # Edge-based scaling with OPTIMIZABLE thresholds
        if edge < self.kelly_edge_threshold_1:
            kelly_multiplier *= self.kelly_multiplier_1
        elif edge < self.kelly_edge_threshold_2:
            kelly_multiplier *= self.kelly_multiplier_2
        elif edge < self.kelly_edge_threshold_3:
            kelly_multiplier *= self.kelly_multiplier_3
        
        # Volatility adjustment
        if mov_std > 18:
            kelly_multiplier *= 0.4
        elif mov_std > 15:
            kelly_multiplier *= 0.6
        elif mov_std < 8:
            kelly_multiplier *= 0.25
        elif mov_std < 11:
            kelly_multiplier *= 0.5
        
        if edge_percentile is not None:
            if edge_percentile < 0.70:
                kelly_multiplier *= 0.5
            elif edge_percentile > 0.90:
                kelly_multiplier *= 1.2
        
        if len(self.recent_performance) >= 20:
            recent_roi = sum([x['profit'] for x in self.recent_performance[-20:]]) / \
                        sum([x['bet_amount'] for x in self.recent_performance[-20:]])
            
            if recent_roi < -0.15:
                kelly_multiplier *= 0.3
            elif recent_roi < -0.05:
                kelly_multiplier *= 0.6
            elif recent_roi > 0.10:
                kelly_multiplier *= 0.8
        
        kelly_fraction *= kelly_multiplier
        kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
        
        bet_amount = kelly_fraction * current_bankroll
        
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
        if len(self.recent_performance) > 50:
            self.recent_performance.pop(0)
    
    def calculate_bet_outcome(self, decision, actual_margin, spread):
        """Calculate if bet won"""
        if abs(actual_margin - spread) < 0.01:
            return None, 0
        
        team_covered = (actual_margin + spread) > 0
        
        if decision['decision'] == 'bet_team_cover':
            bet_correct = team_covered
        elif decision['decision'] == 'bet_opponent_cover':
            bet_correct = not team_covered
        else:
            return None, 0
        
        if bet_correct:
            profit = decision['bet_amount'] * (100/110)
        else:
            profit = -decision['bet_amount']
        
        return bet_correct, profit

    def _calculate_sample_weights(self, df, test_season):
        """Calculate sample weights that emphasize recent data"""
        years_ago = test_season - df['season']
        weights = np.exp(-0.231 * years_ago)
        return weights
    
    def walk_forward_validation(self, df, feature_cols, numeric_cols, categorical_cols, 
                                test_seasons=None):
        """Walk-forward validation"""
        if test_seasons is None:
            all_seasons = sorted(df['season'].unique())
            test_seasons = all_seasons[-5:] if len(all_seasons) >= 8 else all_seasons[-3:]
        
        if not self.quiet_mode:
            logger.info(f"\nTest seasons: {test_seasons}")
        
        all_predictions = []
        season_bankrolls = {test_seasons[0]: self.initial_bankroll}
        
        for i, test_season in enumerate(test_seasons):
            self.current_bankroll = season_bankrolls.get(test_season, self.initial_bankroll)
            
            val_season = test_season - 1
            cal_season = test_season - 2
            
            train_df = df[df['season'] < cal_season].copy()
            cal_df = df[df['season'] == cal_season].copy()
            val_df = df[df['season'] == val_season].copy()
            test_df = df[df['season'] == test_season].copy()
            
            if len(test_df) == 0:
                continue
            
            train_weights = self._calculate_sample_weights(train_df, test_season)
            
            X_train = self._prepare_features(train_df, numeric_cols, categorical_cols)
            X_val = self._prepare_features(val_df, numeric_cols, categorical_cols) if len(val_df) > 0 else None
            X_test = self._prepare_features(test_df, numeric_cols, categorical_cols)
            
            y_train_error = train_df['mov_target'].values
            y_val_error = val_df['mov_target'].values if len(val_df) > 0 else None
            y_test_error = test_df['mov_target'].values
            
            # Train models
            mean_model = self._train_mean_model(X_train, y_train_error, X_val, y_val_error, 
                                               sample_weights=train_weights.values)
            self.mean_models[test_season] = mean_model
            
            test_pred_error = mean_model.predict(X_test, num_iteration=mean_model.best_iteration)
            
            q16_model = self._train_quantile_model(X_train, y_train_error, X_val, y_val_error, 
                                                   quantile=0.16, sample_weights=train_weights.values)
            self.q16_models[test_season] = q16_model
            
            q84_model = self._train_quantile_model(X_train, y_train_error, X_val, y_val_error, 
                                                   quantile=0.84, sample_weights=train_weights.values)
            self.q84_models[test_season] = q84_model
            
            test_pred_q16 = q16_model.predict(X_test, num_iteration=q16_model.best_iteration)
            test_pred_q84 = q84_model.predict(X_test, num_iteration=q84_model.best_iteration)
            
            raw_quantile_std = (test_pred_q84 - test_pred_q16) / 2.0
            raw_quantile_std = np.clip(raw_quantile_std, 3.0, 20.0)
            
            # Conformal calibration
            conformal_quantile = None
            if len(cal_df) >= 30:
                conformal_quantile = self._calibrate_conformal_on_holdout(
                    cal_df, mean_model, q16_model, q84_model,
                    feature_cols, numeric_cols, categorical_cols
                )
                self.conformal_quantiles[test_season] = conformal_quantile
            
            if conformal_quantile is not None:
                test_pred_std_base = np.array([
                    self._combine_quantile_and_conformal(q_std, conformal_quantile)
                    for q_std in raw_quantile_std
                ])
            else:
                test_pred_std_base = raw_quantile_std
            
            # Volatility calibration
            volatility_multiplier = 1.0
            if len(val_df) > 0:
                val_pred_error = mean_model.predict(X_val, num_iteration=mean_model.best_iteration)
                val_pred_q16 = q16_model.predict(X_val, num_iteration=q16_model.best_iteration)
                val_pred_q84 = q84_model.predict(X_val, num_iteration=q84_model.best_iteration)
                val_raw_std = (val_pred_q84 - val_pred_q16) / 2.0
                val_raw_std = np.clip(val_raw_std, 3.0, 20.0)
                
                if conformal_quantile is not None:
                    val_pred_std = np.array([
                        self._combine_quantile_and_conformal(v_std, conformal_quantile)
                        for v_std in val_raw_std
                    ])
                else:
                    val_pred_std = val_raw_std
                
                val_spreads = val_df['team_spread'].values
                val_pred_mov = val_spreads + val_pred_error
                
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
                volatility_multiplier = self.volatility_multiplier_base
            
            test_pred_std = test_pred_std_base * volatility_multiplier
            test_pred_std = np.clip(test_pred_std, 3.0, 28.0)
            
            test_spreads = test_df['team_spread'].values
            test_pred_mov = test_spreads + test_pred_error
            
            # Make betting decisions
            season_results = test_df[[
                'gameId', 'startDate', 'team', 'opponent', 'team_spread', 'actual_margin'
            ]].copy()
            
            season_results['is_push'] = test_df['is_push'].values
            season_results['pred_spread_error'] = test_pred_error
            season_results['mov_pred_mean'] = test_pred_mov
            season_results['mov_pred_std'] = test_pred_std
            season_results['season'] = test_season
            
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
            
            betting_decisions = []
            running_bankroll = self.current_bankroll
            
            for idx in range(len(season_results)):
                row = season_results.iloc[idx]
                
                decision = self.make_betting_decision(
                    row['mov_pred_mean'],
                    row['mov_pred_std'],
                    row['team_spread'],
                    running_bankroll,
                    edge_percentile=edge_percentiles[idx]
                )
                
                if decision['decision'] != 'no_bet':
                    bet_correct, profit = self.calculate_bet_outcome(
                        decision,
                        row['actual_margin'],
                        row['team_spread']
                    )
                    
                    if bet_correct is None:
                        decision['bet_correct'] = None
                        decision['bet_profit'] = 0
                    else:
                        decision['bet_correct'] = bet_correct
                        decision['bet_profit'] = profit
                        running_bankroll += profit
                        
                        self._update_recent_performance(decision['bet_amount'], profit)
                        
                        if running_bankroll <= 0:
                            running_bankroll = 0
                else:
                    decision['bet_correct'] = None
                    decision['bet_profit'] = 0
                
                decision['running_bankroll'] = running_bankroll
                decision['edge_percentile'] = edge_percentiles[idx]
                betting_decisions.append(decision)
            
            for key in betting_decisions[0].keys():
                season_results[key] = [d[key] for d in betting_decisions]
            
            season_results['bet_made'] = season_results['decision'] != 'no_bet'
            
            if i < len(test_seasons) - 1:
                season_bankrolls[test_seasons[i + 1]] = running_bankroll
            
            all_predictions.append(season_results)
        
        predictions_df = pd.concat(all_predictions, ignore_index=True)
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

    def save_models(self, output_path, season):
        """Save trained models and calibration data."""
        save_path = Path(output_path) / 'models'
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save LightGBM models as text files
            if season in self.mean_models:
                self.mean_models[season].save_model(str(save_path / f'mean_model_s{season}.txt'))
                self.q16_models[season].save_model(str(save_path / f'q16_model_s{season}.txt'))
                self.q84_models[season].save_model(str(save_path / f'q84_model_s{season}.txt'))
            else:
                logger.warning(f"No models found for season {season} to save.")
                return

            # Save conformal quantiles (which are just numbers) using joblib
            if season in self.conformal_quantiles:
                joblib.dump(self.conformal_quantiles[season], save_path / f'conformal_q_s{season}.pkl')
            
            logger.info(f"💾 Models for season {season} saved to {save_path}")
        
        except Exception as e:
            logger.error(f"Failed to save models for season {season}: {e}")

    def run(self, test_seasons=None):
        """Execute the betting pipeline"""
        df = self.load_and_prepare_data()
        feature_cols, numeric_cols, categorical_cols = self.get_feature_columns(df)
        
        predictions_df = self.walk_forward_validation(
            df, feature_cols, numeric_cols, categorical_cols, test_seasons
        )
        
        return predictions_df


# ============================================================================
# OPTUNA OPTIMIZATION FRAMEWORK
# ============================================================================

def objective(trial, data_path, hpo_seasons):
    """
    Optuna multi-objective function - returns TWO metrics to MAXIMIZE.
    
    Objective 1: ROI (profitability)
    Objective 2: Bet coverage (statistical significance)
    
    CRITICAL: This function is ONLY evaluated on hpo_seasons (the validation set).
    The final holdout test set is NEVER shown to Optuna.
    
    Args:
        trial: Optuna trial object
        data_path: Path to training data
        hpo_seasons: List of seasons to use for HPO evaluation (e.g., [2019, 2020, 2021, 2022, 2023])
    
    Returns:
        tuple: (roi_percentage, bet_coverage_percentage) - both to maximize
    """
    
    # Suggest parameters with CONSTRAINTS to ensure logical ordering
    
    # Strategy thresholds
    min_edge_threshold = trial.suggest_float('min_edge_threshold', 0.025, 0.070)
    max_kelly_fraction = trial.suggest_float('max_kelly_fraction', 0.05, 0.35)
    
    # Uncertainty estimation
    blend_factor = trial.suggest_float('blend_factor', 0.3, 0.9)
    conformal_alpha = trial.suggest_float('conformal_alpha', 0.05, 0.15)
    volatility_multiplier_base = trial.suggest_float('volatility_multiplier_base', 1.2, 2.0)
    
    # Model hyperparameters
    num_leaves = trial.suggest_int('num_leaves', 20, 50)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.05)
    
    # Feature windows - CONSTRAINED to be ordered
    w1 = trial.suggest_int('window_1', 3, 8)
    w2 = trial.suggest_int('window_2', w1 + 2, 15)  # Force w2 > w1
    w3 = trial.suggest_int('window_3', w2 + 2, 25)  # Force w3 > w2
    feature_windows = [w1, w2, w3]
    
    # Kelly thresholds - CONSTRAINED to increase
    t1 = trial.suggest_float('kelly_edge_threshold_1', 0.045, 0.065)
    t2 = trial.suggest_float('kelly_edge_threshold_2', t1 + 0.01, 0.085)  # Force t2 > t1
    t3 = trial.suggest_float('kelly_edge_threshold_3', t2 + 0.01, 0.110)  # Force t3 > t2
    
    # Kelly multipliers - CONSTRAINED to increase
    m1 = trial.suggest_float('kelly_multiplier_1', 0.15, 0.35)
    m2 = trial.suggest_float('kelly_multiplier_2', m1 + 0.05, 0.50)  # Force m2 > m1
    m3 = trial.suggest_float('kelly_multiplier_3', m2 + 0.05, 0.75)  # Force m3 > m2
    
    params = {
        'min_edge_threshold': min_edge_threshold,
        'max_kelly_fraction': max_kelly_fraction,
        'blend_factor': blend_factor,
        'conformal_alpha': conformal_alpha,
        'volatility_multiplier_base': volatility_multiplier_base,
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_windows': feature_windows,
        'kelly_edge_threshold_1': t1,
        'kelly_edge_threshold_2': t2,
        'kelly_edge_threshold_3': t3,
        'kelly_multiplier_1': m1,
        'kelly_multiplier_2': m2,
        'kelly_multiplier_3': m3,
    }
    
    # Initialize model with these parameters
    model = OptimizableSpreadBettingModel(
        data_path=data_path,
        output_dir='./optuna_temp',
        initial_bankroll=100,
        **params
    )
    
    # Run in quiet mode (suppress logging during optimization)
    model.set_quiet_mode(True)
    
    try:
        # CRITICAL: Only run on HPO seasons, NOT the final test set
        results = model.run(test_seasons=hpo_seasons)
        
        # Calculate ROBUST metrics to optimize
        bets_made = results[results['bet_made']].copy()
        
        # Minimum viability checks
        if len(bets_made) == 0:
            return -100.0, 0.0  # No bets = fail both objectives
        
        # Get decided bets (exclude pushes)
        decided_bets = bets_made[bets_made['bet_correct'].notna()]
        
        if len(decided_bets) < 10:
            return -100.0, 0.0  # Not enough bets to evaluate
        
        # Calculate metrics
        total_wagered = bets_made['bet_amount'].sum()
        total_profit = bets_made['bet_profit'].sum()
        final_bankroll = results['running_bankroll'].iloc[-1]
        
        # Penalize bankruptcy
        if final_bankroll <= 0:
            return -100.0, 0.0
        
        # OBJECTIVE 1: ROI (profitability)
        roi = total_profit / total_wagered if total_wagered > 0 else -1.0
        roi_percentage = roi * 100  # Convert to percentage
        
        # OBJECTIVE 2: Bet Coverage (statistical significance)
        total_games = len(results)
        total_bets = len(bets_made)
        bet_coverage = total_bets / total_games if total_games > 0 else 0
        bet_coverage_percentage = bet_coverage * 100  # Convert to percentage
        
        # Optional: Apply soft constraints to prevent extreme solutions
        # (This helps Optuna focus on reasonable regions of the search space)
        
        # Soft penalty for extremely low coverage (< 5%)
        if bet_coverage < 0.05:
            roi_percentage *= 0.5  # Reduce ROI appeal if coverage too low
        
        # Soft penalty for excessive coverage (> 40%)
        if bet_coverage > 0.40:
            roi_percentage *= 0.8  # Reduce ROI appeal if betting too much
        
        # Log metrics for analysis
        trial.set_user_attr("roi", roi_percentage)
        trial.set_user_attr("bet_coverage", bet_coverage_percentage)
        trial.set_user_attr("total_bets", total_bets)
        trial.set_user_attr("total_games", total_games)
        trial.set_user_attr("final_bankroll", final_bankroll)
        trial.set_user_attr("win_rate", (decided_bets['bet_correct'].astype(bool).sum() / len(decided_bets)))
        
        # Sharpe ratio (for additional analysis, not used in optimization)
        bet_returns = decided_bets['bet_profit'] / decided_bets['bet_amount']
        sharpe = (bet_returns.mean() / bet_returns.std() * np.sqrt(252)) if bet_returns.std() > 0 else -10.0
        trial.set_user_attr("sharpe_ratio", sharpe)
        
        # Return BOTH objectives (Optuna will find Pareto frontier)
        return roi_percentage, bet_coverage_percentage
        
    except Exception as e:
        # If the model fails, return a very poor result
        logger.error(f"Trial failed: {e}")
        return -100.0, 0.0


def run_optimization(data_path, n_trials=100, hpo_seasons=None, study_name=None):
    """
    Run Optuna hyperparameter optimization with PROPER train/val/test split.
    Uses multi-objective optimization to balance ROI and bet coverage.
    
    Args:
        data_path: Path to the training data
        n_trials: Number of trials to run (100 is a good start, 500+ for thorough search)
        hpo_seasons: List of seasons for HPO evaluation (e.g., [2019, 2020, 2021, 2022, 2023])
                     These seasons are used ONLY for optimization, never for final testing
        study_name: Optional name for the study
    
    Returns:
        study: Optuna study object with all results
    """
    
    if hpo_seasons is None:
        # Default: Use seasons 2019-2023 for HPO
        # Final test will be 2024-2025 (handled separately)
        hpo_seasons = [2019, 2020, 2021, 2022, 2023]
    
    # Create a study that MAXIMIZES BOTH objectives
    study = optuna.create_study(
        directions=['maximize', 'maximize'],  # [ROI, Bet Coverage]
        study_name=study_name or 'spread_betting_hpo',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run the optimization
    logger.info(f"\n{'='*80}")
    logger.info(f"🔬 STARTING MULTI-OBJECTIVE HYPERPARAMETER OPTIMIZATION")
    logger.info(f"{'='*80}")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"HPO Seasons (validation): {hpo_seasons}")
    logger.info(f"Objective 1: Maximize ROI (profitability)")
    logger.info(f"Objective 2: Maximize Bet Coverage (statistical significance)")
    logger.info(f"⚠️  CRITICAL: Final test set (2024-2025) is NEVER shown to Optuna")
    logger.info(f"{'='*80}\n")
    
    study.optimize(
        lambda trial: objective(trial, data_path, hpo_seasons),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Report results - with multi-objective, we have a PARETO FRONTIER
    logger.info(f"\n{'='*80}")
    logger.info(f"🎯 OPTIMIZATION COMPLETE - PARETO FRONTIER ANALYSIS")
    logger.info(f"{'='*80}")
    logger.info(f"Total trials: {len(study.trials)}")
    logger.info(f"Pareto-optimal solutions: {len(study.best_trials)}")
    logger.info(f"\nTop 5 Pareto-optimal solutions:")
    logger.info(f"{'='*80}")
    logger.info(f"{'#':<4} {'ROI (%)':<12} {'Coverage (%)':<15} {'Win Rate':<12} {'Sharpe':<10}")
    logger.info(f"{'-'*80}")
    
    # Sort by ROI descending for display
    sorted_trials = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)[:5]
    
    for i, trial in enumerate(sorted_trials, 1):
        roi = trial.values[0]
        coverage = trial.values[1]
        win_rate = trial.user_attrs.get('win_rate', 0)
        sharpe = trial.user_attrs.get('sharpe_ratio', 0)
        
        logger.info(f"{i:<4} {roi:<12.2f} {coverage:<15.2f} {win_rate:<12.2%} {sharpe:<10.2f}")
    
    logger.info(f"{'='*80}\n")
    
    # Recommend a solution based on your preferences
    logger.info(f"💡 RECOMMENDED SOLUTION SELECTION:")
    logger.info(f"{'='*80}")
    logger.info(f"Choose from the Pareto frontier based on your priorities:")
    logger.info(f"  • High ROI, Low Coverage → Conservative, high-confidence bets")
    logger.info(f"  • Medium ROI, Medium Coverage → Balanced approach (RECOMMENDED)")
    logger.info(f"  • Lower ROI, High Coverage → More bets, more robust statistics")
    logger.info(f"\nTo select a specific trial, use:")
    logger.info(f"  best_trial = study.best_trials[index]  # 0 = highest ROI")
    logger.info(f"{'='*80}\n")
    
    return study

def select_best_trial_from_pareto(study, min_coverage=8.0, prefer='balanced'):
    """
    Select the best trial from the Pareto frontier based on preferences.
    
    Args:
        study: Optuna study object
        min_coverage: Minimum acceptable bet coverage (%)
        prefer: 'roi' (maximize profit), 'balanced' (best ROI*Coverage), or 'coverage' (max bets)
    
    Returns:
        best_trial: Selected trial object
    """
    
    # Filter trials that meet minimum coverage requirement
    viable_trials = [t for t in study.best_trials if t.values[1] >= min_coverage]
    
    if not viable_trials:
        logger.warning(f"⚠️  No trials met minimum coverage of {min_coverage}%")
        logger.warning(f"Using best available trial instead")
        viable_trials = study.best_trials
    
    if not viable_trials:
        logger.error("❌ No viable trials found in study!")
        return None
    
    if prefer == 'roi':
        # Select highest ROI
        best_trial = max(viable_trials, key=lambda t: t.values[0])
        logger.info(f"✓ Selected: Highest ROI strategy")
        
    elif prefer == 'coverage':
        # Select highest coverage (with reasonable ROI > 0)
        positive_roi_trials = [t for t in viable_trials if t.values[0] > 0]
        if not positive_roi_trials:
            logger.warning("⚠️  No trials with positive ROI, using highest coverage from all trials")
            positive_roi_trials = viable_trials
        best_trial = max(positive_roi_trials, key=lambda t: t.values[1])
        logger.info(f"✓ Selected: Highest coverage strategy")
        
    else:  # 'balanced'
        # Select best geometric mean (balances both objectives)
        # Add small epsilon to avoid issues with negative values
        best_trial = max(viable_trials, 
                        key=lambda t: ((max(0.1, t.values[0]) * t.values[1]) ** 0.5))
        logger.info(f"✓ Selected: Balanced strategy (geometric mean)")
    
    logger.info(f"  ROI: {best_trial.values[0]:.2f}%")
    logger.info(f"  Coverage: {best_trial.values[1]:.2f}%")
    logger.info(f"  Win Rate: {best_trial.user_attrs.get('win_rate', 0):.2%}")
    logger.info(f"  Total Bets: {best_trial.user_attrs.get('total_bets', 0)}")
    
    return best_trial

def save_optimization_results(study, best_params_dict, output_dir='./optuna_results'):
    """Save optimization results and visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save study object
    import pickle
    with open(output_path / 'study.pkl', 'wb') as f:
        pickle.dump(study, f)
    logger.info(f"💾 Saved study to {output_path / 'study.pkl'}")
    
    # Save best parameters as JSON
    import json
    with open(output_path / 'best_params.json', 'w') as f:
        json.dump(best_params_dict, f, indent=2)
    logger.info(f"💾 Saved best params to {output_path / 'best_params.json'}")
    
    # Create visualizations
    try:
        # Optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_html(str(output_path / 'optimization_history.html'))
        logger.info(f"📊 Saved optimization history to {output_path / 'optimization_history.html'}")
        
        # Parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_html(str(output_path / 'param_importances.html'))
        logger.info(f"📊 Saved parameter importances to {output_path / 'param_importances.html'}")
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")
    
    # Save trial history as CSV
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_path / 'trials_history.csv', index=False)
    logger.info(f"💾 Saved trials history to {output_path / 'trials_history.csv'}")


def run_stability_test(data_path, n_runs=5, n_trials_per_run=100, hpo_seasons=None):
    """
    Run the "Jitter Test" - multiple independent HPO runs to test parameter stability.
    Updated for multi-objective optimization.
    
    This tests whether the optimizer consistently finds the same parameters,
    or if results are just "lucky" and unstable.
    
    Args:
        data_path: Path to training data
        n_runs: Number of independent optimization runs (5 is recommended)
        n_trials_per_run: Number of trials per run (100 minimum)
        hpo_seasons: Seasons for HPO validation
    
    Returns:
        all_studies: List of Optuna study objects
        stability_report: Dict with stability metrics
        consensus_params: Dict with consensus parameter values
    """
    
    if hpo_seasons is None:
        hpo_seasons = [2019, 2020, 2021, 2022, 2023]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🔬 PARAMETER STABILITY TEST (Jitter Test)")
    logger.info(f"{'='*80}")
    logger.info(f"Runs: {n_runs}")
    logger.info(f"Trials per run: {n_trials_per_run}")
    logger.info(f"Total trials: {n_runs * n_trials_per_run}")
    logger.info(f"Mode: Multi-Objective Optimization (ROI + Coverage)")
    logger.info(f"\nThis tests whether parameters are STABLE or just LUCKY")
    logger.info(f"{'='*80}\n")
    
    all_studies = []
    all_best_params = []
    all_best_metrics = []
    
    for run_idx in range(n_runs):
        logger.info(f"\n{'─'*80}")
        logger.info(f"🔄 RUN {run_idx + 1}/{n_runs}")
        logger.info(f"{'─'*80}")
        
        study = run_optimization(
            data_path=data_path,
            n_trials=n_trials_per_run,
            hpo_seasons=hpo_seasons,
            study_name=f'spread_betting_hpo_run_{run_idx + 1}'
        )
        
        all_studies.append(study)
        
        best_trial = select_best_trial_from_pareto(
            study, 
            min_coverage=8.0,
            prefer='balanced'
        )

        # Add robustness check in case no trials are found
        if best_trial is None:
            logger.error(f"❌ Run {run_idx + 1}/{n_runs} produced no viable trials. Skipping.")
            continue # Skip to the next run

        # Get the selected parameters
        selected_params = best_trial.params

        all_best_params.append(selected_params) # Add to list for stability analysis
        all_best_metrics.append({
            'roi': best_trial.values[0],
            'coverage': best_trial.values[1],
            'win_rate': best_trial.user_attrs.get('win_rate', 0),
            'total_bets': best_trial.user_attrs.get('total_bets', 0)
        })

        # Save each run
        save_optimization_results(
            study, 
            selected_params,  # <-- Pass the selected parameters here
            output_dir=f'./optuna_results_run_{run_idx + 1}'
        )
    
    # Analyze stability
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 STABILITY ANALYSIS")
    logger.info(f"{'='*80}\n")
    
    # Convert to DataFrame for easy analysis
    params_df = pd.DataFrame(all_best_params)
    metrics_df = pd.DataFrame(all_best_metrics)
    
    # Calculate statistics for each parameter
    stability_report = {}
    
    logger.info(f"{'Parameter':<30} {'Mean':<12} {'Std':<12} {'CV (%)':<12} {'Status':<15}")
    logger.info("─"*80)
    
    for param in params_df.columns:
        mean_val = params_df[param].mean()
        std_val = params_df[param].std()
        cv = (std_val / mean_val * 100) if mean_val != 0 else 0  # Coefficient of variation
        
        # Determine stability status
        if cv < 5:
            status = "✅ VERY STABLE"
        elif cv < 15:
            status = "✓ STABLE"
        elif cv < 30:
            status = "⚠️  MODERATE"
        else:
            status = "❌ UNSTABLE"
        
        stability_report[param] = {
            'mean': mean_val,
            'std': std_val,
            'cv': cv,
            'status': status
        }
        
        logger.info(f"{param:<30} {mean_val:<12.4f} {std_val:<12.4f} {cv:<12.1f} {status:<15}")
    
    # Analyze metric stability
    logger.info("\n" + "="*80)
    logger.info("📊 PERFORMANCE METRICS STABILITY")
    logger.info("="*80)
    logger.info(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    logger.info("─"*80)
    
    for metric in ['roi', 'coverage', 'win_rate']:
        mean_val = metrics_df[metric].mean()
        std_val = metrics_df[metric].std()
        min_val = metrics_df[metric].min()
        max_val = metrics_df[metric].max()
        
        if metric == 'roi' or metric == 'coverage':
            logger.info(f"{metric.upper():<20} {mean_val:<12.2f} {std_val:<12.2f} {min_val:<12.2f} {max_val:<12.2f}")
        else:
            logger.info(f"{metric.replace('_', ' ').title():<20} {mean_val:<12.2%} {std_val:<12.2%} {min_val:<12.2%} {max_val:<12.2%}")
    
    # Overall stability assessment
    avg_cv = np.mean([v['cv'] for v in stability_report.values()])
    
    logger.info("\n" + "="*80)
    logger.info("🎯 OVERALL STABILITY ASSESSMENT")
    logger.info("="*80)
    logger.info(f"Average Coefficient of Variation: {avg_cv:.1f}%")
    logger.info(f"Mean ROI across runs: {metrics_df['roi'].mean():.2f}% ± {metrics_df['roi'].std():.2f}%")
    logger.info(f"Mean Coverage across runs: {metrics_df['coverage'].mean():.2f}% ± {metrics_df['coverage'].std():.2f}%")
    
    if avg_cv < 10:
        logger.info("\n✅ EXCELLENT: Parameters are highly stable across runs.")
        logger.info("   → Your edge is ROBUST. These parameters can be trusted.")
        logger.info("   → Proceed to final holdout test with confidence.")
    elif avg_cv < 20:
        logger.info("\n✓ GOOD: Parameters are reasonably stable.")
        logger.info("   → Most parameters converge to similar values.")
        logger.info("   → Consider a few more runs or trials for critical params.")
    elif avg_cv < 35:
        logger.info("\n⚠️  MODERATE: Parameters show some instability.")
        logger.info("   → The optimization landscape may be noisy.")
        logger.info("   → Consider: (1) more trials, (2) k-fold CV, (3) feature engineering")
    else:
        logger.info("\n❌ WARNING: Parameters are highly unstable!")
        logger.info("   → The optimizer is likely fitting to noise, not signal.")
        logger.info("   → Your edge may not be as strong as it appears.")
        logger.info("   → Recommended: Improve features before trusting these results.")
    
    # Find consensus parameters (median across runs)
    logger.info("\n" + "="*80)
    logger.info("🎯 CONSENSUS PARAMETERS (Median across runs)")
    logger.info("="*80)
    
    consensus_params = {}
    for param in params_df.columns:
        if params_df[param].dtype in [np.int64, np.int32]:
            consensus_params[param] = int(params_df[param].median())
            logger.info(f"  {param}: {consensus_params[param]}")
        else:
            consensus_params[param] = float(params_df[param].median())
            logger.info(f"  {param}: {consensus_params[param]:.4f}")
    
    # Save stability report
    stability_output = Path('./stability_analysis')
    stability_output.mkdir(exist_ok=True)
    
    params_df.to_csv(stability_output / 'all_best_params.csv', index=False)
    metrics_df.to_csv(stability_output / 'all_best_metrics.csv', index=False)
    
    import json
    with open(stability_output / 'consensus_params.json', 'w') as f:
        json.dump(consensus_params, f, indent=2)
    
    with open(stability_output / 'stability_report.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_report = {}
        for key, value in stability_report.items():
            serializable_report[key] = {
                'mean': float(value['mean']),
                'std': float(value['std']),
                'cv': float(value['cv']),
                'status': value['status']
            }
        json.dump(serializable_report, f, indent=2)
    
    # Save performance metrics summary
    metrics_summary = {
        'mean_roi': float(metrics_df['roi'].mean()),
        'std_roi': float(metrics_df['roi'].std()),
        'mean_coverage': float(metrics_df['coverage'].mean()),
        'std_coverage': float(metrics_df['coverage'].std()),
        'mean_win_rate': float(metrics_df['win_rate'].mean()),
        'std_win_rate': float(metrics_df['win_rate'].std()),
        'avg_cv_params': float(avg_cv)
    }
    
    with open(stability_output / 'metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    logger.info(f"\n💾 Stability analysis saved to {stability_output}/")
    logger.info(f"   - all_best_params.csv: All selected parameters from each run")
    logger.info(f"   - all_best_metrics.csv: Performance metrics from each run")
    logger.info(f"   - consensus_params.json: Median parameters (use these!)")
    logger.info(f"   - stability_report.json: Detailed stability statistics")
    logger.info(f"   - metrics_summary.json: Performance summary across runs")
    logger.info("="*80 + "\n")
    
    return all_studies, stability_report, consensus_params


if __name__ == "__main__":
    DATA_PATH = 'feature_output/spread_training_data.parquet'
    
    # ========================================================================
    # CRITICAL: Proper Train / HPO-Validation / Final-Test Split
    # ========================================================================
    
    HPO_VALIDATION_SEASONS = [2019, 2020, 2021, 2022, 2023]
    FINAL_TEST_SEASONS = [2024, 2025]
    
    logger.info(f"\n{'='*80}")
    logger.info("📊 DATA SPLIT STRATEGY")
    logger.info(f"{'='*80}")
    logger.info(f"Training: < 2019 (used inside walk-forward loop)")
    logger.info(f"HPO Validation: {HPO_VALIDATION_SEASONS}")
    logger.info(f"Final Test (holdout): {FINAL_TEST_SEASONS}")
    logger.info(f"{'='*80}\n")

    # ========================================================================
    # CHOOSE YOUR MODE
    # ========================================================================
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'stability':
        # ====================================================================
        # MODE 1: STABILITY TEST (Recommended first step)
        # ====================================================================
        # Run this FIRST to test if parameters are robust or just lucky
        
        logger.info("\n🔬 RUNNING PARAMETER STABILITY TEST")
        logger.info("="*80)
        logger.info("This will run 1 independent HPO run (100 total trials)")
        logger.info("Expected time: varies depending on hardware")
        logger.info("="*80 + "\n")
        
        all_studies, stability_report, consensus_params = run_stability_test(
            data_path=DATA_PATH,
            n_runs=5,
            n_trials_per_run=100,
            hpo_seasons=HPO_VALIDATION_SEASONS
        )
        
        # Test consensus params on holdout
        logger.info("\n🧪 TESTING CONSENSUS PARAMETERS ON HOLDOUT")
        logger.info("="*80 + "\n")

        try:
            logger.info("Remapping consensus parameters for model constructor...")
            # Fix window parameters
            w1 = consensus_params.pop('window_1')
            w2 = consensus_params.pop('window_2')
            w3 = consensus_params.pop('window_3')
            consensus_params['feature_windows'] = [int(w1), int(w2), int(w3)]
            
            # Ensure num_leaves is an integer, as JSON saved it as a float
            if 'num_leaves' in consensus_params:
                consensus_params['num_leaves'] = int(consensus_params['num_leaves'])
            
            logger.info("Remapping complete.")
            
        except KeyError as e:
            logger.error(f"ERROR: consensus_params is missing a required key: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error processing consensus parameters: {e}")
            sys.exit(1)

        consensus_model = OptimizableSpreadBettingModel(
            data_path=DATA_PATH,
            output_dir='./consensus_test_output',
            initial_bankroll=100,
            **consensus_params
        )
        consensus_model.set_quiet_mode(False)
        
        consensus_results = consensus_model.run(test_seasons=FINAL_TEST_SEASONS)
        consensus_results.to_parquet('./consensus_test_output/holdout_predictions.parquet', index=False)
        
        # Calculate and display consensus results
        consensus_bets = consensus_results[consensus_results['bet_made']].copy()
        consensus_decided = consensus_bets[consensus_bets['bet_correct'].notna()]
        
        if len(consensus_decided) > 0:
            consensus_bankroll = consensus_results['running_bankroll'].iloc[-1]
            consensus_wagered = consensus_bets['bet_amount'].sum()
            consensus_profit = consensus_bets['bet_profit'].sum()
            consensus_roi = (consensus_profit / consensus_wagered * 100) if consensus_wagered > 0 else 0
            
            wins = consensus_decided['bet_correct'].astype(bool).sum()
            win_rate = wins / len(consensus_decided)
            
            logger.info(f"\n{'='*80}")
            logger.info("🏆 CONSENSUS PARAMETERS HOLDOUT TEST RESULTS")
            logger.info(f"{'='*80}")
            logger.info(f"Seasons: {FINAL_TEST_SEASONS}")
            logger.info(f"Initial Bankroll: $100.00")
            logger.info(f"Final Bankroll: ${consensus_bankroll:,.2f}")
            logger.info(f"Total Return: {((consensus_bankroll - 100) / 100 * 100):+.1f}%")
            logger.info(f"ROI: {consensus_roi:+.2f}%")
            logger.info(f"Win Rate: {win_rate:.1%}")
            logger.info(f"Total Bets: {len(consensus_bets)}")
            logger.info(f"Bet Coverage: {len(consensus_bets) / len(consensus_results) * 100:.1f}%")
            logger.info(f"{'='*80}\n")
        
        logger.info("\n✅ STABILITY TEST COMPLETE")
        logger.info("="*80)
        logger.info("Review stability_analysis/ to assess parameter robustness")
        logger.info("="*80 + "\n")
        
        sys.exit(0)
    
    # ========================================================================
    # MODE 2: LOAD SAVED PARAMETERS (if not running stability test)
    # ========================================================================
    
    logger.info(f"\n🧪 LOADING SAVED PARAMETERS")
    logger.info("="*80)
    logger.info("Loading consensus parameters from stability_analysis/consensus_params.json\n")

    # Load the parameters saved by the stability test
    try:
        with open('./stability_analysis/consensus_params.json', 'r') as f:
            consensus_params = json.load(f)
    except FileNotFoundError:
        logger.error("ERROR: stability_analysis/consensus_params.json not found.")
        logger.error("Please run the 'stability' mode first: python lgbmD.py stability")
        sys.exit(1)
        
    try:
        # Fix window parameters
        w1 = consensus_params.pop('window_1')
        w2 = consensus_params.pop('window_2')
        w3 = consensus_params.pop('window_3')
        consensus_params['feature_windows'] = [int(w1), int(w2), int(w3)]
        
        # Ensure num_leaves is an integer, as JSON saved it as a float
        if 'num_leaves' in consensus_params:
            consensus_params['num_leaves'] = int(consensus_params['num_leaves'])
        
        logger.info("Successfully loaded, re-mapped window parameters, and cast num_leaves to int.")
        
    except KeyError as e:
        logger.error(f"ERROR: consensus_params.json is missing a required key: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing parameters: {e}")
        sys.exit(1)
        
    # ========================================================================
    # STEP 2: Evaluate best parameters on FINAL HOLDOUT TEST SET
    # ========================================================================
    
    logger.info("\n🧪 PHASE 2: FINAL EVALUATION ON HOLDOUT TEST SET")
    logger.info("="*80)
    logger.info("⚠️  This is the ONLY performance metric we trust!")
    logger.info("   (Final test seasons were NEVER shown to Optuna)\n")
    
    best_model = OptimizableSpreadBettingModel(
        data_path=DATA_PATH,
        output_dir='./final_test_output',
        initial_bankroll=100,
        **consensus_params
    )
    best_model.set_quiet_mode(False)
    
    final_results = best_model.run(test_seasons=FINAL_TEST_SEASONS)
    
    # Calculate final metrics
    final_bets = final_results[final_results['bet_made']].copy()
    final_decided = final_bets[final_bets['bet_correct'].notna()]
    
    if len(final_decided) > 0:
        final_bankroll = final_results['running_bankroll'].iloc[-1]
        total_wagered = final_bets['bet_amount'].sum()
        total_profit = final_bets['bet_profit'].sum()
        final_roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
        
        wins = final_decided['bet_correct'].astype(bool).sum()
        win_rate = wins / len(final_decided)
        
        bet_returns = final_decided['bet_profit'] / final_decided['bet_amount']
        final_sharpe = (bet_returns.mean() / bet_returns.std() * np.sqrt(252)) if bet_returns.std() > 0 else 0
        
        logger.info(f"\n{'='*80}")
        logger.info("🏆 FINAL HOLDOUT TEST RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Seasons: {FINAL_TEST_SEASONS}")
        logger.info(f"Initial Bankroll: $100.00")
        logger.info(f"Final Bankroll: ${final_bankroll:,.2f}")
        logger.info(f"Total Return: {((final_bankroll - 100) / 100 * 100):+.1f}%")
        logger.info(f"ROI: {final_roi:+.2f}%")
        logger.info(f"Win Rate: {win_rate:.1%}")
        logger.info(f"Sharpe Ratio: {final_sharpe:.2f}")
        logger.info(f"Total Bets: {len(final_bets)}")
        logger.info(f"{'='*80}\n")
    
    final_results.to_parquet('./final_test_output/holdout_predictions.parquet', index=False)
    
    logger.info("\n💾 Saving final trained models...")
    for season in FINAL_TEST_SEASONS:
        best_model.save_models('./final_test_output', season)

    # ========================================================================
    # STEP 3: Compare against baseline
    # ========================================================================
    
    logger.info("\n📊 PHASE 3: COMPARISON WITH BASELINE")
    logger.info("="*80)
    logger.info("Running original hard-coded parameters on same holdout set...\n")
    
    baseline_model = OptimizableSpreadBettingModel(
        data_path=DATA_PATH,
        output_dir='./baseline_output',
        initial_bankroll=100,
        min_edge_threshold=0.05,
        max_kelly_fraction=0.05,
        blend_factor=0.6,
        conformal_alpha=0.10,
        num_leaves=31,
        learning_rate=0.02,
        feature_windows=[5, 10, 20],
        volatility_multiplier_base=1.5,
        kelly_edge_threshold_1=0.055,
        kelly_edge_threshold_2=0.075,
        kelly_edge_threshold_3=0.10,
        kelly_multiplier_1=0.25,
        kelly_multiplier_2=0.40,
        kelly_multiplier_3=0.60
    )
    baseline_model.set_quiet_mode(False)
    
    baseline_results = baseline_model.run(test_seasons=FINAL_TEST_SEASONS)
    
    baseline_bets = baseline_results[baseline_results['bet_made']].copy()
    baseline_decided = baseline_bets[baseline_bets['bet_correct'].notna()]
    
    if len(baseline_decided) > 0:
        baseline_bankroll = baseline_results['running_bankroll'].iloc[-1]
        baseline_wagered = baseline_bets['bet_amount'].sum()
        baseline_profit = baseline_bets['bet_profit'].sum()
        baseline_roi = (baseline_profit / baseline_wagered * 100) if baseline_wagered > 0 else 0
        
        logger.info(f"\n{'='*80}")
        logger.info("📈 BASELINE vs OPTIMIZED COMPARISON")
        logger.info(f"{'='*80}")
        logger.info(f"{'Metric':<20} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
        logger.info("-"*80)
        logger.info(f"{'Final Bankroll':<20} ${baseline_bankroll:<14,.2f} ${final_bankroll:<14,.2f} "
                   f"{((final_bankroll - baseline_bankroll) / baseline_bankroll * 100):+.1f}%")
        logger.info(f"{'ROI':<20} {baseline_roi:<14.2f}% {final_roi:<14.2f}% "
                   f"{(final_roi - baseline_roi):+.2f}pp")
        logger.info(f"{'='*80}\n")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ OPTIMIZATION PIPELINE COMPLETE")
    logger.info(f"{'='*80}")
    logger.info("Results saved to:")
    logger.info("  📁 optuna_results/ - HPO study and visualizations")
    logger.info("  📁 final_test_output/ - Holdout test predictions")
    logger.info("  📁 baseline_output/ - Baseline comparison")
    logger.info("\n⚠️  CRITICAL REMINDER:")
    logger.info("  The ONLY metric you can trust is the Final Holdout Test result.")
    logger.info("  HPO validation performance will be optimistically biased.")
    logger.info("\nNext steps:")
    logger.info("  1. Run stability test: python lgbmD.py stability")
    logger.info("  2. Review parameter importances in optuna_results/")
    logger.info("  3. If params are stable, deploy for 2026 season")
    logger.info("  4. Consider feature engineering for even better edge")
    logger.info(f"{'='*80}\n")