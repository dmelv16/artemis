"""
Adds betting lines and market data.
"""
import pandas as pd
import numpy as np

class BettingLinesProcessor:
    def __init__(self, db_conn):
        self.db = db_conn
        
    def add_lines(self, df):
        print("Adding pre-game betting lines...")
        
        query = """
        SELECT 
            l.gameId,
            l.spread,
            l.overUnder,
            l.homeMoneyline,
            l.awayMoneyline,
            l.spreadOpen,
            l.overUnderOpen,
            CASE 
                WHEN l.homeMoneyline < 0 THEN ABS(l.homeMoneyline)/(ABS(l.homeMoneyline) + 100.0)
                WHEN l.homeMoneyline > 0 THEN 100.0/(l.homeMoneyline + 100.0)
                ELSE NULL
            END as home_implied_prob,
            CASE 
                WHEN l.awayMoneyline < 0 THEN ABS(l.awayMoneyline)/(ABS(l.awayMoneyline) + 100.0)
                WHEN l.awayMoneyline > 0 THEN 100.0/(l.awayMoneyline + 100.0)
                ELSE NULL
            END as away_implied_prob
        FROM lines l
        WHERE l.provider = 'consensus' OR l.provider IS NULL
        """
        
        lines_df = self.db.query(query)
        df = df.merge(lines_df, on='gameId', how='left')
        
        # Calculate line movements
        df['spread_movement'] = df['spread'] - df['spreadOpen']
        df['total_movement'] = df['overUnder'] - df['overUnderOpen']
        
        # Add vig-free (true) probabilities
        df['total_vig'] = df['home_implied_prob'] + df['away_implied_prob']
        df['home_true_prob'] = df['home_implied_prob'] / df['total_vig']
        df['away_true_prob'] = df['away_implied_prob'] / df['total_vig']
        
        # Market efficiency indicators
        df['line_movement_magnitude'] = np.abs(df['spread_movement'])
        df['total_movement_magnitude'] = np.abs(df['total_movement'])
        df['significant_line_move'] = df['line_movement_magnitude'] >= 2.0  # 2+ point move
        df['significant_total_move'] = df['total_movement_magnitude'] >= 2.0
        
        # Expected margins from spread
        df['expected_home_margin'] = -df['spread']  # negative spread means home favored
        df['expected_away_margin'] = df['spread']
        
        # Implied total scores from spread and total
        df['implied_home_score'] = (df['overUnder'] - df['spread']) / 2
        df['implied_away_score'] = (df['overUnder'] + df['spread']) / 2
        
        # Betting value indicators (if actual scores available)
        if 'homeScore' in df.columns and 'awayScore' in df.columns:
            df['actual_margin'] = df['homeScore'] - df['awayScore']
            df['actual_total'] = df['homeScore'] + df['awayScore']
            
            # Spread cover
            df['home_covered_spread'] = df['actual_margin'] > -df['spread']
            df['spread_diff'] = df['actual_margin'] - (-df['spread'])
            
            # Over/under result
            df['went_over'] = df['actual_total'] > df['overUnder']
            df['total_diff'] = df['actual_total'] - df['overUnder']
            
            # Betting outcomes
            df['favorite_covered'] = np.where(
                df['spread'] < 0,  # home favored
                df['home_covered_spread'],
                ~df['home_covered_spread']
            )
            
            # Edge detection (market was wrong)
            df['line_error'] = np.abs(df['spread_diff'])
            df['total_error'] = np.abs(df['total_diff'])
        
        # Favorite/underdog classification
        df['home_favorite'] = df['spread'] < 0
        df['favorite_size'] = np.abs(df['spread'])
        df['is_pick_em'] = np.abs(df['spread']) < 1.5
        
        # Categorize game types by spread
        df['game_type'] = pd.cut(
            df['favorite_size'],
            bins=[-np.inf, 3, 7, 14, np.inf],
            labels=['Close', 'Small_Favorite', 'Medium_Favorite', 'Large_Favorite']
        )
        
        print(f"Added betting lines for {df['spread'].notna().sum()} games")
        return df
    
    def calculate_closing_line_value(self, df, bet_spread):
        """
        Calculate CLV - difference between bet line and closing line
        Positive CLV = got better line than close
        """
        df['clv'] = bet_spread - df['spread']
        return df