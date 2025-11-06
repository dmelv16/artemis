"""
Adds betting lines and market data.
"""
import pandas as pd

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
        
        print(f"Added betting lines for {df['spread'].notna().sum()} games")
        return df