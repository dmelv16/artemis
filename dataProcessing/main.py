"""
Update your main.py to include the player stats step:
"""

from config import DB_CONFIG, PIPELINE_CONFIG
from db_connection import DBConnection
from base_table import BaseTableBuilder
from betting_lines import BettingLinesProcessor
from rolling_stats import RollingStatsCalculator
from player_stats import PlayerStatsCalculator  # NEW IMPORT
from roster_recruiting import RosterRecruitingProcessor
from head_to_head import HeadToHeadProcessor
from rankings import RankingsProcessor
from feature_engineering import FeatureEngineer
from utils import save_master_table, print_summary

def main():
    print("\n" + "="*60)
    print("STARTING FEATURE ENGINEERING PIPELINE")
    print("="*60 + "\n")
    
    # Initialize database connection
    db_conn = DBConnection(DB_CONFIG)
    
    # Step 1: Create base table
    base_builder = BaseTableBuilder(db_conn)
    df = base_builder.create(
        PIPELINE_CONFIG['season_start'], 
        PIPELINE_CONFIG['season_end']
    )
    
    # Step 2: Add betting lines
    lines_processor = BettingLinesProcessor(db_conn)
    df = lines_processor.add_lines(df)
    
    # Step 3: Calculate rolling team stats
    rolling_calc = RollingStatsCalculator(db_conn)
    df = rolling_calc.calculate(df, PIPELINE_CONFIG['rolling_windows'])
    
    # Step 3b: Calculate rolling PLAYER stats (NEW!)
    player_calc = PlayerStatsCalculator(db_conn)
    df = player_calc.calculate(df, PIPELINE_CONFIG['rolling_windows'])
    
    # Step 4: Add roster and recruiting
    roster_processor = RosterRecruitingProcessor(db_conn)
    df = roster_processor.add_features(df)
    
    # Step 5: Add head-to-head
    h2h_processor = HeadToHeadProcessor(db_conn)
    df = h2h_processor.add_features(df)
    
    # Step 6: Add rankings
    rankings_processor = RankingsProcessor(db_conn)
    df = rankings_processor.add_features(df)
    
    # Step 7: Feature engineering
    feature_eng = FeatureEngineer(db_conn)
    df = feature_eng.engineer_all_features(df)
    
    # Save results
    df = save_master_table(
        df, 
        PIPELINE_CONFIG['output_path'],
        PIPELINE_CONFIG['save_parquet']
    )
    
    # Print summary
    print_summary(df)
    
    return df

if __name__ == "__main__":
    master_df = main()
    print("\nReady for model training!")
    print(f"Load data with: pd.read_parquet('{PIPELINE_CONFIG['output_path'].replace('.csv', '.parquet')}')")