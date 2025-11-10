"""
Configuration settings for the feature engineering pipeline.
"""

# Database Configuration
DB_CONFIG = {
    'server': 'DESKTOP-J9IV3OH',  # e.g., 'localhost' or 'DESKTOP-XXX\SQLEXPRESS'
    'database': 'cbbDB',
    'use_windows_auth': True,
    'username': None,  # Set if use_windows_auth = False
    'password': None   # Set if use_windows_auth = False
}

# Pipeline Configuration
PIPELINE_CONFIG = {
    'season_start': 2006,
    'season_end': 2025,
    'output_path': 'cbb_master_features.parquet',
    'rolling_windows': [5, 10, 'season'],
    'save_parquet': True
}

# Features to calculate rolling averages for
ROLLING_STAT_COLUMNS = [
    'pace', 'possessions', 'rating', 'oppRating',
    'points', 'oppPoints',
    'fieldGoalsPct', 'oppFieldGoalsPct',
    'threePointFGPct', 'oppThreePointFGPct',
    'freeThrowsPct', 'oppFreeThrowsPct',
    'effectiveFieldGoalPct', 'oppEffectiveFieldGoalPct',
    'turnoverRatio', 'oppTurnoverRatio',
    'offensiveReboundPct', 'oppOffensiveReboundPct',
    'freeThrowRate', 'oppFreeThrowRate',
    'assists', 'steals', 'blocks',
    'turnovers', 'fouls'
]

# Player statistics to track
PLAYER_STAT_COLUMNS = [
    'points', 'assists', 'totalRebounds',
    'steals', 'blocks', 'turnovers',
    'fieldGoalsMade', 'fieldGoalsAttempted', 'fieldGoalsPct',
    'threePointFGMade', 'threePointFGAttempted', 'threePointFGPct',
    'freeThrowsMade', 'freeThrowsAttempted', 'freeThrowsPct',
    'offensiveRebounds', 'defensiveRebounds',
    'fouls', 'minutes',
    'gameScore', 'offensiveRating', 'defensiveRating',
    'usage', 'effectiveFieldGoalPct', 'trueShootingPct'
]

# How to aggregate player stats to team level
PLAYER_AGGREGATIONS = {
    'starters': ['sum', 'mean', 'max'],
    'bench': ['sum', 'mean'],
    'top_players': ['sum', 'mean', 'max']
}