import pandas as pd

def add_week_column(df):
    """
    Add 'week' column based on weeks since first game of each season.
    Also ensures 'season' column exists by deriving it from startDate if needed.
    """
    df['startDate'] = pd.to_datetime(df['startDate'])
    
    # Ensure season column exists
    if 'season' not in df.columns:
        print("  Deriving 'season' column from startDate...")
        # College basketball season: games from Aug-Dec are season year+1, 
        # games from Jan-July are also season year (which started previous Aug)
        df['season'] = df['startDate'].apply(
            lambda x: x.year + 1 if x.month >= 8 else x.year
        )
    
    def calculate_week(group):
        season_start = group['startDate'].min()
        days_since_start = (group['startDate'] - season_start).dt.days
        group['week'] = (days_since_start // 7) + 1
        return group
    
    # Fix the FutureWarning by adding include_groups=False
    df = df.groupby('season', group_keys=False).apply(calculate_week, include_groups=False)
    df['week'] = df['week'].astype(int)
    
    return df