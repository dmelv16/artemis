import pandas as pd

def add_week_column(df):
    """
    Add 'week' column based on weeks since first game of each season.
    """
    df['startDate'] = pd.to_datetime(df['startDate'])
    
    def calculate_week(group):
        season_start = group['startDate'].min()
        days_since_start = (group['startDate'] - season_start).dt.days
        group['week'] = (days_since_start // 7) + 1
        return group
    
    # Fix the FutureWarning by adding include_groups=False
    df = df.groupby('season', group_keys=False).apply(calculate_week, include_groups=False)
    df['week'] = df['week'].astype(int)
    
    return df