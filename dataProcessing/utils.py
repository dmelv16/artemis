"""
Utility functions for the pipeline.
"""
import pandas as pd

def save_master_table(df, output_path, save_parquet=True):
    print(f"Saving master table to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Master table saved with shape {df.shape}")
    
    if save_parquet:
        parquet_path = output_path.replace('.csv', '.parquet')
        df.to_parquet(parquet_path, index=False)
        print(f"Also saved as parquet: {parquet_path}")
    
    return df

def print_summary(df):
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE!")
    print(f"Final table shape: {df.shape}")
    print(f"Total features: {len(df.columns)}")
    print("="*60 + "\n")
    
    print("FEATURE SUMMARY:")
    print("-" * 40)
    
    feature_types = {
        'Target': [c for c in df.columns if 'actual' in c or 'home_won' in c],
        'Betting Lines': [c for c in df.columns if any(x in c for x in ['spread', 'overUnder', 'moneyline'])],
        'Rolling Stats': [c for c in df.columns if '_L5' in c or '_L10' in c or '_std' in c],
        'Roster/Recruiting': [c for c in df.columns if any(x in c for x in ['roster', 'recruit'])],
        'Head-to-Head': [c for c in df.columns if 'h2h' in c],
        'Rankings': [c for c in df.columns if 'rank' in c or 'AP' in c or 'Coaches' in c],
        'Differentials': [c for c in df.columns if c.startswith('diff_')],
        'Temporal': [c for c in df.columns if any(x in c for x in ['day', 'week', 'month', 'rest'])]
    }
    
    for feature_type, features in feature_types.items():
        print(f"{feature_type}: {len(features)} features")
    
    print("\nTarget Variable Distribution:")
    print("-" * 40)
    print(f"Spread: mean={df['spread_actual'].mean():.1f}, std={df['spread_actual'].std():.1f}")
    print(f"Total: mean={df['total_actual'].mean():.1f}, std={df['total_actual'].std():.1f}")
    print(f"Home Win %: {df['home_won'].mean():.3f}")