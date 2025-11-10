import pandas as pd
import numpy as np

# Load the data
df = pd.read_parquet('feature_output/spread_training_data.parquet')

print("=" * 80)
print("SPREAD DATA DIAGNOSTIC")
print("=" * 80)

# Calculate actual margin
df['calculated_margin'] = df['points'] - df['opp_points']

# Check the difference
df['margin_diff'] = (df['spread_actual'] - df['calculated_margin']).abs()

print("\n1. SPREAD_ACTUAL vs CALCULATED MARGIN")
print(f"   Mean spread_actual: {df['spread_actual'].mean():.2f}")
print(f"   Mean calculated margin: {df['calculated_margin'].mean():.2f}")
print(f"   Mean absolute difference: {df['margin_diff'].mean():.2f}")
print(f"   Max difference: {df['margin_diff'].max():.2f}")

# Show examples where they differ significantly
print("\n2. EXAMPLES WHERE THEY DIFFER (top 5):")
diff_examples = df.nlargest(5, 'margin_diff')[['gameId', 'team', 'opponent', 'points', 
                                                  'opp_points', 'spread_actual', 
                                                  'calculated_margin', 'is_home', 'won']]
print(diff_examples.to_string())

# Check if spread_actual is always 0 or constant
print("\n3. SPREAD_ACTUAL VALUE DISTRIBUTION:")
print(f"   Unique values: {df['spread_actual'].nunique()}")
print(f"   Zero values: {(df['spread_actual'] == 0).sum()} / {len(df)}")
print(f"   Value counts (top 10):")
print(df['spread_actual'].value_counts().head(10))

# Check relationship between spread and spread_actual
print("\n4. SPREAD vs SPREAD_ACTUAL:")
print(f"   Mean 'spread' column: {df['spread'].mean():.2f}")
print(f"   Mean 'spread_actual': {df['spread_actual'].mean():.2f}")
print(f"   Correlation: {df['spread'].corr(df['spread_actual']):.4f}")

# Check if spread_actual might be the actual closing line (not margin)
# If spread_actual is close to the 'spread' column, it's probably the closing line
line_diff = (df['spread_actual'] - df['spread']).abs()
print(f"\n5. IS SPREAD_ACTUAL THE CLOSING LINE?")
print(f"   Mean |spread_actual - spread|: {line_diff.mean():.2f}")
print(f"   Median |spread_actual - spread|: {line_diff.median():.2f}")
if line_diff.mean() < 2.0:
    print("   → LIKELY: spread_actual appears to be the closing spread, NOT the game margin!")

# Calculate what the target SHOULD be
df['correct_spread_target'] = df['calculated_margin'] - df['spread']
df['current_spread_target'] = df['spread_actual'] - df['spread']

print("\n6. SPREAD TARGET VALIDATION:")
print(f"   Using calculated_margin - spread:")
print(f"     Mean: {df['correct_spread_target'].mean():.2f}")
print(f"     Cover rate: {(df['correct_spread_target'] > 0).mean():.1%}")
print(f"   Using spread_actual - spread (current):")
print(f"     Mean: {df['current_spread_target'].mean():.2f}")
print(f"     Cover rate: {(df['current_spread_target'] > 0).mean():.1%}")

# Check for duplicate games
print("\n7. DUPLICATE GAME CHECK:")
dup_check = df.groupby('gameId').size()
print(f"   Total rows: {len(df):,}")
print(f"   Unique games: {len(dup_check):,}")
print(f"   Games appearing twice: {(dup_check == 2).sum():,}")

# For games appearing twice, check if they're inverse perspectives
if (dup_check == 2).any():
    print("\n8. CHECKING INVERSE PERSPECTIVES:")
    # Find a game with 2 rows
    sample_game = dup_check[dup_check == 2].index[0]
    sample_rows = df[df['gameId'] == sample_game]
    
    print(f"\n   Example gameId: {sample_game}")
    print(sample_rows[['team', 'opponent', 'points', 'opp_points', 
                       'calculated_margin', 'spread', 'is_home', 
                       'won']].to_string())
    
    # Check if margins are exactly opposite
    margins = sample_rows['calculated_margin'].values
    if len(margins) == 2 and abs(margins[0] + margins[1]) < 0.01:
        print("\n   → Margins are exact opposites (as expected)")
    
    # Check spreads
    spreads = sample_rows['spread'].values
    print(f"\n   Spread values: {spreads}")
    if len(spreads) == 2:
        if abs(spreads[0] + spreads[1]) < 0.01:
            print("   → Spreads are opposite (each from team's perspective)")
        elif abs(spreads[0] - spreads[1]) < 0.01:
            print("   → Spreads are identical (both from same perspective)")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)

# Determine the issue
if line_diff.mean() < 2.0:
    print("❌ CRITICAL: 'spread_actual' is NOT the game margin!")
    print("   It appears to be the closing spread line.")
    print("\n   FIX: Use calculated_margin (points - opp_points) instead:")
    print("   spread_target = calculated_margin - spread")
else:
    print("✓ spread_actual appears to be a valid game margin")
    print("  (though there are some anomalies to investigate)")

print("\nTo fix, update the feature engineering script to use:")
print("  df['spread_actual'] = df['points'] - df['opp_points']")
print("  # Or just remove spread_actual and always calculate it fresh")