import pandas as pd
import pyarrow.parquet as pq
import numpy as np

def transform_game_to_team(input_file, output_file):
    """
    Transform game-by-game data to team-by-team perspective.
    Each game becomes 2 rows (one per team).
    """
    
    # Read the parquet file
    print("Reading parquet file...")
    df = pq.read_table(input_file).to_pandas()
    
    print(f"Original shape: {df.shape}")
    
    # Identify column types
    # Columns that are game-level (same for both teams)
    game_cols = [
        'gameId', 'gameSourceId', 'seasonType', 'tournament', 'startDate',
        'neutralSite', 'conferenceGame', 'gameType', 'venueId', 'venue',
        'city', 'state', 'spread_actual', 'total_actual', 'spread',
        'overUnder', 'homeMoneyline', 'awayMoneyline', 'spreadOpen',
        'overUnderOpen', 'home_implied_prob', 'away_implied_prob',
        'spread_movement', 'total_movement', 'total_vig',
        'home_true_prob', 'away_true_prob', 'line_movement_magnitude',
        'total_movement_magnitude', 'significant_line_move',
        'significant_total_move', 'expected_home_margin',
        'expected_away_margin', 'implied_home_score', 'implied_away_score',
        'home_favorite', 'favorite_size', 'is_pick_em', 'week', 'season',
        'h2h_games_played', 'h2h_home_win_pct', 'h2h_avg_margin',
        'h2h_last_margin', 'day_of_week', 'is_weekend', 'month',
        'days_since_season_start', 'is_neutral_site', 'is_conference_game'
    ]
    
    # Create home team perspective
    print("Creating home team perspective...")
    home_df = df.copy()
        
    # Create dictionaries for new columns
    home_team_cols = {}
    home_opp_cols = {}
    
    # Columns to skip (already handled or are base names)
    home_skip_cols = {
        'homeTeam', 'homeTeamId', 'homeConference', 'homeConferenceId', 'homePoints',
        'home_won', 'home_rest_days', 'home_is_b2b', 'home_streak', 'home_favorite'
    }
    away_skip_cols = {
        'awayTeam', 'awayTeamId', 'awayConference', 'awayConferenceId', 'awayPoints',
        'away_rest_days', 'away_is_b2b', 'away_streak'
    }
    
    # 1. Rename home_ -> team_
    for col in [c for c in home_df.columns if c.startswith('home_') and c not in home_skip_cols]:
        new_col = col.replace('home_', 'team_')
        home_team_cols[new_col] = home_df[col]
        
    # 2. Rename away_ -> opp_
    for col in [c for c in home_df.columns if c.startswith('away_') and c not in away_skip_cols]:
        new_col = col.replace('away_', 'opp_')
        home_opp_cols[new_col] = home_df[col]

    # 3. Add all new columns at once using assign()
    home_df = home_df.assign(
        team = home_df['homeTeam'],
        teamId = home_df['homeTeamId'],
        opponent = home_df['awayTeam'],
        opponentId = home_df['awayTeamId'],
        conference = home_df['homeConference'],
        conferenceId = home_df['homeConferenceId'],
        opp_conference = home_df['awayConference'],
        opp_conferenceId = home_df['awayConferenceId'],
        is_home = 1,
        points = home_df['homePoints'],
        opp_points = home_df['awayPoints'],
        won = home_df['home_won'],
        margin = home_df['homePoints'] - home_df['awayPoints'],
        rest_days = home_df['home_rest_days'],
        opp_rest_days = home_df['away_rest_days'],
        is_b2b = home_df['home_is_b2b'],
        opp_is_b2b = home_df['away_is_b2b'],
        streak = home_df['home_streak'],
        opp_streak = home_df['away_streak'],
        # Explicitly define team_favorite and opp_favorite
        team_favorite = home_df['home_favorite'],
        opp_favorite = (home_df['is_pick_em'] == 0) & (home_df['home_favorite'] == 0).astype(int),
        **home_team_cols,  # Unpack the dictionaries
        **home_opp_cols
    )
    
    # Create away team perspective
    print("Creating away team perspective...")
    away_df = df.copy()
    
    # Create dictionaries for new columns
    away_team_cols = {}
    away_opp_cols = {}

    # 1. Rename away_ -> team_
    for col in [c for c in away_df.columns if c.startswith('away_') and c not in away_skip_cols]:
        new_col = col.replace('away_', 'team_')
        away_team_cols[new_col] = away_df[col]
        
    # 2. Rename home_ -> opp_
    for col in [c for c in away_df.columns if c.startswith('home_') and c not in home_skip_cols]:
        new_col = col.replace('home_', 'opp_')
        away_opp_cols[new_col] = away_df[col]

    # 3. Add all new columns at once using assign()
    away_df = away_df.assign(
        team = away_df['awayTeam'],
        teamId = away_df['awayTeamId'],
        opponent = away_df['homeTeam'],
        opponentId = away_df['homeTeamId'],
        conference = away_df['awayConference'],
        conferenceId = away_df['awayConferenceId'],
        opp_conference = away_df['homeConference'],
        opp_conferenceId = away_df['homeConferenceId'],
        is_home = 0,
        points = away_df['awayPoints'],
        opp_points = away_df['homePoints'],
        won = (away_df['home_won'] == 0).astype(int),
        margin = away_df['awayPoints'] - away_df['homePoints'],
        rest_days = away_df['away_rest_days'],
        opp_rest_days = away_df['home_rest_days'],
        is_b2b = away_df['away_is_b2b'],
        opp_is_b2b = away_df['home_is_b2b'],
        streak = away_df['away_streak'],
        opp_streak = away_df['home_streak'],
        # Explicitly define team_favorite and opp_favorite
        team_favorite = (away_df['is_pick_em'] == 0) & (away_df['home_favorite'] == 0).astype(int),
        opp_favorite = away_df['home_favorite'],
        **away_team_cols,  # Unpack the dictionaries
        **away_opp_cols
    )
    
    # Get ALL columns from the transformed dataframe
    # We'll build the final column list systematically
    
    # 1. Core game identifiers (keep as-is)
    core_cols = ['gameId', 'gameSourceId', 'seasonType', 'tournament', 
                 'startDate', 'season', 'week', 'venue', 'venueId', 
                 'city', 'state', 'game_type']
    
    # 2. Team identifiers (newly created)
    team_id_cols = ['team', 'teamId', 'opponent', 'opponentId', 
                    'conference', 'conferenceId', 'opp_conference',
                    'opp_conferenceId', 'is_home']
    
    # 3. Game outcome (newly created)
    outcome_cols = ['points', 'opp_points', 'won', 'margin']
    
    # 4. All team_ prefixed columns
    team_cols = sorted([c for c in home_df.columns if c.startswith('team_')])
    
    # 5. All opp_ prefixed columns (opponent stats)
    opp_cols = sorted([c for c in home_df.columns if c.startswith('opp_') 
                       and c not in team_id_cols + outcome_cols])
    
    # 6. All diff_ prefixed columns (differentials)
    diff_cols = sorted([c for c in home_df.columns if c.startswith('diff_')])
    
    # 7. Betting lines and probabilities
    betting_cols = ['spread', 'overUnder', 'spreadOpen', 'overUnderOpen',
                    'homeMoneyline', 'awayMoneyline', 'spread_movement', 
                    'total_movement', 'home_implied_prob', 'away_implied_prob',
                    'home_true_prob', 'away_true_prob', 'line_movement_magnitude',
                    'total_movement_magnitude', 'significant_line_move',
                    'significant_total_move', 'expected_home_margin',
                    'expected_away_margin', 'implied_home_score', 
                    'implied_away_score', 'home_favorite', 'favorite_size',
                    'is_pick_em', 'spread_actual', 'total_actual', 'total_vig']
    
    # 8. Game context
    context_cols = ['neutralSite', 'conferenceGame', 'is_neutral_site', 
                    'is_conference_game', 'rest_days', 'opp_rest_days', 
                    'rest_days_diff', 'is_b2b', 'opp_is_b2b', 'b2b_diff', 
                    'streak', 'opp_streak', 'streak_diff', 'day_of_week', 
                    'is_weekend', 'month', 'days_since_season_start']
    
    # 9. Head-to-head history
    h2h_cols = ['h2h_games_played', 'h2h_home_win_pct', 'h2h_avg_margin', 
                'h2h_last_margin']
    
    # 10. Rankings and polls (keep original names for context)
    ranking_cols = [c for c in home_df.columns if any(x in c for x in 
                    ['_AP_', '_Coaches_', 'AP_rank', 'AP_poll', 'AP_tier',
                     'Coaches_rank', 'Coaches_poll', 'Coaches_tier', 
                     'AP_matchup', 'Coaches_matchup'])]
    
    # Combine all columns
    final_cols = (core_cols + team_id_cols + outcome_cols + team_cols + 
                  opp_cols + diff_cols + betting_cols + context_cols + 
                  h2h_cols + ranking_cols)
    
    # Remove duplicates while preserving order
    final_cols = list(dict.fromkeys(final_cols))
    
    # Keep only columns that exist in the dataframe
    final_cols = [c for c in final_cols if c in home_df.columns]
    
    # Safety check: identify any columns we might be missing
    all_original_cols = set(df.columns)
    all_transformed_cols = set(home_df.columns)
    columns_in_final = set(final_cols)
    
    # Find columns that exist in transformed df but not in our final list
    missing_cols = all_transformed_cols - columns_in_final
    
    # Filter out the old home_/away_ prefixed columns (we don't want these)
    old_prefix_cols = {c for c in missing_cols if 
                       c.startswith(('home_', 'away_')) or 
                       c in ['homeTeam', 'awayTeam', 'homeTeamId', 'awayTeamId',
                             'homeConference', 'awayConference', 'homeConferenceId',
                             'awayConferenceId', 'homePoints', 'awayPoints', 
                             'home_won']}
    
    # These are columns we actually want to exclude
    actually_missing = missing_cols - old_prefix_cols
    
    if actually_missing:
        print(f"\nWARNING: Found {len(actually_missing)} columns not in final selection:")
        for col in sorted(actually_missing):
            print(f"  - {col}")
        print("\nAdding these columns to final output...")
        final_cols.extend(sorted(actually_missing))
    
    # Combine home and away perspectives
    print("Combining perspectives...")
    team_df = pd.concat([
        home_df[final_cols],
        away_df[final_cols]
    ], ignore_index=True)
    
    # Sort by date and team
    team_df = team_df.sort_values(['startDate', 'team']).reset_index(drop=True)
    
    print(f"Final shape: {team_df.shape}")
    print(f"Number of games: {len(df)}")
    print(f"Number of team-game records: {len(team_df)}")
    print(f"Number of columns in original data: {len(df.columns)}")
    print(f"Number of columns in transformed data: {len(team_df.columns)}")
    
    # Verify we're not missing critical columns
    print("\nColumn count by category:")
    print(f"  Core/identifiers: {len([c for c in team_df.columns if c in core_cols + team_id_cols + outcome_cols])}")
    print(f"  Team stats (team_*): {len([c for c in team_df.columns if c.startswith('team_')])}")
    print(f"  Opponent stats (opp_*): {len([c for c in team_df.columns if c.startswith('opp_')])}")
    print(f"  Differentials (diff_*): {len([c for c in team_df.columns if c.startswith('diff_')])}")
    print(f"  Betting/lines: {len([c for c in team_df.columns if c in betting_cols])}")
    print(f"  Context/schedule: {len([c for c in team_df.columns if c in context_cols])}")
    print(f"  Rankings/polls: {len([c for c in team_df.columns if c in ranking_cols])}")
    print(f"  H2H history: {len([c for c in team_df.columns if c in h2h_cols])}")
    
    # Save to parquet
    print(f"Saving to {output_file}...")
    team_df.to_parquet(output_file, index=False, engine='pyarrow')
    
    print("Done!")
    
    # Print sample
    print("\nSample of transformed data:")
    print(team_df[['gameId', 'startDate', 'team', 'opponent', 
                   'is_home', 'points', 'opp_points', 'won', 'margin']].head(10))
    
    return team_df

# Example usage
if __name__ == "__main__":
    input_file = r"C:\Users\DMelv\Documents\artemis\cbb_master_features.parquet"  # Replace with your input file
    output_file = "cbb_team_features.parquet"  # Output file name
    
    team_df = transform_game_to_team(input_file, output_file)
    
    # Optionally, explore the data
    print(f"\nColumns in transformed data: {len(team_df.columns)}")
    print("\nColumn categories:")
    print(f"- Team stats: {len([c for c in team_df.columns if c.startswith('team_')])}")
    print(f"- Opponent stats: {len([c for c in team_df.columns if c.startswith('opp_')])}")
    print(f"- Differential stats: {len([c for c in team_df.columns if c.startswith('diff_')])}")