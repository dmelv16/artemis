"""
Calculates head-to-head historical features.
"""
import pandas as pd

class HeadToHeadProcessor:
    def __init__(self, db_conn):
        self.db = db_conn
        
    def add_features(self, df):
        print("Adding head-to-head historical features...")
        h2h_features = []
        
        for idx, game in df.iterrows():
            h2h_data = self._get_h2h_stats(
                game['homeTeamId'], 
                game['awayTeamId'], 
                game['startDate']
            )
            h2h_data['gameId'] = game['gameId']
            h2h_features.append(h2h_data)
        
        h2h_df = pd.DataFrame(h2h_features)
        df = df.merge(h2h_df, on='gameId', how='left')
        
        print("Added head-to-head features")
        return df
    
    def _get_h2h_stats(self, home_id, away_id, game_date):
        query = f"""
        SELECT g.homePoints, g.awayPoints, g.homeTeamId, g.awayTeamId
        FROM games g
        WHERE ((g.homeTeamId = {home_id} AND g.awayTeamId = {away_id})
               OR (g.homeTeamId = {away_id} AND g.awayTeamId = {home_id}))
            AND g.startDate < '{game_date}'
            AND g.status = 'Final'
        ORDER BY g.startDate DESC
        """
        
        try:
            h2h_games = self.db.query(query)
            
            if len(h2h_games) > 0:
                stats = self._calculate_h2h_metrics(h2h_games, home_id)
                return stats
        except:
            pass
        
        return {
            'h2h_games_played': 0,
            'h2h_home_win_pct': 0.5,
            'h2h_avg_margin': 0,
            'h2h_last_margin': 0
        }
    
    def _calculate_h2h_metrics(self, games, home_id):
        home_wins = 0
        total_margin = 0
        
        for _, game in games.iterrows():
            if game['homeTeamId'] == home_id:
                margin = game['homePoints'] - game['awayPoints']
                if margin > 0:
                    home_wins += 1
            else:
                margin = game['awayPoints'] - game['homePoints']
                if margin < 0:
                    home_wins += 1
                margin = -margin
            total_margin += margin
        
        return {
            'h2h_games_played': len(games),
            'h2h_home_win_pct': home_wins / len(games),
            'h2h_avg_margin': total_margin / len(games),
            'h2h_last_margin': games.iloc[0]['homePoints'] - games.iloc[0]['awayPoints']
        }