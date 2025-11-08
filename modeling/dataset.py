"""
PyTorch Dataset for loading basketball features and dynamic data.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sqlalchemy import create_engine
import pickle
from urllib.parse import quote_plus

class BasketballDataset(Dataset):
    """
    Custom dataset that loads static features from parquet 
    and dynamically fetches play-by-play data.
    """
    
    def __init__(self, parquet_path, db_config=None, 
                 mode='train', split_date='2022-01-01',
                 scale_features=True, fetch_plays=False):
        """
        Args:
            parquet_path: Path to master features parquet file
            db_config: Database configuration for dynamic data
            mode: 'train', 'val', or 'test'
            split_date: Date to split train/val/test (default 2022 for better historical coverage)
            scale_features: Whether to normalize features
            fetch_plays: Whether to fetch play-by-play sequences
        """
        # Load master features
        self.df = pd.read_parquet(parquet_path)
        self.df['startDate'] = pd.to_datetime(self.df['startDate'])
        
        # Store db_config for play-by-play fetching
        self.db_config = db_config
        
        # Split data by date (no data leakage)
        if mode == 'train':
            self.df = self.df[self.df['startDate'] < split_date]
        elif mode == 'val':
            val_start = pd.to_datetime(split_date)
            val_end = val_start + pd.Timedelta(days=365)  # 1 year validation
            self.df = self.df[(self.df['startDate'] >= val_start) & 
                            (self.df['startDate'] < val_end)]
        else:  # test
            test_start = pd.to_datetime(split_date) + pd.Timedelta(days=365)
            self.df = self.df[self.df['startDate'] >= test_start]
        
        # Setup database connection if needed
        self.fetch_plays = fetch_plays
        if fetch_plays and db_config:
            self.db_engine = self._create_db_connection(db_config)
        
        # Separate features and targets
        self._prepare_features_and_targets()
        
        # Scale features
        self.scaler = None
        if scale_features:
            self._scale_features(mode)
        
        # Convert to tensors
        self.features_tensor = torch.FloatTensor(self.features.values)
        self.targets_tensor = torch.FloatTensor(self.targets.values)
        
        print(f"Dataset initialized: {len(self)} games in {mode} set")
    
    def _create_db_connection(self, config):
        """Create database connection for dynamic data fetching."""
        if config.get('use_windows_auth', False):
            conn_str = (
                f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                f'SERVER={config["server"]};'
                f'DATABASE={config["database"]};'
                f'Trusted_Connection=yes;'
            )
        else:
            conn_str = (
                f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                f'SERVER={config["server"]};'
                f'DATABASE={config["database"]};'
                f'UID={config["username"]};'
                f'PWD={config["password"]};'
            )
        
        conn_str_quoted = quote_plus(conn_str)
        return create_engine(f'mssql+pyodbc:///?odbc_connect={conn_str_quoted}')
    
    def _prepare_features_and_targets(self):
        """Separate and clean features and targets."""
        # Define target columns
        target_cols = ['spread_actual', 'total_actual', 'home_won']
        
        # --- START FIX (Drop NaN targets) ---
        # Drop rows where any of the targets are missing
        self.df = self.df.dropna(subset=target_cols)
        # --- END FIX ---
        
        # Define columns to exclude (already have or not features)
        exclude_cols = [
            # IDs and metadata
            'gameId', 'gameSourceId', 'startDate', 'homeTeam', 'awayTeam', 
            'venue', 'city', 'state', 'teamId',
            # Actual outcomes (targets)
            'homePoints', 'awayPoints',
            # Conference/team IDs we already have in other forms
            'homeTeamId', 'awayTeamId', 'homeConferenceId', 'awayConferenceId',
            'homeConference', 'awayConference', 
            # Venue ID already captured
            'venueId',
            # Season/type info 
            'seasonType', 'tournament', 'gameType'
        ]
        
        # Combine exclude lists
        all_exclude = target_cols + exclude_cols
        
        # Get feature columns
        feature_cols = [col for col in self.df.columns 
                       if col not in all_exclude]
        
        # Handle missing values
        self.features = self.df[feature_cols].copy()
        
        # --- START FIX (Correct Order) ---
        
        # 1. Convert categorical columns (object or category) to numeric codes
        # This handles NaN in categorical columns by default (converts to -1)
        cat_cols = self.features.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            self.features[col] = pd.Categorical(self.features[col]).codes
        
        # 2. Convert bool columns to numeric
        bool_cols = self.features.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            self.features[col] = self.features[col].astype(int)
        
        # 3. NOW, fill all remaining NaNs (in numeric columns) with 0
        self.features = self.features.fillna(0)
        
        # --- END FIX ---

        # Extract targets
        self.targets = self.df[target_cols].copy()
        
        # Store metadata for lookups
        self.metadata = self.df[['gameId', 'gameSourceId', 'startDate', 
                                 'homeTeam', 'awayTeam', 'homeTeamId', 
                                 'awayTeamId']].copy()
        
        print(f"Features shape: {self.features.shape}")
        print(f"Targets shape: {self.targets.shape}")
    
    def _scale_features(self, mode):
        """Scale features using RobustScaler (handles outliers better)."""
        self.features = self.features.astype(float)

        if mode == 'train':
            self.scaler = RobustScaler()
            self.features.loc[:, :] = self.scaler.fit_transform(self.features)
            # Save scaler for later use
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            # Load existing scaler
            try:
                with open('scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                self.features.loc[:, :] = self.scaler.transform(self.features)
            except FileNotFoundError:
                print("Warning: No scaler found, using unscaled features")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Get a single game sample.
        Returns static features and optionally dynamic play sequences.
        """
        # Get static features and targets
        static_features = self.features_tensor[idx]
        targets = self.targets_tensor[idx]
        
        # Package everything
        sample = {
            'features': static_features,
            'spread_actual': targets[0],
            'total_actual': targets[1],
            'home_won': targets[2],
            'gameId': self.metadata.iloc[idx]['gameId'],
            'gameSourceId': self.metadata.iloc[idx]['gameSourceId']
        }
        
        # Optionally fetch dynamic data
        if self.fetch_plays:
            sample['play_sequences'] = self._get_play_sequences(idx)
            sample['momentum_features'] = self._extract_momentum_features(idx)
        
        return sample
    
    def _get_play_sequences(self, idx):
        """
        Fetch play-by-play sequences for the game.
        """
        game_info = self.metadata.iloc[idx]
        game_source_id = game_info['gameSourceId']
        
        if not self.db_engine:
            return torch.zeros(1, 10)  # Return dummy if no DB connection
        
        query = f"""
        SELECT 
            period, secondsRemaining, homeScore, awayScore,
            homeWinProbability, scoringPlay, shootingPlay, scoreValue,
            isHomeTeam, playType, shotMade, shotRange
        FROM [cbbDB].[dbo].[plays]
        WHERE gameSourceId = '{game_source_id}'
        ORDER BY period, secondsRemaining DESC
        """
        
        try:
            plays_df = pd.read_sql(query, self.db_engine)
            
            if len(plays_df) == 0:
                return torch.zeros(1, 10)
            
            # Engineer play features
            play_features = []
            for _, play in plays_df.iterrows():
                features = [
                    play['period'],
                    play['secondsRemaining'] / 1200.0,  # Normalize to 0-1
                    play['homeScore'],
                    play['awayScore'],
                    play['homeScore'] - play['awayScore'],  # Score differential
                    play['homeWinProbability'] if pd.notna(play['homeWinProbability']) else 0.5,
                    int(play['scoringPlay']) if pd.notna(play['scoringPlay']) else 0,
                    int(play['shootingPlay']) if pd.notna(play['shootingPlay']) else 0,
                    play['scoreValue'] if pd.notna(play['scoreValue']) else 0,
                    int(play['isHomeTeam']) if pd.notna(play['isHomeTeam']) else 0
                ]
                play_features.append(features)
            
            # Convert to tensor and pad/truncate to fixed size
            plays_tensor = torch.FloatTensor(play_features)
            
            # Limit to last 200 plays (or pad if fewer)
            max_plays = 200
            if len(plays_tensor) > max_plays:
                plays_tensor = plays_tensor[-max_plays:]
            elif len(plays_tensor) < max_plays:
                padding = torch.zeros(max_plays - len(plays_tensor), plays_tensor.shape[1])
                plays_tensor = torch.cat([padding, plays_tensor], dim=0)
            
            return plays_tensor
            
        except Exception as e:
            print(f"Error fetching plays for game {game_source_id}: {e}")
            return torch.zeros(200, 10)
    
    def _extract_momentum_features(self, idx):
        """
        Extract momentum and rhythm features from play sequences.
        """
        game_info = self.metadata.iloc[idx]
        game_source_id = game_info['gameSourceId']
        
        if not self.db_engine:
            return torch.zeros(20)  # Return dummy if no DB connection
        
        query = f"""
        SELECT 
            period, secondsRemaining, homeScore, awayScore,
            scoringPlay, scoreValue, isHomeTeam, shotMade
        FROM [cbbDB].[dbo].[plays]
        WHERE gameSourceId = '{game_source_id}'
            AND scoringPlay = 1
        ORDER BY period, secondsRemaining DESC
        """
        
        try:
            scoring_plays = pd.read_sql(query, self.db_engine)
            
            if len(scoring_plays) == 0:
                return torch.zeros(20)
            
            # Calculate momentum features
            home_runs = []
            away_runs = []
            current_run = 0
            last_team = None
            
            for _, play in scoring_plays.iterrows():
                is_home = play['isHomeTeam']
                points = play['scoreValue'] if pd.notna(play['scoreValue']) else 0
                
                if last_team == is_home:
                    current_run += points
                else:
                    if last_team == 1:
                        home_runs.append(current_run)
                    elif last_team == 0:
                        away_runs.append(current_run)
                    current_run = points
                    last_team = is_home
            
            # Add final run
            if last_team == 1:
                home_runs.append(current_run)
            elif last_team == 0:
                away_runs.append(current_run)
            
            # Compute momentum statistics
            momentum_features = [
                np.mean(home_runs) if home_runs else 0,
                np.max(home_runs) if home_runs else 0,
                np.std(home_runs) if home_runs else 0,
                len(home_runs),
                np.mean(away_runs) if away_runs else 0,
                np.max(away_runs) if away_runs else 0,
                np.std(away_runs) if away_runs else 0,
                len(away_runs),
                # Add more features as needed
            ]
            
            # Pad to fixed size
            while len(momentum_features) < 20:
                momentum_features.append(0)
            
            return torch.FloatTensor(momentum_features[:20])
            
        except Exception as e:
            print(f"Error extracting momentum for game {game_source_id}: {e}")
            return torch.zeros(20)


# Example usage
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'server': 'your_server',
        'database': 'cbbDB',
        'username': 'your_username',
        'password': 'your_password',
        'use_windows_auth': True  # Set to False if using SQL auth
    }
    
    # Create datasets
    train_dataset = BasketballDataset(
        parquet_path='features.parquet',
        db_config=db_config,
        mode='train',
        split_date='2022-01-01',  # Use 2005-2022 for training
        scale_features=True,
        fetch_plays=True
    )
    
    val_dataset = BasketballDataset(
        parquet_path='features.parquet',
        db_config=db_config,
        mode='val',
        split_date='2022-01-01',  # Use 2022-2023 for validation
        scale_features=True,
        fetch_plays=True
    )
    
    test_dataset = BasketballDataset(
        parquet_path='features.parquet',
        db_config=db_config,
        mode='test',
        split_date='2022-01-01',  # Use 2023+ for testing
        scale_features=True,
        fetch_plays=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test a batch
    for batch in train_loader:
        print(f"Features shape: {batch['features'].shape}")
        print(f"Play sequences shape: {batch['play_sequences'].shape}")
        print(f"Momentum features shape: {batch['momentum_features'].shape}")
        break