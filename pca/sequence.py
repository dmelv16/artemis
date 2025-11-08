import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class GameSequenceDataset(Dataset):
    """
    PyTorch Dataset for loading game sequences.
    Each sample contains the sequence of past games for both teams.
    """
    
    def __init__(self, features_df, sequence_length=10, prediction_horizon=0):
        """
        Args:
            features_df: DataFrame with rolling PCA/cluster features
            sequence_length: Number of past games to use as sequence
            prediction_horizon: How many days ahead to predict (0 = same day)
        """
        self.features_df = features_df.sort_values(['teamId', 'game_date'])
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Identify feature columns (PCA components and clusters)
        self.feature_cols = [col for col in features_df.columns 
                           if ('_PC' in col or '_cluster' in col)]
        
        # Prepare sequences for each team
        self.sequences = self._prepare_sequences()
        
    def _prepare_sequences(self):
        """
        Create sequences of games for each team over time
        """
        sequences = []
        
        # Group by team and create sequences
        for team_id in self.features_df['teamId'].unique():
            team_data = self.features_df[self.features_df['teamId'] == team_id].copy()
            team_data = team_data.sort_values('game_date')
            
            # Create sliding windows
            for i in range(self.sequence_length, len(team_data)):
                # Get sequence of past games
                seq_data = team_data.iloc[i-self.sequence_length:i]
                
                # Get current game (target)
                target_data = team_data.iloc[i]
                
                sequences.append({
                    'team_id': team_id,
                    'sequence_dates': seq_data['game_date'].values,
                    'sequence_features': seq_data[self.feature_cols].values,
                    'target_date': target_data['game_date'],
                    'target_gameId': target_data.get('gameId', None),
                    'is_home': target_data.get('is_home', 1)
                })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


class DynamicTeamStateModel(nn.Module):
    """
    LSTM-based model that learns from evolving team states over time.
    This model understands that features evolve and learns patterns
    in the sequences rather than fixed meanings.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, 
                 dropout=0.2, attention=True):
        """
        Args:
            input_dim: Number of input features (PCA components + clusters)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            attention: Whether to use attention mechanism
        """
        super(DynamicTeamStateModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = attention
        
        # LSTM for processing team sequences
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism (if enabled)
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Team state encoder
        self.team_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Matchup predictor (combines both teams)
        self.matchup_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output heads for different predictions
        self.spread_head = nn.Linear(hidden_dim // 2, 1)
        self.total_head = nn.Linear(hidden_dim // 2, 1)
        self.winner_head = nn.Linear(hidden_dim // 2, 1)  # Sigmoid for probability
        
    def forward_single_team(self, x):
        """
        Process a single team's sequence
        
        Args:
            x: Tensor of shape (batch_size, sequence_length, input_dim)
        
        Returns:
            Team representation vector
        """
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        
        if self.use_attention:
            # Apply attention to focus on important games in sequence
            attention_weights = self.attention(lstm_out)
            attention_weights = torch.softmax(attention_weights, dim=1)
            
            # Weighted sum of LSTM outputs
            context = torch.sum(lstm_out * attention_weights, dim=1)
        else:
            # Use final hidden state
            context = lstm_out[:, -1, :]
        
        # Encode team state
        team_state = self.team_encoder(context)
        
        return team_state
    
    def forward(self, home_seq, away_seq):
        """
        Forward pass for game prediction
        
        Args:
            home_seq: Home team sequence (batch_size, seq_len, input_dim)
            away_seq: Away team sequence (batch_size, seq_len, input_dim)
        
        Returns:
            Dictionary with predictions
        """
        # Encode both teams
        home_state = self.forward_single_team(home_seq)
        away_state = self.forward_single_team(away_seq)
        
        # Combine team states for matchup
        matchup = torch.cat([home_state, away_state], dim=1)
        matchup_features = self.matchup_layers(matchup)
        
        # Make predictions
        spread_pred = self.spread_head(matchup_features)
        total_pred = self.total_head(matchup_features)
        winner_prob = torch.sigmoid(self.winner_head(matchup_features))
        
        return {
            'spread': spread_pred,
            'total': total_pred,
            'win_probability': winner_prob
        }


class SequenceTrainer:
    """
    Training pipeline for the dynamic sequence model
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        
    def prepare_matchup_data(self, features_df, games_df, sequence_length=10):
        """
        Prepare data for matchup prediction
        
        Args:
            features_df: DataFrame with team sequences from RollingWindowClusterPipeline
            games_df: DataFrame with game matchups and outcomes
            sequence_length: Length of sequence to use
        
        Returns:
            Training data ready for the model
        """
        matchup_data = []
        
        for _, game in games_df.iterrows():
            home_id = game['homeTeamId']
            away_id = game['awayTeamId']
            game_date = game['startDate']
            
            # Get sequences for both teams leading up to this game
            home_seq = self._get_team_sequence(
                features_df, home_id, game_date, sequence_length
            )
            away_seq = self._get_team_sequence(
                features_df, away_id, game_date, sequence_length
            )
            
            if home_seq is not None and away_seq is not None:
                matchup_data.append({
                    'game_id': game['gameId'],
                    'date': game_date,
                    'home_sequence': home_seq,
                    'away_sequence': away_seq,
                    'spread_actual': game.get('spread_actual', 0),
                    'total_actual': game.get('total_actual', 0),
                    'home_won': game.get('home_won', 0)
                })
        
        return matchup_data
    
    def _get_team_sequence(self, features_df, team_id, game_date, seq_length):
        """
        Get sequence of past games for a team
        """
        # Filter to team's games before the target date
        team_data = features_df[
            (features_df['teamId'] == team_id) &
            (features_df['game_date'] < game_date)
        ].sort_values('game_date')
        
        if len(team_data) < seq_length:
            return None
        
        # Get the most recent seq_length games
        recent_games = team_data.tail(seq_length)
        
        # Extract feature columns
        feature_cols = [col for col in features_df.columns 
                       if ('_PC' in col or '_cluster' in col)]
        
        return recent_games[feature_cols].values
    
    def train_epoch(self, dataloader, optimizer, criterion_spread, 
                   criterion_total, criterion_winner):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Move data to device
            home_seq = batch['home_seq'].to(self.device)
            away_seq = batch['away_seq'].to(self.device)
            spread_actual = batch['spread'].to(self.device)
            total_actual = batch['total'].to(self.device)
            home_won = batch['winner'].to(self.device)
            
            # Forward pass
            predictions = self.model(home_seq, away_seq)
            
            # Calculate losses
            loss_spread = criterion_spread(predictions['spread'], spread_actual)
            loss_total = criterion_total(predictions['total'], total_actual)
            loss_winner = criterion_winner(predictions['win_probability'], home_won)
            
            # Combined loss
            loss = loss_spread + 0.5 * loss_total + 0.5 * loss_winner
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """
        Evaluate model performance
        """
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in dataloader:
                home_seq = batch['home_seq'].to(self.device)
                away_seq = batch['away_seq'].to(self.device)
                
                preds = self.model(home_seq, away_seq)
                
                predictions.append({
                    'spread': preds['spread'].cpu().numpy(),
                    'total': preds['total'].cpu().numpy(),
                    'win_prob': preds['win_probability'].cpu().numpy()
                })
                
                actuals.append({
                    'spread': batch['spread'].numpy(),
                    'total': batch['total'].numpy(),
                    'winner': batch['winner'].numpy()
                })
        
        return predictions, actuals


class DynamicPipeline:
    """
    Complete pipeline combining rolling features and sequence modeling
    """
    
    def __init__(self, features_path, games_path, output_dir='dynamic_models'):
        self.features_path = features_path
        self.games_path = games_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def run_temporal_backtest(self, sequence_length=10, retrain_days=30):
        """
        Run backtesting with proper temporal ordering, training, and scaling
        """
        print("Loading data...")
        features_df = pd.read_parquet(self.features_path)
        games_df = pd.read_parquet(self.games_path)
        
        # Ensure dates are datetime
        features_df['game_date'] = pd.to_datetime(features_df['game_date'])
        games_df['startDate'] = pd.to_datetime(games_df['startDate'])
        
        # Get feature dimension
        feature_cols = [col for col in features_df.columns 
                    if ('_PC' in col or '_cluster' in col)]
        input_dim = len(feature_cols)
        
        print(f"Input dimension: {input_dim} features")
        
        # Initialize results storage
        all_predictions = []
        
        # Get unique dates for testing
        test_dates = games_df['startDate'].unique()
        test_dates.sort()
        
        # Skip early dates without enough history
        min_date = test_dates[365]  # Start after 1 year of data
        test_dates = test_dates[test_dates >= min_date]
        
        current_model = None
        last_train_date = None
        trainer = None  # Keep trainer persistent for scaler
        
        for test_date in test_dates:
            # Check if we need to retrain
            if (current_model is None or 
                (test_date - last_train_date).days >= retrain_days):
                
                print(f"\nRetraining model for {test_date.strftime('%Y-%m-%d')}")
                
                # Get training data (all games before test_date)
                train_games = games_df[games_df['startDate'] < test_date]
                
                if len(train_games) < 1000:
                    continue
                
                # Initialize new model
                current_model = DynamicTeamStateModel(
                    input_dim=input_dim,
                    hidden_dim=128,
                    num_layers=2,
                    dropout=0.2,
                    attention=True
                )
                
                trainer = SequenceTrainer(current_model)
                
                # Prepare training data
                train_data = trainer.prepare_matchup_data(
                    features_df, train_games, sequence_length
                )
                
                if len(train_data) < 100:  # Skip if not enough training data
                    continue
                
                # ============= FIX #1: SCALE THE DATA =============
                print(f"  Preparing and scaling {len(train_data)} game sequences...")
                
                # Extract all sequences
                train_home_seqs = np.array([item['home_sequence'] for item in train_data])
                train_away_seqs = np.array([item['away_sequence'] for item in train_data])
                
                # Get dimensions
                n_samples, seq_len, n_features = train_home_seqs.shape
                
                # Reshape and combine for scaling
                all_train_seqs = np.concatenate([
                    train_home_seqs.reshape(-1, n_features), 
                    train_away_seqs.reshape(-1, n_features)
                ])
                
                # Fit new scaler
                scaler = StandardScaler()
                scaler.fit(all_train_seqs)
                trainer.scaler = scaler
                
                # Transform training sequences
                for item in train_data:
                    item['home_sequence'] = scaler.transform(item['home_sequence'])
                    item['away_sequence'] = scaler.transform(item['away_sequence'])
                
                print(f"  Scaler fitted on {len(all_train_seqs)} vectors.")
                
                # ============= FIX #2: ACTUAL TRAINING =============
                print(f"  Training on {len(train_data)} game sequences...")
                
                # Custom collate function
                def collate_fn(batch):
                    home_seqs = torch.FloatTensor(np.array([item['home_sequence'] for item in batch]))
                    away_seqs = torch.FloatTensor(np.array([item['away_sequence'] for item in batch]))
                    spreads = torch.FloatTensor(np.array([item['spread_actual'] for item in batch])).unsqueeze(1)
                    totals = torch.FloatTensor(np.array([item['total_actual'] for item in batch])).unsqueeze(1)
                    winners = torch.FloatTensor(np.array([item['home_won'] for item in batch])).unsqueeze(1)
                    
                    return {
                        'home_seq': home_seqs, 'away_seq': away_seqs,
                        'spread': spreads, 'total': totals, 'winner': winners
                    }
                
                # Create DataLoader
                train_loader = DataLoader(
                    train_data, 
                    batch_size=64, 
                    shuffle=True, 
                    collate_fn=collate_fn
                )
                
                # Setup training
                optimizer = optim.Adam(current_model.parameters(), lr=0.001)
                criterion_spread = nn.MSELoss()
                criterion_total = nn.MSELoss()
                criterion_winner = nn.BCELoss()
                
                # Training epochs
                num_epochs = 5
                for epoch in range(num_epochs):
                    loss = trainer.train_epoch(
                        train_loader, optimizer, 
                        criterion_spread, criterion_total, criterion_winner
                    )
                    print(f"    Epoch {epoch+1}/{num_epochs}, Train Loss: {loss:.4f}")
                
                last_train_date = test_date
            
            # ============= MAKE PREDICTIONS WITH SCALING =============
            if current_model is not None and trainer is not None:
                test_games = games_df[games_df['startDate'] == test_date]
                
                for _, game in test_games.iterrows():
                    # Get sequences
                    home_seq = trainer._get_team_sequence(
                        features_df, game['homeTeamId'], 
                        test_date, sequence_length
                    )
                    away_seq = trainer._get_team_sequence(
                        features_df, game['awayTeamId'], 
                        test_date, sequence_length
                    )
                    
                    if home_seq is not None and away_seq is not None:
                        # SCALE the sequences before prediction
                        home_seq_scaled = trainer.scaler.transform(home_seq)
                        away_seq_scaled = trainer.scaler.transform(away_seq)
                        
                        # Convert to tensors
                        home_tensor = torch.FloatTensor(home_seq_scaled).unsqueeze(0).to(trainer.device)
                        away_tensor = torch.FloatTensor(away_seq_scaled).unsqueeze(0).to(trainer.device)
                        
                        # Get prediction
                        current_model.eval()
                        with torch.no_grad():
                            pred = current_model(home_tensor, away_tensor)
                        
                        all_predictions.append({
                            'game_id': game['gameId'],
                            'date': test_date,
                            'spread_pred': pred['spread'].cpu().item(),
                            'total_pred': pred['total'].cpu().item(),
                            'win_prob': pred['win_probability'].cpu().item(),
                            'spread_actual': game.get('spread_actual', np.nan),
                            'total_actual': game.get('total_actual', np.nan),
                            'home_won': game.get('home_won', np.nan)
                        })
        
        return pd.DataFrame(all_predictions)


# Example usage
if __name__ == "__main__":
    # This assumes you've already run the RollingWindowClusterPipeline
    # to generate the features_df with evolving PCA/cluster features
    
    pipeline = DynamicPipeline(
        features_path='rolling_clusters/rolling_features.parquet',
        games_path='cbb_features.parquet',  # Original games data
        output_dir='dynamic_models'
    )
    
    # Run temporal backtest
    results = pipeline.run_temporal_backtest(
        sequence_length=10,  # Use last 10 games
        retrain_days=30      # Retrain monthly
    )
    
    # Calculate performance
    results['spread_error'] = results['spread_pred'] - results['spread_actual']
    mae = results['spread_error'].abs().mean()
    print(f"\nSpread MAE: {mae:.2f}")
    
    # Save results
    results.to_csv('dynamic_model_results.csv', index=False)
    print(f"Results saved to dynamic_model_results.csv")