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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
from collections import deque
import os  # Added for OS detection
warnings.filterwarnings('ignore')

# ============= GPU OPTIMIZATION: Enable all optimizations =============
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')  # Allow TF32 on Ampere GPUs

from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_


class FastGameSequenceDataset(Dataset):
    """Ultra-fast dataset that keeps data on CPU but optimizes transfer"""
    
    def __init__(self, matchup_data, pin_memory=True, device=None):
        # Keep on CPU but optimize for transfer
        self.home_seqs = np.array([item['home_sequence'] for item in matchup_data], dtype=np.float32)
        self.away_seqs = np.array([item['away_sequence'] for item in matchup_data], dtype=np.float32)
        self.spreads = np.array([item['spread_actual'] for item in matchup_data], dtype=np.float32)
        self.totals = np.array([item['total_actual'] for item in matchup_data], dtype=np.float32)
        self.winners = np.array([item['home_won'] for item in matchup_data], dtype=np.float32)
        
        self.num_samples = len(matchup_data)
        
        # OPTIMIZATION: Pre-allocate contiguous memory and convert to tensors
        # This ensures memory is laid out optimally for GPU transfer
        if os.name == 'nt':  # Windows optimization
            # Store as contiguous tensors for faster transfer
            self.home_seqs = torch.from_numpy(self.home_seqs).contiguous()
            self.away_seqs = torch.from_numpy(self.away_seqs).contiguous()
            self.spreads = torch.from_numpy(self.spreads).contiguous()
            self.totals = torch.from_numpy(self.totals).contiguous()
            self.winners = torch.from_numpy(self.winners).contiguous()
            
            # Optional: Pre-transfer to GPU if enough VRAM (for smaller datasets)
            if device and torch.cuda.is_available() and self.num_samples < 10000:
                self.home_seqs = self.home_seqs.to(device, non_blocking=True)
                self.away_seqs = self.away_seqs.to(device, non_blocking=True)
                self.spreads = self.spreads.to(device, non_blocking=True)
                self.totals = self.totals.to(device, non_blocking=True)
                self.winners = self.winners.to(device, non_blocking=True)
                self.on_gpu = True
            else:
                self.on_gpu = False
        else:  # Linux/Mac
            if pin_memory and torch.cuda.is_available():
                self.home_seqs = torch.from_numpy(self.home_seqs).pin_memory()
                self.away_seqs = torch.from_numpy(self.away_seqs).pin_memory()
                self.spreads = torch.from_numpy(self.spreads).pin_memory()
                self.totals = torch.from_numpy(self.totals).pin_memory()
                self.winners = torch.from_numpy(self.winners).pin_memory()
            else:
                self.home_seqs = torch.from_numpy(self.home_seqs)
                self.away_seqs = torch.from_numpy(self.away_seqs)
                self.spreads = torch.from_numpy(self.spreads)
                self.totals = torch.from_numpy(self.totals)
                self.winners = torch.from_numpy(self.winners)
            self.on_gpu = False
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'home_seq': self.home_seqs[idx],
            'away_seq': self.away_seqs[idx],
            'spread': self.spreads[idx],
            'total': self.totals[idx],
            'winner': self.winners[idx]
        }


class OptimizedDynamicModel(nn.Module):
    """Faster model with better convergence"""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Single bidirectional LSTM is often faster and equally effective
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Simplified attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1)
        )
        
        # Faster architecture with fewer layers
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # GELU often trains faster
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Combined matchup processor
        self.matchup = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Single output head
        self.output = nn.Linear(hidden_dim // 2, 3)
        
        # Initialize weights for faster convergence
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def process_sequence(self, x):
        """Process a single sequence through LSTM and attention"""
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        return self.encoder(context)
    
    def forward(self, home_seq, away_seq):
        # Process both sequences
        home_state = self.process_sequence(home_seq)
        away_state = self.process_sequence(away_seq)
        
        # Combine and predict
        combined = torch.cat([home_state, away_state], dim=1)
        features = self.matchup(combined)
        outputs = self.output(features)
        
        return {
            'spread': outputs[:, 0:1],
            'total': outputs[:, 1:2], 
            'win_probability': outputs[:, 2:3]
        }


class AdaptiveTrainer:
    """Smart trainer with early stopping and adaptive learning"""
    
    def __init__(self, model, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.scaler_amp = GradScaler()
        
        # For caching preprocessed data
        self.cache = {}
        self.cache_size = 10
        
        # CHANGED: Relaxed early stopping parameters
        self.patience = 5  # Increased from 3
        self.min_delta = 0.001  # Reduced from 0.01 for more sensitivity
        
    def train_adaptive(self, train_loader, max_epochs=30, min_epochs=8):  # Increased min_epochs from 5
        """Train with early stopping and adaptive learning rate"""
        
        # Use OneCycleLR for faster convergence
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            fused=torch.cuda.is_available()
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=max_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='linear'
        )
        
        criterion_mse = nn.MSELoss()
        criterion_bce = nn.BCEWithLogitsLoss()
        
        # CHANGED: Advanced loss weighting with dynamic adjustment
        # Start with heavy emphasis on spread, gradually balance
        spread_weight_start = 3.0
        spread_weight_end = 2.0
        total_weight = 0.7
        win_weight = 0.3
        
        best_loss = float('inf')
        patience_counter = 0
        loss_history = deque(maxlen=5)
        
        # OPTIMIZATION: Pre-allocate tensors for Windows to avoid repeated allocation
        if os.name == 'nt':
            batch_home = torch.empty((train_loader.batch_size, 10, 161), device=self.device, dtype=torch.float32)
            batch_away = torch.empty((train_loader.batch_size, 10, 161), device=self.device, dtype=torch.float32)
        
        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0
            
            # CHANGED: Dynamic weight adjustment - heavier spread focus early, then gradual balance
            progress = epoch / max_epochs
            spread_weight = spread_weight_start - (spread_weight_start - spread_weight_end) * progress
            
            for batch_idx, batch in enumerate(train_loader):
                # OPTIMIZATION: Faster GPU transfer on Windows
                if os.name == 'nt' and not hasattr(train_loader.dataset, 'on_gpu'):
                    # Use non_blocking for all transfers
                    home_seq = batch['home_seq'].to(self.device, non_blocking=True)
                    away_seq = batch['away_seq'].to(self.device, non_blocking=True)
                    spread = batch['spread'].unsqueeze(1).to(self.device, non_blocking=True)
                    total = batch['total'].unsqueeze(1).to(self.device, non_blocking=True)
                    winner = batch['winner'].unsqueeze(1).to(self.device, non_blocking=True)
                    
                    # Synchronize only once after all transfers
                    if batch_idx == 0:
                        torch.cuda.synchronize()
                else:
                    # Standard transfer for Linux or pre-loaded GPU data
                    home_seq = batch['home_seq'] if hasattr(train_loader.dataset, 'on_gpu') and train_loader.dataset.on_gpu else batch['home_seq'].to(self.device, non_blocking=True)
                    away_seq = batch['away_seq'] if hasattr(train_loader.dataset, 'on_gpu') and train_loader.dataset.on_gpu else batch['away_seq'].to(self.device, non_blocking=True)
                    spread = batch['spread'].unsqueeze(1) if hasattr(train_loader.dataset, 'on_gpu') and train_loader.dataset.on_gpu else batch['spread'].unsqueeze(1).to(self.device, non_blocking=True)
                    total = batch['total'].unsqueeze(1) if hasattr(train_loader.dataset, 'on_gpu') and train_loader.dataset.on_gpu else batch['total'].unsqueeze(1).to(self.device, non_blocking=True)
                    winner = batch['winner'].unsqueeze(1) if hasattr(train_loader.dataset, 'on_gpu') and train_loader.dataset.on_gpu else batch['winner'].unsqueeze(1).to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with autocast():
                    preds = self.model(home_seq, away_seq)
                    
                    # CHANGED: Focused loss weighting with dynamic spread emphasis
                    spread_loss = criterion_mse(preds['spread'], spread)
                    total_loss = criterion_mse(preds['total'], total)
                    win_loss = criterion_bce(preds['win_probability'], winner)
                    
                    # Weighted combination with heavy spread emphasis
                    loss = (spread_weight * spread_loss + 
                           total_weight * total_loss + 
                           win_weight * win_loss)
                
                self.scaler_amp.scale(loss).backward()
                
                # Gradient clipping for stability
                self.scaler_amp.unscale_(optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler_amp.step(optimizer)
                self.scaler_amp.update()
                scheduler.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            
            # CHANGED: More nuanced early stopping logic
            if epoch >= min_epochs:
                # Check if loss is plateauing (but with more tolerance)
                if len(loss_history) >= 5:
                    # Convert deque to list for slicing
                    recent_losses = list(loss_history)[-3:]
                    recent_improvement = max(recent_losses) - min(recent_losses)
                    if recent_improvement < self.min_delta:
                        print(f"    Early stopping at epoch {epoch+1}, loss plateaued (improvement: {recent_improvement:.6f})")
                        break
                
                # Check if loss increased significantly
                if avg_loss > best_loss - self.min_delta:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"    Early stopping at epoch {epoch+1}, patience exhausted")
                        break
                else:
                    patience_counter = 0
                    best_loss = min(best_loss, avg_loss)
            else:
                # Update best loss even during min_epochs period
                best_loss = min(best_loss, avg_loss)
            
            if epoch % 3 == 0:
                print(f"    Epoch {epoch+1}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}, Spread Weight: {spread_weight:.2f}")
        
        return avg_loss


class OptimizedPipeline:
    """Pipeline with all optimizations"""
    
    def __init__(self, features_path, games_path):
        self.features_path = features_path
        self.games_path = games_path
        
        # Pre-load and cache data
        print("Pre-loading data into memory...")
        self.features_df = pd.read_parquet(features_path)
        self.games_df = pd.read_parquet(games_path)
        
        # Pre-process dates
        self.features_df['startDate'] = pd.to_datetime(self.features_df['startDate'])
        self.games_df['startDate'] = pd.to_datetime(self.games_df['startDate'])
        self.features_df = self.features_df.rename(columns={'startDate': 'game_date'})
        self.games_df['game_day'] = self.games_df['startDate'].dt.normalize()
        
        # Pre-compute feature columns
        self.feature_cols = [col for col in self.features_df.columns 
                            if ('_PC' in col or '_cluster' in col)]
        self.input_dim = len(self.feature_cols)
        
        # Create team lookup for ultra-fast access
        print("Building team lookup tables...")
        self.team_lookup = {}
        for team_id in self.features_df['teamId'].unique():
            team_data = self.features_df[self.features_df['teamId'] == team_id].sort_values('game_date')
            # Convert to numpy for faster access
            self.team_lookup[team_id] = {
                'dates': team_data['game_date'].values,
                'features': team_data[self.feature_cols].values.astype(np.float32)
            }
        
        # Pre-compute unique games
        self.unique_games = self.games_df[self.games_df['is_home'] == 1].copy()
        
        print(f"Loaded {len(self.features_df)} features, {len(self.games_df)} games")
        print(f"  -> {len(self.unique_games)} unique games (home team perspective)")
        print(f"Input dimension: {self.input_dim}")
        
    def get_sequence_vectorized(self, team_id, game_date, seq_length=10):
        """Vectorized sequence retrieval"""
        if team_id not in self.team_lookup:
            return None
            
        team_data = self.team_lookup[team_id]
        mask = team_data['dates'] < game_date
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) < seq_length:
            return None
            
        # Get last seq_length games
        indices = valid_indices[-seq_length:]
        return team_data['features'][indices]
    
    def prepare_batch_parallel(self, games_batch, seq_length=10):
        """Parallel batch preparation"""
        matchup_data = []
        
        for _, game in games_batch.iterrows():
            home_seq = self.get_sequence_vectorized(game['teamId'], game['startDate'], seq_length)
            away_seq = self.get_sequence_vectorized(game['opponentId'], game['startDate'], seq_length)
            
            if home_seq is not None and away_seq is not None:
                matchup_data.append({
                    'home_sequence': home_seq,
                    'away_sequence': away_seq,
                    'spread_actual': game['points'] - game['opp_points'],
                    'total_actual': game['points'] + game['opp_points'],
                    'home_won': 1 if game['points'] > game['opp_points'] else 0
                })
        
        return matchup_data
    
    def run_fast_backtest(self, sequence_length=10, retrain_days=30, 
                          batch_size=512, num_workers=8):
        """Optimized backtesting pipeline"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # OPTIMIZATION: Adjust batch size for Windows
        if os.name == 'nt' and torch.cuda.is_available():
            # Larger batch size on Windows to compensate for single-threaded loading
            batch_size = min(1024, batch_size * 2)
            print(f"  Windows detected: Using batch size {batch_size} for better GPU utilization")
        
        test_dates = np.sort(self.games_df['game_day'].unique())
        min_date = test_dates[365]
        test_dates = test_dates[test_dates >= min_date]
        
        all_predictions = []
        current_model = None
        last_train_date = None
        trainer = None
        scaler = StandardScaler()
        
        # Reuse model weights when possible
        model_checkpoint = None
        
        # CHANGED: Track batch predictions for detailed MAE reporting
        batch_predictions = []
        
        for i, test_date in enumerate(test_dates):
            # Convert numpy datetime to days for comparison
            if last_train_date is None:
                days_since_train = float('inf')
            else:
                # Handle numpy datetime64 objects
                days_diff = test_date - last_train_date
                days_since_train = int(days_diff / np.timedelta64(1, 'D'))
            
            should_retrain = current_model is None or days_since_train >= retrain_days
            
            if should_retrain:
                
                # CHANGED: Print stats for the *previous* batch right before retraining
                if len(batch_predictions) > 0:
                    # Calculate batch MAE (predictions since last retrain)
                    batch_df = pd.DataFrame(batch_predictions)
                    batch_spread_mae = (batch_df['spread_pred'] - batch_df['spread_actual']).abs().mean()
                    batch_total_mae = (batch_df['total_pred'] - batch_df['total_actual']).abs().mean()
                    
                    # Calculate overall MAE (now only done once per retrain)
                    all_df = pd.DataFrame(all_predictions)
                    overall_spread_mae = (all_df['spread_pred'] - all_df['spread_actual']).abs().mean()
                    overall_total_mae = (all_df['total_pred'] - all_df['total_actual']).abs().mean()
                    
                    print(f"\n  STATS FOR PREVIOUS BATCH ({len(batch_predictions)} preds):")
                    print(f"    Batch MAE (S/T):   {batch_spread_mae:.3f} / {batch_total_mae:.3f}")
                    print(f"    Overall MAE (S/T): {overall_spread_mae:.3f} / {overall_total_mae:.3f} ({len(all_predictions)} total)")

                
                print(f"\nRetraining for {pd.Timestamp(test_date).strftime('%Y-%m-%d')} ({i+1}/{len(test_dates)})")
                
                # Get training data
                train_games = self.unique_games[self.unique_games['startDate'] < test_date]
                if len(train_games) < 1000:
                    continue
                
                # Prepare data in parallel
                print(f"  Preparing {len(train_games)} games...")
                with ThreadPoolExecutor(max_workers=4) as executor:
                    # Split games into chunks for parallel processing
                    chunk_size = len(train_games) // 4
                    chunks = [train_games.iloc[i:i+chunk_size] for i in range(0, len(train_games), chunk_size)]
                    results = executor.map(lambda c: self.prepare_batch_parallel(c, sequence_length), chunks)
                    train_data = [item for chunk in results for item in chunk]
                
                if len(train_data) < 100:
                    continue
                
                # Fast scaling
                print(f"  Scaling {len(train_data)} sequences...")
                all_seqs = np.vstack([
                    np.vstack([d['home_sequence'] for d in train_data]),
                    np.vstack([d['away_sequence'] for d in train_data])
                ])
                all_seqs = np.nan_to_num(all_seqs, nan=0.0, posinf=0.0, neginf=0.0)
                scaler.fit(all_seqs)
                
                # Scale in-place for speed
                for item in train_data:
                    item['home_sequence'] = scaler.transform(
                        np.nan_to_num(item['home_sequence'], nan=0.0, posinf=0.0, neginf=0.0)
                    )
                    item['away_sequence'] = scaler.transform(
                        np.nan_to_num(item['away_sequence'], nan=0.0, posinf=0.0, neginf=0.0)
                    )
                
                # Create model (reuse weights if available)
                current_model = OptimizedDynamicModel(
                    input_dim=self.input_dim,
                    hidden_dim=256,
                    num_layers=2,
                    dropout=0.3
                )
                
                # Load previous weights as starting point (transfer learning)
                if model_checkpoint is not None:
                    try:
                        current_model.load_state_dict(model_checkpoint)
                        print("  Loaded previous model weights for transfer learning")
                    except:
                        pass
                
                # Create dataset and loader with Windows optimizations
                if os.name == 'nt':  # Windows-specific optimizations
                    # Pass device for potential GPU pre-loading on smaller datasets
                    dataset = FastGameSequenceDataset(train_data, pin_memory=False, device=device if len(train_data) < 10000 else None)
                    
                    # Windows optimizations for DataLoader
                    train_loader = DataLoader(
                        dataset, 
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,  # Single-threaded on Windows
                        pin_memory=not dataset.on_gpu,  # Only pin if not already on GPU
                        persistent_workers=False,
                        prefetch_factor=None,
                        drop_last=True  # Consistent batch sizes for better GPU utilization
                    )
                else:  # Linux/Mac
                    dataset = FastGameSequenceDataset(train_data, pin_memory=True)
                    train_loader = DataLoader(
                        dataset, 
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=False,  # Already pinned in dataset
                        persistent_workers=True,
                        prefetch_factor=2
                    )
                
                # Train with adaptive stopping
                trainer = AdaptiveTrainer(current_model, device)
                trainer.scaler_data = scaler
                print(f"  Training adaptively (max 30 epochs)...")
                final_loss = trainer.train_adaptive(train_loader, max_epochs=30, min_epochs=8)
                
                # Save checkpoint for next iteration
                model_checkpoint = current_model.state_dict().copy()
                last_train_date = test_date
                
                # CHANGED: Reset batch predictions after retraining
                batch_predictions = []
            
            # Make predictions for test date
            if current_model is not None and trainer is not None:
                test_games = self.games_df[
                    (self.games_df['game_day'] == test_date) & 
                    (self.games_df['is_home'] == 1)
                ]
                
                current_model.eval()
                with torch.no_grad():
                    for _, game in test_games.iterrows():
                        home_seq = self.get_sequence_vectorized(
                            game['teamId'], test_date, sequence_length
                        )
                        away_seq = self.get_sequence_vectorized(
                            game['opponentId'], test_date, sequence_length
                        )
                        
                        if home_seq is not None and away_seq is not None:
                            # Scale and convert to tensors
                            home_scaled = trainer.scaler_data.transform(
                                np.nan_to_num(home_seq, nan=0.0, posinf=0.0, neginf=0.0)
                            )
                            away_scaled = trainer.scaler_data.transform(
                                np.nan_to_num(away_seq, nan=0.0, posinf=0.0, neginf=0.0)
                            )
                            
                            home_tensor = torch.FloatTensor(home_scaled).unsqueeze(0).to(device)
                            away_tensor = torch.FloatTensor(away_scaled).unsqueeze(0).to(device)
                            
                            with autocast():
                                pred = current_model(home_tensor, away_tensor)
                            
                            prediction = {
                                'game_id': game['gameId'],
                                'date': test_date,
                                'spread_pred': pred['spread'].cpu().item(),
                                'total_pred': pred['total'].cpu().item(),
                                'win_prob': torch.sigmoid(pred['win_probability']).cpu().item(),
                                'spread_actual': game['points'] - game['opp_points'],
                                'total_actual': game['points'] + game['opp_points'],
                                'home_won': 1 if game['points'] > game['opp_points'] else 0
                            }
                            
                            all_predictions.append(prediction)
                            batch_predictions.append(prediction)
                
        return pd.DataFrame(all_predictions)


if __name__ == "__main__":
    # Optimize PyTorch settings
    torch.set_num_threads(8)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        # Enable TensorFloat-32 for Ampere GPUs (RTX 30xx, 40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    pipeline = OptimizedPipeline(
        features_path='rolling_clusters_output/rolling_features.parquet',
        games_path='cbb_team_features.parquet'
    )
    
    results = pipeline.run_fast_backtest(
        sequence_length=10,
        retrain_days=30,
        batch_size=512,      # Larger batch size
        num_workers=8        # More workers
    )
    
    # Calculate final metrics
    if len(results) > 0:
        results['spread_error'] = results['spread_pred'] - results['spread_actual']
        results['total_error'] = results['total_pred'] - results['total_actual']
        
        spread_mae = results['spread_error'].abs().mean()
        total_mae = results['total_error'].abs().mean()
        
        print(f"\n{'='*50}")
        print(f"Final Results:")
        print(f"  Spread MAE: {spread_mae:.2f} points")
        print(f"  Total MAE: {total_mae:.2f} points")
        print(f"  Total predictions: {len(results)}")
        
        # Additional statistics
        print(f"\nSpread Prediction Stats:")
        print(f"  RMSE: {np.sqrt((results['spread_error']**2).mean()):.2f}")
        print(f"  Std Dev: {results['spread_error'].std():.2f}")
        print(f"  95th percentile error: {results['spread_error'].abs().quantile(0.95):.2f}")
        
        # Save results
        results.to_csv('optimized_dynamic_results.csv', index=False)
        print(f"\nResults saved to optimized_dynamic_results.csv")
    else:
        print("No predictions generated!")