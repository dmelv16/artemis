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
from typing import Tuple, List, Dict, Optional
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import functools
from collections import deque
import os
import time
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')

# ============= GPU OPTIMIZATION: Enable all optimizations =============
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')  # Allow TF32 on Ampere GPUs

from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

class SubsetRandomSampler(torch.utils.data.Sampler):
    """Sampler that randomly samples from a subset of indices"""
    def __init__(self, indices):
        self.indices = list(indices)  # Convert to list for indexing
    
    def __iter__(self):
        # Generate random permutation of positions
        perm = torch.randperm(len(self.indices)).tolist()
        # Return the actual indices at those positions
        return iter([self.indices[i] for i in perm])
    
    def __len__(self):
        return len(self.indices)
    
    def update_indices(self, new_indices):
        """Update the indices to sample from"""
        self.indices = list(new_indices)

class DynamicBacktestDataset(Dataset):
    """
    Dataset that holds ALL matchups in memory but only exposes a
    dynamic subset via an 'indices' list. This allows the DataLoader
    and its persistent workers to be reused.
    """
    def __init__(self, all_matchups_data, scaler, pin_memory=True):
        self.scaler = scaler
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
        # Store all data. We will slice this with self.indices
        self.all_home_seqs = np.array([item['home_sequence'] for item in all_matchups_data], dtype=np.float32)
        self.all_away_seqs = np.array([item['away_sequence'] for item in all_matchups_data], dtype=np.float32)
        self.all_spreads = np.array([item['spread_actual'] for item in all_matchups_data], dtype=np.float32)
        self.all_totals = np.array([item['total_actual'] for item in all_matchups_data], dtype=np.float32)
        self.all_winners = np.array([item['home_won'] for item in all_matchups_data], dtype=np.float32)
        
        # Store game IDs for predictions
        self.all_game_ids = [item.get('game_id') for item in all_matchups_data]

        # Convert to tensors (pinned if possible)
        if self.pin_memory:
            self.all_home_seqs = torch.from_numpy(self.all_home_seqs).pin_memory()
            self.all_away_seqs = torch.from_numpy(self.all_away_seqs).pin_memory()
            self.all_spreads = torch.from_numpy(self.all_spreads).pin_memory()
            self.all_totals = torch.from_numpy(self.all_totals).pin_memory()
            self.all_winners = torch.from_numpy(self.all_winners).pin_memory()
        else:
            self.all_home_seqs = torch.from_numpy(self.all_home_seqs)
            self.all_away_seqs = torch.from_numpy(self.all_away_seqs)
            self.all_spreads = torch.from_numpy(self.all_spreads)
            self.all_totals = torch.from_numpy(self.all_totals)
            self.all_winners = torch.from_numpy(self.all_winners)
        
        # Pre-scale all data ONCE to avoid repeated scaling
        if self.scaler is not None:
            print("  Pre-scaling all sequences (one-time cost)...")
            # Get numpy versions
            home_np = self.all_home_seqs.cpu().numpy() if isinstance(self.all_home_seqs, torch.Tensor) else self.all_home_seqs
            away_np = self.all_away_seqs.cpu().numpy() if isinstance(self.all_away_seqs, torch.Tensor) else self.all_away_seqs
            
            # Reshape for scaling
            n_samples, seq_len, n_features = home_np.shape
            home_flat = home_np.reshape(-1, n_features)
            away_flat = away_np.reshape(-1, n_features)
            f32_max = np.finfo(np.float32).max
            f32_min = np.finfo(np.float32).min
            # Clean and scale
            home_flat = np.nan_to_num(home_flat, nan=0.0, posinf=f32_max, neginf=f32_min)
            away_flat = np.nan_to_num(away_flat, nan=0.0, posinf=f32_max, neginf=f32_min)
            
            home_scaled = self.scaler.transform(home_flat).reshape(n_samples, seq_len, n_features)
            away_scaled = self.scaler.transform(away_flat).reshape(n_samples, seq_len, n_features)
            
            def clean_array(arr):
                """Aggressively clean arrays to prevent NaN/Inf"""
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                # Clip extreme values
                arr = np.clip(arr, -100, 100)
                return arr.astype(np.float32)

            # In DynamicBacktestDataset.__init__, replace the cleaning section with:
            home_scaled = clean_array(home_scaled)
            away_scaled = clean_array(away_scaled)
            
            # Convert back to tensors
            if self.pin_memory:
                self.all_home_seqs_scaled = torch.from_numpy(home_scaled).pin_memory()
                self.all_away_seqs_scaled = torch.from_numpy(away_scaled).pin_memory()
            else:
                self.all_home_seqs_scaled = torch.from_numpy(home_scaled)
                self.all_away_seqs_scaled = torch.from_numpy(away_scaled)
        else:
            self.all_home_seqs_scaled = self.all_home_seqs
            self.all_away_seqs_scaled = self.all_away_seqs
        
    def __len__(self):
        return len(self.all_home_seqs_scaled)
    
    def __getitem__(self, idx):
        """Get item - idx is now the actual index into all_matchups"""
        # idx is now a direct index into the full dataset
        if idx >= len(self.all_home_seqs_scaled):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.all_home_seqs_scaled)}")
        
        # Get pre-scaled sequences directly
        home_seq = self.all_home_seqs_scaled[idx].clone()
        away_seq = self.all_away_seqs_scaled[idx].clone()
        
        # Add noise for regularization during training
        if self.scaler is not None:  # Only add noise during training
            noise_level = 0.01
            home_seq = home_seq + (torch.randn_like(home_seq) * noise_level)
            away_seq = away_seq + (torch.randn_like(away_seq) * noise_level)

        return {
            'home_seq': home_seq,
            'away_seq': away_seq,
            'spread': self.all_spreads[idx],
            'total': self.all_totals[idx],
            'winner': self.all_winners[idx]
        }


class FastGameSequenceDataset(Dataset):
    """Original dataset for compatibility - kept for single prediction use"""
    
    def __init__(self, matchup_data, scaler=None, pin_memory=True, device=None):
        self.scaler = scaler
        self.num_samples = len(matchup_data)
        
        # Store unscaled data as numpy arrays
        self.home_seqs = np.array([item['home_sequence'] for item in matchup_data], dtype=np.float32)
        self.away_seqs = np.array([item['away_sequence'] for item in matchup_data], dtype=np.float32)
        self.spreads = np.array([item['spread_actual'] for item in matchup_data], dtype=np.float32)
        self.totals = np.array([item['total_actual'] for item in matchup_data], dtype=np.float32)
        self.winners = np.array([item['home_won'] for item in matchup_data], dtype=np.float32)
        
        # Store game dates for potential filtering (not transferred to GPU)
        self.game_dates = [item.get('game_date') for item in matchup_data] if 'game_date' in matchup_data[0] else None
        
        # Platform-specific optimizations
        self.on_gpu = False
        if os.name == 'nt':  # Windows
            # Convert to contiguous tensors for faster transfer
            self.home_seqs = torch.from_numpy(self.home_seqs).contiguous()
            self.away_seqs = torch.from_numpy(self.away_seqs).contiguous()
            self.spreads = torch.from_numpy(self.spreads).contiguous()
            self.totals = torch.from_numpy(self.totals).contiguous()
            self.winners = torch.from_numpy(self.winners).contiguous()
                
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
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get item with on-the-fly scaling if scaler provided"""
        home_seq = self.home_seqs[idx]
        away_seq = self.away_seqs[idx]
        
        # Apply scaling on-the-fly in worker process (parallel)
        if self.scaler is not None and not self.on_gpu:
            
            # Get data as numpy array
            home_seq_np = home_seq.cpu().numpy() if isinstance(home_seq, torch.Tensor) else home_seq
            away_seq_np = away_seq.cpu().numpy() if isinstance(away_seq, torch.Tensor) else away_seq
            f32_max = np.finfo(np.float32).max
            f32_min = np.finfo(np.float32).min
            home_seq_scaled = self.scaler.transform(
                np.nan_to_num(home_seq_np, nan=0.0, posinf=f32_max, neginf=f32_min)
            )
            away_seq_scaled = self.scaler.transform(
                np.nan_to_num(away_seq_np, nan=0.0, posinf=f32_max, neginf=f32_min)
            )
            
            home_seq = torch.from_numpy(home_seq_scaled.astype(np.float32))
            away_seq = torch.from_numpy(away_seq_scaled.astype(np.float32))

        if self.scaler is not None: 
            noise_level = 0.01
            home_seq = home_seq + (torch.randn_like(home_seq) * noise_level)
            away_seq = away_seq + (torch.randn_like(away_seq) * noise_level)

        return {
            'home_seq': home_seq,
            'away_seq': away_seq,
            'spread': self.spreads[idx],
            'total': self.totals[idx],
            'winner': self.winners[idx]
        }


class TransformerModel(nn.Module):
    """State-of-the-art Transformer architecture with CLS token"""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, n_heads=8, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Learnable CLS token for sequence aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Positional encoding (now for seq_len + 1 to account for CLS)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 11, hidden_dim))  # 10 + 1 for CLS
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Feature extraction layers
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Matchup combination
        self.matchup = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Output heads
        self.output = nn.Linear(hidden_dim // 2, 3)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Glorot initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize CLS token and positional encoding
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_encoder, std=0.02)
    
    def process_sequence(self, x):
        """Process sequence through transformer with CLS token"""
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Prepend CLS token to sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, seq_len + 1, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Use CLS token output (first position) as sequence representation
        context = x[:, 0, :]  # (batch, hidden_dim)
        
        return self.encoder(context)
    
    def forward(self, home_seq, away_seq):
        home_state = self.process_sequence(home_seq)
        away_state = self.process_sequence(away_seq)
        
        combined = torch.cat([home_state, away_state], dim=1)
        features = self.matchup(combined)
        outputs = self.output(features)
        
        return {
            'spread': outputs[:, 0:1],
            'total': outputs[:, 1:2],
            'win_probability': outputs[:, 2:3]
        }


class AdaptiveTrainer:
    """Advanced trainer with validation set and gradient accumulation"""
    
    def __init__(self, model, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.scaler_amp = GradScaler()
        self.scaler_data = None
        
        # Training config
        self.patience = 5
        self.min_delta = 0.001
        self.gradient_accumulation_steps = 1  # Can be increased for larger effective batch size

    def check_model_health(self):
        """Check if model parameters contain NaN/Inf"""
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                return False
        return True
    
    def validate_epoch(self, val_loader):
        """Run validation epoch and return average loss"""
        self.model.eval()
        epoch_loss = 0
        epoch_spread_loss = 0
        epoch_total_loss = 0
        epoch_win_loss = 0
        
        # Loss functions
        criterion_reg = nn.HuberLoss(delta=1.0)
        criterion_bce = nn.BCEWithLogitsLoss()
        
        # Fixed weights for validation (no dynamic adjustment)
        spread_weight = 2.0
        total_weight = 0.7
        win_weight = 0.3
        
        with torch.no_grad():
            for batch in val_loader:
                # Transfer to GPU
                home_seq = batch['home_seq'].to(self.device, non_blocking=True)
                away_seq = batch['away_seq'].to(self.device, non_blocking=True)
                spread = batch['spread'].unsqueeze(1).to(self.device, non_blocking=True)
                total = batch['total'].unsqueeze(1).to(self.device, non_blocking=True)
                winner = batch['winner'].unsqueeze(1).to(self.device, non_blocking=True)
                
                with autocast():
                    preds = self.model(home_seq, away_seq)
                    
                    spread_loss = criterion_reg(preds['spread'], spread)
                    total_loss = criterion_reg(preds['total'], total)
                    win_loss = criterion_bce(preds['win_probability'], winner)
                    
                    loss = (spread_weight * spread_loss + 
                           total_weight * total_loss + 
                           win_weight * win_loss)
                
                epoch_loss += loss.item()
                epoch_spread_loss += spread_loss.item()
                epoch_total_loss += total_loss.item()
                epoch_win_loss += win_loss.item()
        
        avg_loss = epoch_loss / len(val_loader)
        avg_spread_loss = epoch_spread_loss / len(val_loader)
        avg_total_loss = epoch_total_loss / len(val_loader)
        avg_win_loss = epoch_win_loss / len(val_loader)
        
        return avg_loss, avg_spread_loss, avg_total_loss, avg_win_loss
    
    def train_adaptive(self, train_loader, val_loader=None, max_epochs=30, min_epochs=8, 
                    writer=None, global_step_start=0):
        """Train with validation-based early stopping and advanced features"""
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.0005,  # REDUCED from 0.001 for stability
            weight_decay=0.01,
            betas=(0.9, 0.999),
            fused=torch.cuda.is_available()
        )
        
        # OneCycleLR scheduler for super-convergence
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.002,  # REDUCED from 0.005 to prevent gradient explosion
            epochs=max_epochs,
            steps_per_epoch=len(train_loader) // self.gradient_accumulation_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1000
        )
        
        # Loss functions - using Huber for robustness
        criterion_reg = nn.HuberLoss(delta=1.0)
        criterion_bce = nn.BCEWithLogitsLoss()
        
        # Dynamic loss weighting
        spread_weight_start = 3.0
        spread_weight_end = 2.0
        total_weight = 0.7
        win_weight = 0.3
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        val_loss_history = deque(maxlen=5)
        global_step = global_step_start
        
        # Track best model weights
        best_model_weights = None
        
        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0
            epoch_spread_loss = 0
            epoch_total_loss = 0
            epoch_win_loss = 0
            nan_batch_count = 0  # Track NaN batches per epoch
            valid_batches = 0  # Track successful batches
            
            # Dynamic weight adjustment
            progress = epoch / max_epochs
            spread_weight = spread_weight_start - (spread_weight_start - spread_weight_end) * progress
            
            for batch_idx, batch in enumerate(train_loader):
                # Transfer to GPU
                home_seq = batch['home_seq'].to(self.device, non_blocking=True)
                away_seq = batch['away_seq'].to(self.device, non_blocking=True)
                spread = batch['spread'].unsqueeze(1).to(self.device, non_blocking=True)
                total = batch['total'].unsqueeze(1).to(self.device, non_blocking=True)
                winner = batch['winner'].unsqueeze(1).to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                with autocast():
                    preds = self.model(home_seq, away_seq)
                    
                    spread_loss = criterion_reg(preds['spread'], spread)
                    total_loss = criterion_reg(preds['total'], total)
                    win_loss = criterion_bce(preds['win_probability'], winner)
                    
                    loss = (spread_weight * spread_loss + 
                        total_weight * total_loss + 
                        win_weight * win_loss)
                    
                    # NaN detection and recovery
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_batch_count += 1
                        optimizer.zero_grad(set_to_none=True)
                        
                        # If too many NaN batches, abort this epoch
                        if nan_batch_count > len(train_loader) * 0.15:  # More than 15% NaN
                            print(f"    CRITICAL: Too many NaN batches ({nan_batch_count}), stopping epoch early")
                            break
                        
                        # Don't print warning for every batch to avoid spam
                        if nan_batch_count <= 5 or nan_batch_count % 10 == 0:
                            print(f"    WARNING: NaN/Inf loss at batch {batch_idx} (total: {nan_batch_count})")
                        continue
                    
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                self.scaler_amp.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler_amp.unscale_(optimizer)
                    grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    
                    # Optimizer step (normal path)
                    self.scaler_amp.step(optimizer)
                    self.scaler_amp.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                
                # Logging
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                epoch_spread_loss += spread_loss.item()
                epoch_total_loss += total_loss.item()
                epoch_win_loss += win_loss.item()
                valid_batches += 1
                
                # TensorBoard logging
                if writer and global_step % 50 == 0:
                    writer.add_scalar('Loss/train_batch', loss.item() * self.gradient_accumulation_steps, global_step)
                    writer.add_scalar('Loss_Components/spread', spread_loss.item(), global_step)
                    writer.add_scalar('Loss_Components/total', total_loss.item(), global_step)
                    writer.add_scalar('Loss_Components/win', win_loss.item(), global_step)
                    writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)
                    writer.add_scalar('Training/nan_batches', nan_batch_count, global_step)
                
                global_step += 1
            
            # Handle final accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
                self.scaler_amp.unscale_(optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.scaler_amp.step(optimizer)
                self.scaler_amp.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            # Check model health after epoch
            model_has_nan = False
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"    CRITICAL: Model parameter '{name}' has NaN/Inf after epoch {epoch+1}")
                    model_has_nan = True
                    break
            
            if model_has_nan:
                if best_model_weights is not None:
                    print(f"    Restoring best weights and stopping training")
                    self.model.load_state_dict(best_model_weights)
                break
            
            # If too many NaN batches in this epoch, consider it failed
            if nan_batch_count > len(train_loader) * 0.15:
                print(f"    Epoch {epoch+1} had {nan_batch_count} NaN batches ({nan_batch_count/len(train_loader)*100:.1f}%)")
                if best_model_weights is not None:
                    print(f"    Restoring best weights")
                    self.model.load_state_dict(best_model_weights)
                break
            
            # Calculate training metrics (only from valid batches)
            if valid_batches > 0:
                avg_train_loss = epoch_loss / valid_batches
                avg_train_spread = epoch_spread_loss / valid_batches
                avg_train_total = epoch_total_loss / valid_batches
                avg_train_win = epoch_win_loss / valid_batches
            else:
                print(f"    No valid batches in epoch {epoch+1}, stopping training")
                break
            
            # Validation phase
            if val_loader is not None:
                avg_val_loss, avg_val_spread, avg_val_total, avg_val_win = self.validate_epoch(val_loader)
                val_loss_history.append(avg_val_loss)
                
                # Track best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_weights = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                # If no validation set, use training loss (less ideal)
                avg_val_loss = avg_train_loss
                val_loss_history.append(avg_val_loss)
                
                if avg_val_loss < best_val_loss - self.min_delta:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            # TensorBoard epoch logging
            if writer:
                writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
                writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
                writer.add_scalar('Loss_Epoch/train_spread', avg_train_spread, epoch)
                writer.add_scalar('Loss_Epoch/train_total', avg_train_total, epoch)
                writer.add_scalar('Loss_Epoch/train_win', avg_train_win, epoch)
                writer.add_scalar('Training/nan_epochs', nan_batch_count, epoch)
                if val_loader:
                    writer.add_scalar('Loss_Epoch/val_spread', avg_val_spread, epoch)
                    writer.add_scalar('Loss_Epoch/val_total', avg_val_total, epoch)
                    writer.add_scalar('Loss_Epoch/val_win', avg_val_win, epoch)
            
            # Early stopping based on validation loss
            if epoch >= min_epochs:
                # Check if validation loss is plateauing
                if len(val_loss_history) >= 5:
                    recent_losses = list(val_loss_history)[-3:]
                    recent_improvement = max(recent_losses) - min(recent_losses)
                    if recent_improvement < self.min_delta:
                        print(f"    Early stopping at epoch {epoch+1}, validation loss plateaued")
                        break
                
                # Check patience
                if patience_counter >= self.patience:
                    print(f"    Early stopping at epoch {epoch+1}, patience exhausted")
                    break
            
            # Progress reporting
            if epoch % 3 == 0 or nan_batch_count > 0:
                if val_loader:
                    status = f"    Epoch {epoch+1}/{max_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
                    if nan_batch_count > 0:
                        status += f", NaN batches: {nan_batch_count}"
                    print(status)
                else:
                    status = f"    Epoch {epoch+1}/{max_epochs}, Loss: {avg_train_loss:.4f} (S: {avg_train_spread:.4f}, T: {avg_train_total:.4f}, W: {avg_train_win:.4f}), LR: {scheduler.get_last_lr()[0]:.6f}"
                    if nan_batch_count > 0:
                        status += f", NaN batches: {nan_batch_count}"
                    print(status)
        
        # Restore best model weights if we have them
        if best_model_weights is not None and val_loader is not None:
            self.model.load_state_dict(best_model_weights)
            print(f"    Restored best model weights (val_loss: {best_val_loss:.4f})")
        
        return best_val_loss, global_step


class OptimizedPipeline:
    """Advanced pipeline with pre-generated sequences and persistent DataLoaders"""
    
    def __init__(self, features_path, games_path):
        self.features_path = features_path
        self.games_path = games_path
        
        # Pre-load data
        print("Loading data into memory...")
        self.features_df = pd.read_parquet(features_path)
        self.games_df = pd.read_parquet(games_path)
        
        # Process dates
        self.features_df['startDate'] = pd.to_datetime(self.features_df['startDate'])
        self.games_df['startDate'] = pd.to_datetime(self.games_df['startDate'])
        self.features_df = self.features_df.rename(columns={'startDate': 'game_date'})
        self.games_df['game_day'] = self.games_df['startDate'].dt.normalize()
        
        # Feature columns
        self.feature_cols = [col for col in self.features_df.columns 
                            if ('_PC' in col or '_cluster' in col)]
        self.input_dim = len(self.feature_cols)
        
        # Build team lookup
        print("Building optimized team lookup tables...")
        self.team_lookup = {}
        for team_id in self.features_df['teamId'].unique():
            team_data = self.features_df[self.features_df['teamId'] == team_id].sort_values('game_date')
            self.team_lookup[team_id] = {
                'dates': team_data['game_date'].values,
                'features': team_data[self.feature_cols].values.astype(np.float32)
            }
        
        # Unique games
        self.unique_games = self.games_df[self.games_df['is_home'] == 1].copy()
        
        # Pre-generate all sequences
        print("Pre-generating all matchup sequences (one-time cost)...")
        self.all_matchups = self._generate_all_matchups(seq_length=10)
        
        print(f"Pipeline initialized:")
        print(f"  Features: {len(self.features_df):,}")
        print(f"  Games: {len(self.games_df):,}")
        print(f"  Unique games: {len(self.unique_games):,}")
        print(f"  Pre-generated matchups: {len(self.all_matchups):,}")
        print(f"  Input dimension: {self.input_dim}")
    
    def _generate_all_matchups(self, seq_length=10):
        """Pre-generate all possible matchup sequences"""
        matchup_data = []
        games_to_process = self.unique_games.sort_values('startDate')
        
        print(f"  Processing {len(games_to_process):,} games...")
        for idx, (_, game) in enumerate(games_to_process.iterrows()):
            if idx % 5000 == 0:
                print(f"    Generated {idx:,} / {len(games_to_process):,} matchups...")
            
            home_seq = self.get_sequence_vectorized(game['teamId'], game['startDate'], seq_length)
            away_seq = self.get_sequence_vectorized(game['opponentId'], game['startDate'], seq_length)
            
            if home_seq is not None and away_seq is not None:
                matchup_data.append({
                    'game_date': game['startDate'],
                    'game_id': game['gameId'],
                    'home_sequence': home_seq,
                    'away_sequence': away_seq,
                    'spread_actual': game['points'] - game['opp_points'],
                    'total_actual': game['points'] + game['opp_points'],
                    'home_won': 1 if game['points'] > game['opp_points'] else 0
                })
        
        return matchup_data
    
    def get_sequence_vectorized(self, team_id, game_date, seq_length=10):
        """Fast vectorized sequence retrieval"""
        if team_id not in self.team_lookup:
            return None
        
        team_data = self.team_lookup[team_id]
        mask = team_data['dates'] < game_date
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) < seq_length:
            return None
        
        indices = valid_indices[-seq_length:]
        return team_data['features'][indices]
    
    def run_fast_backtest(self, sequence_length=10, retrain_days=30, 
                        batch_size=512, use_transformer=True,
                        checkpoint_path='backtest_checkpoint.pth'):
        """Optimized backtesting with truly persistent DataLoaders"""
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Platform-specific optimizations
        num_workers = 0  # Default
        if os.name == 'nt':  # Windows
            num_workers = min(6, os.cpu_count() // 2)
            batch_size = min(1024, batch_size * 2)
            print(f"  Windows detected: Workers={num_workers}, Batch size={batch_size}")
        else:  # Linux/Mac
            num_workers = min(os.cpu_count() - 1, 8)
            print(f"  Unix detected: Workers={num_workers}, Batch size={batch_size}")
        
        # TensorBoard setup
        run_name = f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        writer = SummaryWriter(f'runs/{run_name}')
        global_step = 0
        
        # --- FIT SCALER ONCE ---
        print("\nFitting global scaler...")
        scaler = StandardScaler()

        # Initialize tracking variables
        all_predictions = []
        last_train_date = None
        model_checkpoint = None

        # Check if loading from checkpoint
        if os.path.exists(checkpoint_path):
            print(f"  Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if 'scaler' in checkpoint:
                print("  Loading scaler from checkpoint.")
                scaler = checkpoint['scaler']
            model_checkpoint = checkpoint.get('model_state_dict')
            last_train_date = checkpoint.get('last_train_date')
            all_predictions = checkpoint.get('all_predictions', [])
            global_step = checkpoint.get('global_step', 0)
            print(f"  Resuming from {pd.Timestamp(last_train_date).strftime('%Y-%m-%d') if last_train_date else 'beginning'}")
            
            # === DATA QUALITY CHECK AROUND FAILURE POINT ===
            if last_train_date is not None:
                print("\n=== Checking data quality around last training date ===")
                check_start = last_train_date - np.timedelta64(30, 'D')
                check_end = last_train_date + np.timedelta64(30, 'D')
                problem_indices = [i for i, m in enumerate(self.all_matchups) 
                                if check_start <= m['game_date'] <= check_end]
                
                print(f"Checking {len(problem_indices)} matchups near {pd.Timestamp(last_train_date).strftime('%Y-%m-%d')}")
                
                # Check a REPRESENTATIVE SAMPLE across the entire range
                if len(problem_indices) > 100:
                    # Sample evenly across the range instead of just the first 20
                    check_indices = np.linspace(0, len(problem_indices)-1, 100, dtype=int)
                    sample_indices = [problem_indices[i] for i in check_indices]
                    print(f"  Sampling 100 matchups evenly distributed across the date range")
                else:
                    sample_indices = problem_indices
                    print(f"  Checking all {len(sample_indices)} matchups in range")
                
                nan_count = 0
                inf_count = 0
                extreme_count = 0
                nan_indices = []
                inf_indices = []
                extreme_indices = []
                
                for idx in sample_indices:
                    matchup = self.all_matchups[idx]
                    home = matchup['home_sequence']
                    away = matchup['away_sequence']
                    
                    has_nan = np.isnan(home).any() or np.isnan(away).any()
                    has_inf = np.isinf(home).any() or np.isinf(away).any()
                    has_extreme = (np.abs(home) > 100).any() or (np.abs(away) > 100).any()
                    
                    if has_nan:
                        nan_count += 1
                        nan_indices.append(idx)
                    if has_inf:
                        inf_count += 1
                        inf_indices.append(idx)
                    if has_extreme:
                        extreme_count += 1
                        extreme_indices.append(idx)
                
                print(f"  Raw data issues in sample of {len(sample_indices)}:")
                print(f"    NaN values: {nan_count} ({nan_count/len(sample_indices)*100:.1f}%)")
                print(f"    Inf values: {inf_count} ({inf_count/len(sample_indices)*100:.1f}%)")
                print(f"    Extreme values (>100): {extreme_count} ({extreme_count/len(sample_indices)*100:.1f}%)")
                
                # Show examples of problematic data
                if nan_count > 0:
                    print(f"\n  Example NaN matchup (index {nan_indices[0]}):")
                    example = self.all_matchups[nan_indices[0]]
                    print(f"    Date: {pd.Timestamp(example['game_date']).strftime('%Y-%m-%d')}")
                    print(f"    Home NaN count: {np.isnan(example['home_sequence']).sum()}")
                    print(f"    Away NaN count: {np.isnan(example['away_sequence']).sum()}")
                
                if inf_count > 0:
                    print(f"\n  Example Inf matchup (index {inf_indices[0]}):")
                    example = self.all_matchups[inf_indices[0]]
                    print(f"    Date: {pd.Timestamp(example['game_date']).strftime('%Y-%m-%d')}")
                    print(f"    Home Inf count: {np.isinf(example['home_sequence']).sum()}")
                    print(f"    Away Inf count: {np.isinf(example['away_sequence']).sum()}")
                
                if extreme_count > 0:
                    print(f"\n  Example extreme value matchup (index {extreme_indices[0]}):")
                    example = self.all_matchups[extreme_indices[0]]
                    home_max = np.abs(example['home_sequence']).max()
                    away_max = np.abs(example['away_sequence']).max()
                    print(f"    Date: {pd.Timestamp(example['game_date']).strftime('%Y-%m-%d')}")
                    print(f"    Home max absolute value: {home_max:.2f}")
                    print(f"    Away max absolute value: {away_max:.2f}")
                
                # CRITICAL: If significant data quality issues, recommend action
                total_issues = nan_count + inf_count + extreme_count
                if total_issues > len(sample_indices) * 0.1:  # More than 10% problematic
                    print(f"\n  ⚠️  WARNING: {total_issues/len(sample_indices)*100:.1f}% of samples have issues!")
                    print(f"  RECOMMENDATION: Clean raw data in self.all_matchups before scaling")
        else:
            # Fit new scaler on ALL data
            print(f"  No checkpoint found, fitting new scaler on ALL {len(self.all_matchups):,} matchups...")
            
            # Extract all sequences (no sampling)
            home_seqs_list = [m['home_sequence'] for m in self.all_matchups]
            away_seqs_list = [m['away_sequence'] for m in self.all_matchups]
            
            print(f"  Converting to arrays...")
            home_seqs_3d = np.array(home_seqs_list, dtype=np.float32)
            away_seqs_3d = np.array(away_seqs_list, dtype=np.float32)
            all_seqs = np.vstack([home_seqs_3d, away_seqs_3d])
            
            print(f"  Reshaping {all_seqs.shape[0]:,} sequences for scaling...")
            n_samples, seq_len, n_features = all_seqs.shape
            all_seqs_flat = all_seqs.reshape(-1, n_features)
            
            print(f"  Cleaning data (removing NaN/Inf)...")
            all_seqs_flat = np.nan_to_num(all_seqs_flat, nan=0.0, posinf=0.0, neginf=0.0)
            
            print(f"  Fitting scaler on {all_seqs_flat.shape[0]:,} feature vectors...")
            scaler.fit(all_seqs_flat)
            print(f"  ✓ Global scaler fit complete on ALL data!")
            print(f"    Total sequences: {n_samples:,}")
            print(f"    Features per timestep: {n_features}")
            print(f"    Total feature vectors: {all_seqs_flat.shape[0]:,}")
        
        # --- CREATE DATASETS ONCE (OUTSIDE LOOP) ---
        print("\nInitializing Dynamic Datasets...")
        train_dataset = DynamicBacktestDataset(
            self.all_matchups, 
            scaler=scaler,
            pin_memory=(not os.name == 'nt')
        )
        
        val_dataset = DynamicBacktestDataset(
            self.all_matchups,
            scaler=scaler,
            pin_memory=(not os.name == 'nt')
        )
        print("  Datasets initialized!")
        
        # === PRE-FLIGHT DATA QUALITY CHECK ===
        print("\n=== Pre-flight Data Quality Check ===")
        nan_count = 0
        inf_count = 0
        extreme_count = 0
        for idx in range(min(1000, len(train_dataset.all_home_seqs_scaled))):
            home = train_dataset.all_home_seqs_scaled[idx]
            away = train_dataset.all_away_seqs_scaled[idx]
            
            if torch.isnan(home).any() or torch.isnan(away).any():
                nan_count += 1
            if torch.isinf(home).any() or torch.isinf(away).any():
                inf_count += 1
            if (home.abs() > 50).any() or (away.abs() > 50).any():
                extreme_count += 1
        
        print(f"  Checked 1000 scaled samples:")
        print(f"    NaN values: {nan_count}")
        print(f"    Inf values: {inf_count}")
        print(f"    Extreme values (>50): {extreme_count}")
        
        if nan_count > 0 or inf_count > 0:
            print("  WARNING: Data quality issues detected! Training may be unstable.")
        
        # --- CREATE PERSISTENT DATALOADERS ONCE ---
        print("\nCreating truly persistent DataLoaders (one-time initialization)...")
        
        # Create custom samplers with empty indices initially
        train_sampler = SubsetRandomSampler([])
        val_sampler = SubsetRandomSampler([])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=False
        )
        print("  Persistent DataLoaders created! Workers will stay alive for entire backtest.")
        
        # Setup test dates
        test_dates = np.sort(self.games_df['game_day'].unique())
        min_date = test_dates[365]
        test_dates = test_dates[test_dates >= min_date]
        
        # Fast-forward dates if resuming from checkpoint
        if last_train_date is not None:
            test_dates = test_dates[test_dates > last_train_date]
        
        # Initialize remaining tracking variables
        current_model = None
        trainer = None
        batch_predictions = []
        consecutive_nan_failures = 0  # Track consecutive NaN failures
        
        # Main backtesting loop
        print(f"\nStarting backtest over {len(test_dates)} dates...")
        for i, test_date in enumerate(test_dates):
            # Check if retraining needed
            if last_train_date is None:
                days_since_train = float('inf')
            else:
                days_diff = test_date - last_train_date
                days_since_train = int(days_diff / np.timedelta64(1, 'D'))
            
            should_retrain = current_model is None or days_since_train >= retrain_days
            
            if should_retrain:
                # Report previous batch stats
                if len(batch_predictions) > 0:
                    batch_df = pd.DataFrame(batch_predictions)
                    batch_spread_mae = (batch_df['spread_pred'] - batch_df['spread_actual']).abs().mean()
                    batch_total_mae = (batch_df['total_pred'] - batch_df['total_actual']).abs().mean()
                    
                    # Log to TensorBoard
                    writer.add_scalar('MAE/batch_spread', batch_spread_mae, global_step)
                    writer.add_scalar('MAE/batch_total', batch_total_mae, global_step)
                    
                    print(f"\n  Previous batch stats ({len(batch_predictions)} predictions):")
                    print(f"    Spread MAE: {batch_spread_mae:.3f}")
                    print(f"    Total MAE:  {batch_total_mae:.3f}")
                
                print(f"\n[{i+1}/{len(test_dates)}] Retraining for {pd.Timestamp(test_date).strftime('%Y-%m-%d')}")
                
                # Get INDICES for the pre-generated data
                train_indices_full = [
                    idx for idx, m in enumerate(self.all_matchups) 
                    if m['game_date'] < test_date
                ]
                
                if len(train_indices_full) < 1000:
                    print(f"  Skipping - only {len(train_indices_full)} training samples")
                    continue
                
                # Split indices into train and validation sets
                train_indices, val_indices = train_test_split(
                    train_indices_full, 
                    test_size=0.1,
                    shuffle=True,
                    random_state=42
                )
                print(f"  Training on {len(train_indices):,} games, validating on {len(val_indices):,} games...")
                
                # --- UPDATE INDICES (INSTANT!) ---
                print("  Updating indices in datasets and samplers...")
                
                # Update samplers
                train_sampler.update_indices(train_indices)
                val_sampler.update_indices(val_indices)
                print("  ✓ Indices updated instantly! Workers remain persistent.")
                
                # === MODEL HEALTH CHECK ===
                if current_model is not None and trainer is not None:
                    print("  Checking model health before training...")
                    test_input = torch.randn(1, 10, self.input_dim).to(device)
                    try:
                        with torch.no_grad():
                            test_output = current_model(test_input, test_input)
                            if (torch.isnan(test_output['spread']).any() or 
                                torch.isinf(test_output['spread']).any()):
                                print("  WARNING: Model producing NaN/Inf outputs, reinitializing...")
                                current_model = None
                                trainer = None
                                consecutive_nan_failures += 1
                            else:
                                print("  ✓ Model health check passed")
                                consecutive_nan_failures = 0  # Reset counter
                    except Exception as e:
                        print(f"  WARNING: Model failed health check ({e}), reinitializing...")
                        current_model = None
                        trainer = None
                        consecutive_nan_failures += 1
                
                # If too many consecutive failures, use more conservative settings
                if consecutive_nan_failures >= 3:
                    print("  WARNING: Multiple consecutive NaN failures detected!")
                    print("  Switching to ultra-conservative training settings...")
                    use_conservative_mode = True
                else:
                    use_conservative_mode = False
                
                # Create model (or reuse)
                if current_model is None:
                    if use_transformer:
                        current_model = TransformerModel(
                            input_dim=self.input_dim,
                            hidden_dim=256,
                            num_layers=3,
                            n_heads=8,
                            dropout=0.3
                        )
                        print("  Using Transformer architecture")
                    else:
                        from gpuSequence import OptimizedDynamicModel
                        current_model = OptimizedDynamicModel(
                            input_dim=self.input_dim,
                            hidden_dim=256,
                            num_layers=2,
                            dropout=0.3
                        )
                        print("  Using LSTM architecture")
                    
                    # Transfer learning (only if not in recovery mode)
                    if model_checkpoint is not None and consecutive_nan_failures == 0:
                        try:
                            current_model.load_state_dict(model_checkpoint, strict=False)
                            print("  Loaded previous weights (transfer learning)")
                        except:
                            print("  Could not load previous weights (architecture mismatch)")
                    elif consecutive_nan_failures > 0:
                        print("  Starting fresh (no transfer learning due to previous NaN issues)")
                
                # Create trainer (or reuse)
                if trainer is None:
                    trainer = AdaptiveTrainer(current_model, device)
                
                trainer.scaler_data = scaler
                
                # Adjust training parameters based on mode
                if use_conservative_mode:
                    max_epochs = 20
                    min_epochs = 5
                    print(f"  Training in CONSERVATIVE mode (max {max_epochs} epochs)...")
                else:
                    max_epochs = 30
                    min_epochs = 8
                    print(f"  Training (max {max_epochs} epochs)...")
                
                start_time = time.time()
                try:
                    final_loss, global_step = trainer.train_adaptive(
                        train_loader,
                        val_loader,
                        max_epochs=max_epochs,
                        min_epochs=min_epochs,
                        writer=writer,
                        global_step_start=global_step
                    )
                    train_time = time.time() - start_time
                    
                    # Check if training succeeded
                    if np.isfinite(final_loss):
                        print(f"  ✓ Training completed in {train_time:.1f}s (best val loss: {final_loss:.4f})")
                        consecutive_nan_failures = 0  # Reset counter on success
                    else:
                        print(f"  WARNING: Training resulted in infinite loss!")
                        consecutive_nan_failures += 1
                        
                except Exception as e:
                    print(f"  ERROR during training: {e}")
                    consecutive_nan_failures += 1
                    continue
                
                # Save checkpoint only if training was successful
                if np.isfinite(final_loss):
                    model_checkpoint = current_model.state_dict().copy()
                    last_train_date = test_date
                    
                    torch.save({
                        'model_state_dict': model_checkpoint,
                        'last_train_date': last_train_date,
                        'scaler': scaler,
                        'all_predictions': all_predictions,
                        'global_step': global_step
                    }, checkpoint_path)
                    print(f"  Checkpoint saved to {checkpoint_path}")
                else:
                    print(f"  Skipping checkpoint save due to training issues")
                
                # Reset batch predictions
                batch_predictions = []
            
            # --- BATCHED PREDICTION ---
            if current_model is not None and trainer is not None:
                test_games_data = [m for m in self.all_matchups 
                                if m['game_date'] == test_date]
                
                if len(test_games_data) > 0:
                    current_model.eval()
                    
                    test_dataset = FastGameSequenceDataset(
                        test_games_data, 
                        scaler=trainer.scaler_data,
                        pin_memory=(not os.name == 'nt')
                    )
                    
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=len(test_games_data),
                        shuffle=False,
                        num_workers=0
                    )
                    
                    with torch.no_grad():
                        for batch in test_loader:
                            home_tensor = batch['home_seq'].to(device, non_blocking=True)
                            away_tensor = batch['away_seq'].to(device, non_blocking=True)
                            
                            with autocast():
                                preds = current_model(home_tensor, away_tensor)
                            
                            spread_preds = preds['spread'].cpu().numpy().flatten()
                            total_preds = preds['total'].cpu().numpy().flatten()
                            win_probs = torch.sigmoid(preds['win_probability']).cpu().numpy().flatten()
                            
                            for i, game_meta in enumerate(test_games_data):
                                prediction = {
                                    'game_id': game_meta['game_id'],
                                    'date': test_date,
                                    'spread_pred': spread_preds[i],
                                    'total_pred': total_preds[i],
                                    'win_prob': win_probs[i],
                                    'spread_actual': game_meta['spread_actual'],
                                    'total_actual': game_meta['total_actual'],
                                    'home_won': game_meta['home_won']
                                }
                                all_predictions.append(prediction)
                                batch_predictions.append(prediction)
        
        # Final statistics
        writer.close()
        results_df = pd.DataFrame(all_predictions)
        
        if len(results_df) > 0:
            results_df['spread_error'] = results_df['spread_pred'] - results_df['spread_actual']
            results_df['total_error'] = results_df['total_pred'] - results_df['total_actual']
            
            spread_mae = results_df['spread_error'].abs().mean()
            total_mae = results_df['total_error'].abs().mean()
            spread_rmse = np.sqrt((results_df['spread_error']**2).mean())
            total_rmse = np.sqrt((results_df['total_error']**2).mean())
            
            print(f"\n{'='*60}")
            print(f"FINAL BACKTEST RESULTS")
            print(f"{'='*60}")
            print(f"Total predictions: {len(results_df):,}")
            print(f"\nSpread predictions:")
            print(f"  MAE:  {spread_mae:.2f} points")
            print(f"  RMSE: {spread_rmse:.2f} points")
            print(f"  Std:  {results_df['spread_error'].std():.2f} points")
            print(f"  95%:  {results_df['spread_error'].abs().quantile(0.95):.2f} points")
            
            print(f"\nTotal predictions:")
            print(f"  MAE:  {total_mae:.2f} points")
            print(f"  RMSE: {total_rmse:.2f} points")
            print(f"  Std:  {results_df['total_error'].std():.2f} points")
            print(f"  95%:  {results_df['total_error'].abs().quantile(0.95):.2f} points")
            
            # Save results
            output_file = f'results_{run_name}.csv'
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
            print(f"TensorBoard logs saved to runs/{run_name}")
            print(f"Run: tensorboard --logdir=runs to view training metrics")
        
        return results_df


if __name__ == "__main__":
    # Optimize PyTorch settings
    torch.set_num_threads(min(8, os.cpu_count()))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        # Enable TensorFloat-32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Initialize pipeline
    pipeline = OptimizedPipeline(
        features_path='rolling_clusters_output/rolling_features.parquet',
        games_path='cbb_team_features.parquet'
    )
    
    # Run backtest with persistent DataLoaders
    results = pipeline.run_fast_backtest(
        sequence_length=10,
        retrain_days=30,
        batch_size=512,
        use_transformer=True,  # Use advanced Transformer architecture
        checkpoint_path='backtest_checkpoint.pth'
    )
    
    print("\nBacktest completed successfully!")
    print("Workers remain persistent across retraining cycles!")