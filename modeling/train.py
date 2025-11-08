"""
Main training script with proper validation and early stopping.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

class Trainer:
    """
    Training manager with validation, early stopping, and checkpointing.
    """
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Loss functions
        self.spread_criterion = nn.MSELoss()
        self.total_criterion = nn.MSELoss()
        self.win_criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=5,
            factor=0.5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'spread_mae': [],
            'total_mae': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        spread_losses = []
        total_losses = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            features = batch['features'].to(self.device)
            spread_target = batch['spread_actual'].to(self.device).unsqueeze(1)
            total_target = batch['total_actual'].to(self.device).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            # Calculate losses
            spread_loss = self.spread_criterion(predictions['spread'], spread_target)
            total_loss_val = self.total_criterion(predictions['total'], total_target)
            
            # Combined loss with weights
            loss = (self.config['spread_weight'] * spread_loss + 
                   self.config['total_weight'] * total_loss_val)
            
            # Add win loss if available
            if 'win_prob' in predictions:
                win_target = batch['home_won'].to(self.device).unsqueeze(1).float()
                win_loss = self.win_criterion(predictions['win_prob'], win_target)
                loss += self.config.get('win_weight', 0.1) * win_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            spread_losses.append(spread_loss.item())
            total_losses.append(total_loss_val.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'spread_loss': spread_loss.item(),
                'total_loss': total_loss_val.item()
            })
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        spread_mae = 0
        total_mae = 0
        correct_wins = 0
        total_games = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                features = batch['features'].to(self.device)
                spread_target = batch['spread_actual'].to(self.device).unsqueeze(1)
                total_target = batch['total_actual'].to(self.device).unsqueeze(1)
                
                # Forward pass
                predictions = self.model(features)
                
                # Calculate losses
                spread_loss = self.spread_criterion(predictions['spread'], spread_target)
                total_loss_val = self.total_criterion(predictions['total'], total_target)
                
                loss = (self.config['spread_weight'] * spread_loss + 
                       self.config['total_weight'] * total_loss_val)
                
                # Calculate MAE for interpretability
                spread_mae += torch.abs(predictions['spread'] - spread_target).mean().item()
                total_mae += torch.abs(predictions['total'] - total_target).mean().item()
                
                # Track accuracy
                if 'win_prob' in predictions:
                    predicted_wins = (predictions['win_prob'] > 0.0).float()
                    actual_wins = batch['home_won'].to(self.device).unsqueeze(1).float()
                    correct_wins += (predicted_wins == actual_wins).sum().item()
                    total_games += len(actual_wins)
                
                total_loss += loss.item()
        
        # Calculate averages
        avg_loss = total_loss / len(self.val_loader)
        avg_spread_mae = spread_mae / len(self.val_loader)
        avg_total_mae = total_mae / len(self.val_loader)
        win_accuracy = correct_wins / total_games if total_games > 0 else 0
        
        return avg_loss, avg_spread_mae, avg_total_mae, win_accuracy
    
    def train(self, num_epochs):
        """Main training loop."""
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, spread_mae, total_mae, win_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['spread_mae'].append(spread_mae)
            self.history['total_mae'].append(total_mae)
            
            # Print epoch results
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Spread MAE: {spread_mae:.2f} points")
            print(f"  Total MAE: {total_mae:.2f} points")
            print(f"  Win Accuracy: {win_acc:.2%}")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss)
                print(f"  ✓ New best model saved!")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.config['patience']}")
                
                if self.patience_counter >= self.config['patience']:
                    print("\nEarly stopping triggered!")
                    break
        
        # Plot training history
        self.plot_history()
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'config': self.config
        }
        
        filename = f"checkpoint_epoch{epoch}_loss{val_loss:.4f}.pth"
        torch.save(checkpoint, filename)
        
        # Also save as best model
        torch.save(checkpoint, 'best_model.pth')
    
    def plot_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Spread MAE
        axes[0, 1].plot(self.history['spread_mae'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (points)')
        axes[0, 1].set_title('Spread Prediction MAE')
        axes[0, 1].grid(True)
        
        # Total MAE
        axes[1, 0].plot(self.history['total_mae'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE (points)')
        axes[1, 0].set_title('Total Prediction MAE')
        axes[1, 0].grid(True)
        
        # Combined MAE
        axes[1, 1].plot(self.history['spread_mae'], label='Spread MAE')
        axes[1, 1].plot(self.history['total_mae'], label='Total MAE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE (points)')
        axes[1, 1].set_title('Prediction Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()