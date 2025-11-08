"""
Simplified version of the deep architecture for training.
Start with this, then add complexity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasketballPredictor(nn.Module):
    """
    Multi-task neural network for spread and total prediction.
    This is a simplified version to start with.
    """
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Task-specific heads
        self.spread_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, 1)
        )
        
        self.total_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, 1)
        )
        
        # Optional: Win probability head
        self.win_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Make predictions
        spread = self.spread_head(features)
        total = self.total_head(features)
        win_prob = self.win_head(features)
        
        return {
            'spread': spread,
            'total': total,
            'win_prob': win_prob
        }


class AdvancedBasketballPredictor(nn.Module):
    """
    More advanced architecture with attention and expert models.
    Use this once the basic model is working.
    """
    
    def __init__(self, input_dim, config=None):
        super().__init__()
        
        # Feature projection
        self.input_proj = nn.Linear(input_dim, 256)
        
        # Self-attention layers
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Expert networks for different aspects
        self.pace_expert = self._build_expert(256, 128)
        self.matchup_expert = self._build_expert(256, 128)
        self.form_expert = self._build_expert(256, 128)
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )
        
        # Final prediction layers
        self.spread_predictor = nn.Linear(128, 1)
        self.total_predictor = nn.Linear(128, 1)
        
    def _build_expert(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Project input
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # Add sequence dimension for attention
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        attn_out = attn_out.squeeze(1)
        
        # Expert predictions
        pace_out = self.pace_expert(attn_out)
        matchup_out = self.matchup_expert(attn_out)
        form_out = self.form_expert(attn_out)
        
        # Gate and combine experts
        expert_stack = torch.stack([pace_out, matchup_out, form_out], dim=1)
        gates = self.gate(attn_out).unsqueeze(2)
        combined = (expert_stack * gates).sum(dim=1)
        
        # Final predictions
        spread = self.spread_predictor(combined)
        total = self.total_predictor(combined)
        
        return {
            'spread': spread,
            'total': total,
            'expert_weights': gates.squeeze(2)
        }