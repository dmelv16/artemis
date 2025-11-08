"""
Make predictions for upcoming games.
"""
import torch
import pandas as pd
from model import BasketballPredictor
import pickle

def predict_games(model_path, upcoming_games_df):
    """
    Make predictions for upcoming games.
    
    Args:
        model_path: Path to saved model
        upcoming_games_df: DataFrame with same features as training
    
    Returns:
        DataFrame with predictions
    """
    # Load model
    checkpoint = torch.load(model_path)
    model = BasketballPredictor(
        input_dim=upcoming_games_df.shape[1]
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Scale features
    features_scaled = scaler.transform(upcoming_games_df.values)
    features_tensor = torch.FloatTensor(features_scaled).to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(features_tensor)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'spread_prediction': predictions['spread'].cpu().numpy().flatten(),
        'total_prediction': predictions['total'].cpu().numpy().flatten(),
        'home_win_probability': predictions['win_prob'].cpu().numpy().flatten()
    })
    
    return results