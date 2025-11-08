"""
Evaluation script for analyzing model performance.
"""
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from model import BasketballPredictor

def evaluate_model(model_path, test_dataset):
    """
    Comprehensive model evaluation.
    """
    # Load model
    checkpoint = torch.load(model_path)
    model = BasketballPredictor(
        input_dim=test_dataset.features.shape[1]
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Make predictions
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            features = sample['features'].unsqueeze(0).to(device)
            
            pred = model(features)
            
            all_predictions.append({
                'spread_pred': pred['spread'].item(),
                'total_pred': pred['total'].item(),
                'win_prob': pred.get('win_prob', torch.tensor([0.5])).item()
            })
            
            all_targets.append({
                'spread_actual': sample['spread_actual'].item(),
                'total_actual': sample['total_actual'].item(),
                'home_won': sample['home_won'].item()
            })
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    targets_df = pd.DataFrame(all_targets)
    
    # Calculate metrics
    spread_mae = mean_absolute_error(targets_df['spread_actual'], predictions_df['spread_pred'])
    spread_rmse = np.sqrt(mean_squared_error(targets_df['spread_actual'], predictions_df['spread_pred']))
    
    total_mae = mean_absolute_error(targets_df['total_actual'], predictions_df['total_pred'])
    total_rmse = np.sqrt(mean_squared_error(targets_df['total_actual'], predictions_df['total_pred']))
    
    # Against the spread accuracy
    predictions_df['cover'] = predictions_df['spread_pred'] > 0
    targets_df['actual_cover'] = targets_df['spread_actual'] > 0
    ats_accuracy = (predictions_df['cover'] == targets_df['actual_cover']).mean()
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nSpread Predictions:")
    print(f"  MAE: {spread_mae:.2f} points")
    print(f"  RMSE: {spread_rmse:.2f} points")
    print(f"  ATS Accuracy: {ats_accuracy:.1%}")
    
    print(f"\nTotal Predictions:")
    print(f"  MAE: {total_mae:.2f} points")
    print(f"  RMSE: {total_rmse:.2f} points")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Spread scatter plot
    axes[0, 0].scatter(targets_df['spread_actual'], predictions_df['spread_pred'], 
                      alpha=0.5, s=10)
    axes[0, 0].plot([-40, 40], [-40, 40], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Spread')
    axes[0, 0].set_ylabel('Predicted Spread')
    axes[0, 0].set_title(f'Spread Predictions (MAE: {spread_mae:.2f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Total scatter plot
    axes[0, 1].scatter(targets_df['total_actual'], predictions_df['total_pred'],
                      alpha=0.5, s=10)
    axes[0, 1].plot([100, 180], [100, 180], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Total')
    axes[0, 1].set_ylabel('Predicted Total')
    axes[0, 1].set_title(f'Total Predictions (MAE: {total_mae:.2f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distributions
    spread_errors = predictions_df['spread_pred'] - targets_df['spread_actual']
    axes[1, 0].hist(spread_errors, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Spread Prediction Errors')
    axes[1, 0].axvline(0, color='red', linestyle='--')
    axes[1, 0].grid(True, alpha=0.3)
    
    total_errors = predictions_df['total_pred'] - targets_df['total_actual']
    axes[1, 1].hist(total_errors, bins=50, edgecolor='black')
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Total Prediction Errors')
    axes[1, 1].axvline(0, color='red', linestyle='--')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    plt.show()
    
    return predictions_df, targets_df