"""
Main script to run the complete training pipeline.
"""
import torch
from dataset import BasketballDataset
from model import BasketballPredictor, AdvancedBasketballPredictor
from train import Trainer
from torch.utils.data import DataLoader
import yaml
import argparse
import json

def main(config_path='config.yaml'):
    """
    Main training pipeline.
    """
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("BASKETBALL PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Create datasets
    print("\nLoading datasets...")
    
    train_dataset = BasketballDataset(
        parquet_path=config['data']['parquet_path'],
        db_config=config['data'].get('db_config'),
        mode='train',
        split_date=config['data']['split_date'],
        scale_features=True,
        fetch_plays=config['data'].get('fetch_plays', False)
    )
    
    val_dataset = BasketballDataset(
        parquet_path=config['data']['parquet_path'],
        db_config=config['data'].get('db_config'),
        mode='val',
        split_date=config['data']['split_date'],
        scale_features=True,
        fetch_plays=config['data'].get('fetch_plays', False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True
    )
    
    print(f"Train set: {len(train_dataset)} games")
    print(f"Validation set: {len(val_dataset)} games")
    
    # Create model
    print("\nInitializing model...")
    
    input_dim = train_dataset.features.shape[1]
    
    if config['model']['type'] == 'advanced':
        model = AdvancedBasketballPredictor(
            input_dim=input_dim,
            config=config['model']
        )
    else:
        model = BasketballPredictor(
            input_dim=input_dim,
            hidden_dims=config['model']['hidden_dims'],
            dropout=config['model']['dropout']
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training']
    )
    
    # Train model
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Save final results
    with open('training_results.json', 'w') as f:
        json.dump({
            'history': trainer.history,
            'best_val_loss': trainer.best_val_loss,
            'config': config
        }, f, indent=2)
    
    print("\nResults saved to training_results.json")
    print("Best model saved to best_model.pth")
    print("Training plots saved to training_history.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config)