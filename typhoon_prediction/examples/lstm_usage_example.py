"""
LSTM Data Loading Example for Typhoon Prediction

This shows how to load the organized CLIP-aligned features for LSTM training.
"""

import torch
from data_loaders.lstm_loader import create_dataloaders, TyphoonSequenceDataset

# Data directory (organized by typhoon ID)
DATA_DIR = r"D:\typhoon_aligned\organized_by_typhoon"

def example_basic_loading():
    """Basic example of loading data"""
    print("="*80)
    print("Example: Basic Data Loading")
    print("="*80)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=DATA_DIR,
        sequence_length=10,      # Use 10 timesteps as input
        prediction_horizon=1,    # Predict 1 timestep ahead
        batch_size=16,
        target_type='wind',       # Predict wind speed
        train_split=0.8
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Load a batch
    sequence, target, metadata = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Sequence: {sequence.shape}")  # (batch, T, feature_dim)
    print(f"  Target: {target.shape}")      # (batch, H, target_dim)
    print(f"  Metadata: {list(metadata.keys())}")
    
    return train_loader, val_loader, test_loader


def example_different_targets():
    """Example with different prediction targets"""
    print("\n" + "="*80)
    print("Example: Different Prediction Targets")
    print("="*80)
    
    # Predict wind speed only
    train_loader_wind, _, _ = create_dataloaders(
        data_dir=DATA_DIR,
        sequence_length=10,
        prediction_horizon=1,
        batch_size=16,
        target_type='wind',  # (batch, 1)
    )
    
    # Predict location (lat, lon)
    train_loader_loc, _, _ = create_dataloaders(
        data_dir=DATA_DIR,
        sequence_length=10,
        prediction_horizon=1,
        batch_size=16,
        target_type='location',  # (batch, 2)
    )
    
    # Predict both wind and location
    train_loader_both, _, _ = create_dataloaders(
        data_dir=DATA_DIR,
        sequence_length=10,
        prediction_horizon=1,
        batch_size=16,
        target_type='both',  # (batch, 3)
    )
    
    seq_w, t_w, _ = next(iter(train_loader_wind))
    seq_l, t_l, _ = next(iter(train_loader_loc))
    seq_b, t_b, _ = next(iter(train_loader_both))
    
    print(f"Wind only - Target shape: {t_w.shape}")
    print(f"Location only - Target shape: {t_l.shape}")
    print(f"Both - Target shape: {t_b.shape}")


def example_multi_step_prediction():
    """Example for multi-step prediction"""
    print("\n" + "="*80)
    print("Example: Multi-Step Prediction")
    print("="*80)
    
    # Predict 5 timesteps ahead
    train_loader, _, _ = create_dataloaders(
        data_dir=DATA_DIR,
        sequence_length=15,      # Use longer context
        prediction_horizon=5,    # Predict 5 timesteps ahead
        batch_size=16,
        target_type='wind',
    )
    
    sequence, target, metadata = next(iter(train_loader))
    print(f"Sequence shape: {sequence.shape}")  # (batch, 15, 512)
    print(f"Target shape: {target.shape}")      # (batch, 5, 1)


def example_training_loop():
    """Example training loop"""
    print("\n" + "="*80)
    print("Example: Training Loop")
    print("="*80)
    
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=DATA_DIR,
        sequence_length=10,
        prediction_horizon=1,
        batch_size=16,
        target_type='wind',
    )
    
    # Simple LSTM model (example)
    class SimpleLSTM(torch.nn.Module):
        def __init__(self, input_dim=512, hidden_dim=128, output_dim=1):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            # x: (batch, T, D)
            lstm_out, (h_n, c_n) = self.lstm(x)
            # Use last timestep
            output = self.fc(lstm_out[:, -1, :])  # (batch, output_dim)
            return output
    
    model = SimpleLSTM(input_dim=512, hidden_dim=128, output_dim=1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(1):  # Just one epoch for example
        total_loss = 0
        for batch_idx, (sequence, target, metadata) in enumerate(train_loader):
            # sequence: (batch, T, D)
            # target: (batch, H, target_dim)
            
            # Forward pass
            output = model(sequence)  # (batch, output_dim)
            loss = criterion(output, target.squeeze(-1))  # (batch, 1) -> (batch,)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch}, Average Loss: {total_loss/len(train_loader):.4f}")


if __name__ == '__main__':
    # Run examples
    example_basic_loading()
    example_different_targets()
    example_multi_step_prediction()
    example_training_loop()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)

