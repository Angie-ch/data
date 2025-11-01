# Typhoon CLIP Alignment - Data Organization Guide

## Overview

The CLIP alignment script now saves data in **two formats**:

1. **Flat format** (original): All samples in single files
2. **Organized format** (new): Data organized by typhoon ID for LSTM sequences

## Directory Structure

After running `clip_align_data.py`, you'll have:

```
D:\typhoon_aligned\
├── aligned_features.npy           # All features (flat)
├── aligned_features.h5           # All features + metadata (flat)
├── metadata.json                 # All metadata (flat)
├── summary.json                  # Processing summary
└── organized_by_typhoon\         # NEW: Organized by typhoon
    ├── 2018039N08151\
    │   ├── features.npy          # (T, 512) - Full sequence
    │   ├── metadata.json         # List of metadata dicts
    │   └── timesteps\            # Individual timestep files
    │       ├── timestep_000000.npz
    │       ├── timestep_000001.npz
    │       └── ...
    ├── 2018082N04147\
    │   └── ...
    └── ...
```

## Why This Organization?

For **LSTM prediction**, you need:
- ✅ **Sequential data** organized by typhoon (time-ordered)
- ✅ **Easy access** to load full sequences for each typhoon
- ✅ **Metadata** aligned with features for each timestep
- ✅ **Flexible loading** for different sequence lengths

## Loading Data for LSTM Training

### Quick Start

```python
from data_loaders.lstm_loader import create_dataloaders

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir=r"D:\typhoon_aligned\organized_by_typhoon",
    sequence_length=10,      # Use 10 timesteps as input
    prediction_horizon=1,    # Predict 1 timestep ahead
    batch_size=32,
    target_type='wind',       # Predict wind speed
    train_split=0.8
)

# Use in training loop
for sequence, target, metadata in train_loader:
    # sequence: (batch, 10, 512) - input features
    # target: (batch, 1, 1) - wind speed to predict
    # metadata: dict with typhoon_id, times, etc.
    ...
```

### Advanced Usage

```python
from data_loaders.lstm_loader import TyphoonSequenceDataset

# Create custom dataset
dataset = TyphoonSequenceDataset(
    data_dir=r"D:\typhoon_aligned\organized_by_typhoon",
    sequence_length=15,           # Longer context
    prediction_horizon=5,         # Predict 5 steps ahead
    target_type='both',           # Predict wind + location
    stride=2,                     # Skip every other timestep
    normalize=True,
    mode='train'
)

# Load sequences
sequence, target, metadata = dataset[0]
# sequence: (15, 512)
# target: (5, 3) - [wind, lat, lon] for 5 timesteps
```

## Target Types

- `'wind'`: Predict wind speed only → `target.shape = (batch, H, 1)`
- `'location'`: Predict latitude/longitude → `target.shape = (batch, H, 2)`
- `'both'`: Predict wind + location → `target.shape = (batch, H, 3)`

## Example: Simple LSTM Model

```python
import torch
import torch.nn as nn
from data_loaders.lstm_loader import create_dataloaders

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir=r"D:\typhoon_aligned\organized_by_typhoon",
    sequence_length=10,
    prediction_horizon=1,
    batch_size=32,
    target_type='wind'
)

# Define LSTM model
class TyphoonLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: (batch, T, D)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last timestep for prediction
        output = self.fc(lstm_out[:, -1, :])  # (batch, output_dim)
        return output

# Initialize model
model = TyphoonLSTM(input_dim=512, hidden_dim=128, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(10):
    for sequence, target, metadata in train_loader:
        # Forward pass
        output = model(sequence)  # (batch, 1)
        loss = criterion(output, target.squeeze(-1))  # (batch, 1) -> (batch,)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Data Format Details

### `features.npy`
- Shape: `(T, 512)` where T = number of timesteps
- Type: `float16` numpy array
- Content: CLIP-aligned features (visual + text)

### `metadata.json`
- Format: List of dictionaries
- Each dict contains:
  ```json
  {
    "typhoon_id": "2018039N08151",
    "typhoon_name": "SANBA",
    "time": "8020600.0",
    "year": 2018,
    "lat": 9.5,
    "lon": 149.8,
    "wind": 20.0,
    "data_file": "era5_merged_20180208_0600_fused.nc",
    "text_description": "..."
  }
  ```

### `timesteps/timestep_XXXXXX.npz`
- Individual timestep files (optional, for easy access)
- Contains: `feature`, `time`, `lat`, `lon`, `wind`, `year`

## Key Features

1. **Automatic time sorting**: Sequences are sorted by time
2. **Train/val/test split**: Automatic splitting by typhoon (not by timestep)
3. **Normalization**: Optional feature normalization using training stats
4. **Sliding windows**: Creates multiple sequences from each typhoon
5. **Flexible targets**: Predict wind, location, or both

## Notes

- **Sequences are sorted by time** automatically
- **Train/val/test split is by typhoon** (not random timesteps) to avoid data leakage
- **Normalization stats** are computed from training data and saved for inference
- **Minimum sequence length** filter ensures all sequences are long enough

## Files

- `clip_align_data.py`: Main alignment script (updated to save organized format)
- `data_loaders/lstm_loader.py`: LSTM data loader
- `examples/lstm_usage_example.py`: Usage examples

