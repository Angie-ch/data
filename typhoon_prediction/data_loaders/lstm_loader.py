"""
LSTM Data Loader for Typhoon Prediction
Loads sequential CLIP-aligned features organized by typhoon ID
"""

import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd


class TyphoonSequenceDataset(Dataset):
    """
    Dataset for loading typhoon sequences for LSTM prediction
    
    Directory structure:
    D:\typhoon_aligned\organized_by_typhoon\
        ├── 2018039N08151\
        │   ├── features.npy          # (T, 512) - full sequence
        │   ├── metadata.json         # List of metadata dicts
        │   └── timesteps\           # Individual timestep files (optional)
        │       ├── timestep_000000.npz
        │       └── ...
        └── 2018082N04147\
            └── ...
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 10,
        prediction_horizon: int = 1,
        stride: int = 1,
        target_type: str = 'wind',  # 'wind', 'location', 'both'
        normalize: bool = True,
        train_split: float = 0.8,
        mode: str = 'train',  # 'train', 'val', 'test', 'all'
        min_sequence_length: int = None
    ):
        """
        Args:
            data_dir: Path to organized_by_typhoon directory
            sequence_length: Number of timesteps to use as input
            prediction_horizon: Number of timesteps to predict ahead
            stride: Stride when creating sequences (1 = use all, 2 = skip 1)
            target_type: What to predict ('wind', 'location', 'both')
            normalize: Whether to normalize features
            train_split: Train/val split ratio
            mode: 'train', 'val', 'test', or 'all'
            min_sequence_length: Minimum sequence length to include
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.target_type = target_type
        self.normalize = normalize
        self.mode = mode
        
        if min_sequence_length is None:
            min_sequence_length = sequence_length + prediction_horizon
        
        # Load all typhoon sequences
        print(f"Loading typhoon sequences from {data_dir}...")
        self.sequences = self._load_sequences(min_sequence_length)
        
        if len(self.sequences) == 0:
            raise ValueError(f"No sequences found in {data_dir}")
        
        # Split into train/val/test
        self._split_sequences(train_split)
        
        # Compute normalization stats from training data
        if normalize:
            self._compute_normalization_stats()
        
        print(f"Loaded {len(self.sequences)} typhoon sequences")
        print(f"Mode: {mode}, Samples: {len(self.indices)}")
    
    def _load_sequences(self, min_length: int) -> List[Dict]:
        """Load all typhoon sequences"""
        sequences = []
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Iterate through typhoon folders
        for typhoon_folder in sorted(self.data_dir.iterdir()):
            if not typhoon_folder.is_dir():
                continue
            
            typhoon_id = typhoon_folder.name
            features_file = typhoon_folder / 'features.npy'
            metadata_file = typhoon_folder / 'metadata.json'
            
            if not features_file.exists() or not metadata_file.exists():
                print(f"Warning: Missing files for {typhoon_id}, skipping...")
                continue
            
            # Load features (T, feature_dim)
            features = np.load(features_file)
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Ensure sequence length matches
            if len(features) != len(metadata):
                print(f"Warning: Mismatch for {typhoon_id}: {len(features)} features vs {len(metadata)} metadata")
                min_len = min(len(features), len(metadata))
                features = features[:min_len]
                metadata = metadata[:min_len]
            
            # Check minimum length
            if len(features) < min_length:
                print(f"Skipping {typhoon_id}: sequence too short ({len(features)} < {min_length})")
                continue
            
            sequences.append({
                'typhoon_id': typhoon_id,
                'features': features,
                'metadata': metadata,
                'length': len(features)
            })
        
        return sequences
    
    def _split_sequences(self, train_split: float):
        """Split sequences into train/val/test"""
        n_sequences = len(self.sequences)
        n_train = int(n_sequences * train_split)
        n_val = int(n_sequences * (1 - train_split) / 2)
        
        # Split by typhoon (not by timesteps)
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, n_sequences))
        
        # Create sample indices (typhoon_idx, start_idx)
        self.indices = []
        
        if self.mode == 'train':
            seq_indices = train_indices
        elif self.mode == 'val':
            seq_indices = val_indices
        elif self.mode == 'test':
            seq_indices = test_indices
        else:  # 'all'
            seq_indices = list(range(n_sequences))
        
        for seq_idx in seq_indices:
            seq = self.sequences[seq_idx]
            seq_len = seq['length']
            
            # Create sliding windows
            for start_idx in range(0, seq_len - self.sequence_length - self.prediction_horizon + 1, self.stride):
                self.indices.append((seq_idx, start_idx))
    
    def _compute_normalization_stats(self):
        """Compute mean and std from training sequences"""
        if self.mode != 'train':
            # Load stats from a saved file or use defaults
            stats_file = self.data_dir.parent / 'normalization_stats.json'
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                self.feature_mean = np.array(stats['feature_mean'])
                self.feature_std = np.array(stats['feature_std'])
                return
        
        # Compute from training data
        print("Computing normalization statistics...")
        all_features = []
        for seq in self.sequences[:int(len(self.sequences) * 0.8)]:  # Use first 80% for stats
            all_features.append(seq['features'])
        
        all_features = np.vstack(all_features)
        self.feature_mean = all_features.mean(axis=0)
        self.feature_std = all_features.std(axis=0) + 1e-8  # Avoid division by zero
        
        # Save stats
        stats_file = self.data_dir.parent / 'normalization_stats.json'
        stats = {
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist()
        }
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved normalization stats to {stats_file}")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            sequence: (sequence_length, feature_dim) - input sequence
            target: (prediction_horizon, target_dim) - target values
            metadata: Dict with typhoon_id, times, etc.
        """
        seq_idx, start_idx = self.indices[idx]
        seq = self.sequences[seq_idx]
        
        # Extract sequence
        end_idx = start_idx + self.sequence_length
        sequence = seq['features'][start_idx:end_idx].copy()  # (T, D)
        
        # Extract target
        target_start = end_idx
        target_end = target_start + self.prediction_horizon
        target_metadata = seq['metadata'][target_start:target_end]
        
        # Build target based on target_type
        if self.target_type == 'wind':
            target = np.array([m['wind'] for m in target_metadata])  # (H,)
            target = target.reshape(-1, 1)  # (H, 1)
        elif self.target_type == 'location':
            target = np.array([[m['lat'], m['lon']] for m in target_metadata])  # (H, 2)
        elif self.target_type == 'both':
            target = np.array([[m['wind'], m['lat'], m['lon']] for m in target_metadata])  # (H, 3)
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")
        
        # Normalize features
        if self.normalize:
            sequence = (sequence - self.feature_mean) / self.feature_std
        
        # Convert to tensors
        sequence = torch.FloatTensor(sequence)  # (T, D)
        target = torch.FloatTensor(target)  # (H, target_dim)
        
        # Metadata
        metadata = {
            'typhoon_id': seq['typhoon_id'],
            'start_time': seq['metadata'][start_idx].get('time', ''),
            'target_time': target_metadata[0].get('time', '') if target_metadata else '',
            'start_lat': seq['metadata'][start_idx].get('lat', 0.0),
            'start_lon': seq['metadata'][start_idx].get('lon', 0.0),
        }
        
        return sequence, target, metadata


def create_dataloaders(
    data_dir: str,
    sequence_length: int = 10,
    prediction_horizon: int = 1,
    batch_size: int = 32,
    target_type: str = 'wind',
    train_split: float = 0.8,
    normalize: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = TyphoonSequenceDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        target_type=target_type,
        normalize=normalize,
        train_split=train_split,
        mode='train',
        **kwargs
    )
    
    val_dataset = TyphoonSequenceDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        target_type=target_type,
        normalize=normalize,
        train_split=train_split,
        mode='val',
        **kwargs
    )
    
    test_dataset = TyphoonSequenceDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        target_type=target_type,
        normalize=normalize,
        train_split=train_split,
        mode='test',
        **kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == '__main__':
    # Example: Load data for LSTM training
    data_dir = r"D:\typhoon_aligned\organized_by_typhoon"
    
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        sequence_length=10,  # Use 10 timesteps to predict
        prediction_horizon=1,  # Predict 1 timestep ahead
        batch_size=16,
        target_type='wind',  # Predict wind speed
        train_split=0.8
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading a batch
    print("\nTesting batch loading...")
    for sequence, target, metadata in train_loader:
        print(f"Batch - Sequence shape: {sequence.shape}")  # (batch, T, D)
        print(f"Batch - Target shape: {target.shape}")  # (batch, H, target_dim)
        print(f"Batch - Metadata keys: {list(metadata.keys())}")
        print(f"Sample typhoon_id: {metadata['typhoon_id'][0]}")
        break
    
    print("\nData loading successful!")

