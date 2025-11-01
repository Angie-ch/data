import torch
import torch.nn as nn


class FeatureCalibration(nn.Module):
    """Feature Calibration Module for typhoon prediction."""
    
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.calibration = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor
        Returns:
            calibrated: Calibrated tensor
        """
        return self.calibration(x)

