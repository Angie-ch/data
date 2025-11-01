import torch
import torch.nn as nn


class DiffusionModule(nn.Module):
    """Diffusion Module for typhoon trajectory generation."""
    
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor
        Returns:
            encoded: Encoded tensor
        """
        return self.encoder(x)


class HistoryLSTM(nn.Module):
    """LSTM for processing historical typhoon data."""
    
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            output: LSTM output of shape (batch, seq_len, hidden_dim)
        """
        output, _ = self.lstm(x)
        return output


class FutureLSTM(nn.Module):
    """LSTM for generating future typhoon trajectories."""
    
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            output: LSTM output of shape (batch, seq_len, hidden_dim)
        """
        output, _ = self.lstm(x)
        return output

