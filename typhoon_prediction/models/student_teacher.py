import torch
import torch.nn as nn
from models.generation import HistoryLSTM, FutureLSTM


class StudentModel(nn.Module):
    """Student Model for typhoon prediction."""
    
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoders for different inputs
        self.hist_encoder = HistoryLSTM(input_dim, hidden_dim)
        self.pos_encoder = nn.Linear(2, hidden_dim)
        self.phys_encoder = nn.Linear(input_dim, hidden_dim)
        self.temp_encoder = nn.Linear(1, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Future predictor
        self.future_lstm = FutureLSTM(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, hist_data, pos_input, phys_input, temp_input):
        """
        Args:
            hist_data: Historical data (batch, hist_len, 128)
            pos_input: Position input (batch, total_len, 2)
            phys_input: Physical input (batch, total_len, 128)
            temp_input: Temperature input (batch, total_len, 1)
        Returns:
            pred: Prediction (batch, fut_len, 128)
            features: Intermediate features
        """
        batch_size = hist_data.size(0)
        hist_len = hist_data.size(1)
        fut_len = pos_input.size(1) - hist_len
        
        # Encode historical data
        hist_encoded = self.hist_encoder(hist_data)  # (batch, hist_len, hidden_dim)
        hist_last = hist_encoded[:, -1, :]  # (batch, hidden_dim)
        
        # Encode future inputs
        fut_pos = pos_input[:, hist_len:, :]  # (batch, fut_len, 2)
        fut_phys = phys_input[:, hist_len:, :]  # (batch, fut_len, 128)
        fut_temp = temp_input[:, hist_len:, :]  # (batch, fut_len, 1)
        
        # Encode each modality
        pos_encoded = self.pos_encoder(fut_pos)  # (batch, fut_len, hidden_dim)
        phys_encoded = self.phys_encoder(fut_phys)  # (batch, fut_len, hidden_dim)
        temp_encoded = self.temp_encoder(fut_temp)  # (batch, fut_len, hidden_dim)
        
        # Repeat historical context for each future timestep
        hist_context = hist_last.unsqueeze(1).repeat(1, fut_len, 1)  # (batch, fut_len, hidden_dim)
        
        # Fuse features
        fused = torch.cat([hist_context, pos_encoded, phys_encoded, temp_encoded], dim=-1)
        fused = self.fusion(fused)  # (batch, fut_len, hidden_dim)
        
        # Predict future
        future_out = self.future_lstm(fused)  # (batch, fut_len, hidden_dim)
        pred = self.output(future_out)  # (batch, fut_len, 128)
        
        return pred, fused


class TeacherModel(nn.Module):
    """Teacher Model for typhoon prediction with ground truth access."""
    
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoders
        self.hist_encoder = HistoryLSTM(input_dim, hidden_dim)
        self.gt_encoder = HistoryLSTM(input_dim, hidden_dim)
        self.pos_encoder = nn.Linear(2, hidden_dim)
        self.phys_encoder = nn.Linear(input_dim, hidden_dim)
        self.temp_encoder = nn.Linear(1, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Future predictor
        self.future_lstm = FutureLSTM(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, hist_data, fut_gt, pos_input, phys_input, temp_input):
        """
        Args:
            hist_data: Historical data (batch, hist_len, 128)
            fut_gt: Future ground truth (batch, fut_len, 128)
            pos_input: Position input (batch, total_len, 2)
            phys_input: Physical input (batch, total_len, 128)
            temp_input: Temperature input (batch, total_len, 1)
        Returns:
            pred: Prediction (batch, fut_len, 128)
            features: Intermediate features
        """
        batch_size = hist_data.size(0)
        hist_len = hist_data.size(1)
        fut_len = fut_gt.size(1)
        
        # Encode historical data
        hist_encoded = self.hist_encoder(hist_data)  # (batch, hist_len, hidden_dim)
        hist_last = hist_encoded[:, -1, :]  # (batch, hidden_dim)
        
        # Encode ground truth future
        gt_encoded = self.gt_encoder(fut_gt)  # (batch, fut_len, hidden_dim)
        
        # Encode future inputs
        fut_pos = pos_input[:, hist_len:, :]  # (batch, fut_len, 2)
        fut_phys = phys_input[:, hist_len:, :]  # (batch, fut_len, 128)
        fut_temp = temp_input[:, hist_len:, :]  # (batch, fut_len, 1)
        
        # Encode each modality
        pos_encoded = self.pos_encoder(fut_pos)  # (batch, fut_len, hidden_dim)
        phys_encoded = self.phys_encoder(fut_phys)  # (batch, fut_len, hidden_dim)
        temp_encoded = self.temp_encoder(fut_temp)  # (batch, fut_len, hidden_dim)
        
        # Repeat historical context for each future timestep
        hist_context = hist_last.unsqueeze(1).repeat(1, fut_len, 1)  # (batch, fut_len, hidden_dim)
        
        # Fuse features (including ground truth)
        fused = torch.cat([hist_context, gt_encoded, pos_encoded, phys_encoded, temp_encoded], dim=-1)
        fused = self.fusion(fused)  # (batch, fut_len, hidden_dim)
        
        # Predict future
        future_out = self.future_lstm(fused)  # (batch, fut_len, hidden_dim)
        pred = self.output(future_out)  # (batch, fut_len, 128)
        
        return pred, fused

