import torch
import torch.nn as nn
import sys
import os

# Add CLIP to path
clip_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CLIP')
if os.path.exists(clip_path):
    sys.path.insert(0, clip_path)
try:
    import clip  # type: ignore
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    # Only print warning if CLIP directory exists but import fails
    if os.path.exists(clip_path):
        print("Warning: CLIP directory found but import failed. Using fallback alignment module.")


class DualMultiModalityAlignment(nn.Module):
    """
    Dual Multi-Modality Alignment Module for typhoon prediction using CLIP.
    
    Aligns image features (e.g., satellite imagery) with text/metadata features
    (e.g., weather data, trajectory descriptions) using CLIP's contrastive learning.
    """
    
    def __init__(self, 
                 input_dim=128, 
                 hidden_dim=256,
                 clip_model_name="ViT-B/32",
                 use_clip=True,
                 device=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_clip = use_clip and CLIP_AVAILABLE
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if self.use_clip:
            # Load CLIP model
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
            self.clip_model.eval()
            
            # Get CLIP feature dimensions
            with torch.no_grad():
                dummy_text = clip.tokenize(["dummy"]).to(self.device)
                dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
                text_feat = self.clip_model.encode_text(dummy_text)
                image_feat = self.clip_model.encode_image(dummy_image)
                self.clip_text_dim = text_feat.shape[-1]
                self.clip_image_dim = image_feat.shape[-1]
            
            # Projection layers to map CLIP features to hidden_dim
            self.image_proj = nn.Linear(self.clip_image_dim, hidden_dim)
            self.text_proj = nn.Linear(self.clip_text_dim, hidden_dim)
            self.align_proj = nn.Linear(self.clip_text_dim, hidden_dim)
        else:
            # Fallback: Simple linear alignment
            self.align_layer = nn.Linear(input_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Move all modules to device
        if self.use_clip:
            self.image_proj = self.image_proj.to(self.device)
            self.text_proj = self.text_proj.to(self.device)
            self.align_proj = self.align_proj.to(self.device)
        self.norm = self.norm.to(self.device)
        self.dropout = self.dropout.to(self.device)
        
    def encode_image(self, images):
        """
        Encode images using CLIP image encoder.
        
        Args:
            images: Image tensor of shape (batch, C, H, W) or (batch, seq_len, C, H, W)
        Returns:
            image_features: Encoded image features
        """
        if not self.use_clip:
            # Fallback: simple projection
            if len(images.shape) == 4:
                # (batch, C, H, W) -> flatten and project
                B, C, H, W = images.shape
                images = images.view(B, -1)
            elif len(images.shape) == 5:
                # (batch, seq_len, C, H, W) -> flatten
                B, T, C, H, W = images.shape
                images = images.view(B, T, -1)
            return self.align_layer(images)
        
        # Move images to device
        images = images.to(self.device)
        
        # Use CLIP image encoder
        with torch.no_grad():
            if len(images.shape) == 4:
                # Single image per batch
                image_features = self.clip_model.encode_image(images)
            elif len(images.shape) == 5:
                # Sequence of images: (batch, seq_len, C, H, W)
                B, T, C, H, W = images.shape
                images = images.view(B * T, C, H, W)
                image_features = self.clip_model.encode_image(images)
                image_features = image_features.view(B, T, -1)
            else:
                # Assume already encoded features
                image_features = images
        
        # Project to hidden_dim (ensure float32 for compatibility)
        image_features = image_features.float()
        image_features = self.image_proj(image_features)
        return image_features
    
    def encode_text(self, text_inputs):
        """
        Encode text using CLIP text encoder.
        
        Args:
            text_inputs: Either tokenized text tensor or list of strings
        Returns:
            text_features: Encoded text features
        """
        if not self.use_clip:
            # Fallback: simple projection
            if isinstance(text_inputs, torch.Tensor):
                return self.align_layer(text_inputs)
            else:
                # Convert to tensor if needed
                text_tensor = torch.tensor(text_inputs).to(self.device)
                return self.align_layer(text_tensor)
        
        # Tokenize if needed (list of strings)
        if isinstance(text_inputs, list):
            text_tokens = clip.tokenize(text_inputs, truncate=True).to(self.device)
        else:
            text_tokens = text_inputs
        
        # Use CLIP text encoder
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        
        # Project to hidden_dim (ensure float32 for compatibility)
        text_features = text_features.float()
        text_features = self.text_proj(text_features)
        return text_features
    
    def align_modalities(self, image_features, text_features):
        """
        Align image and text features using CLIP's contrastive alignment.
        
        Args:
            image_features: Image features tensor
            text_features: Text features tensor
        Returns:
            aligned_features: Aligned multi-modal features
        """
        if not self.use_clip:
            # Fallback: simple concatenation and projection
            aligned = torch.cat([image_features, text_features], dim=-1)
            if aligned.shape[-1] != self.hidden_dim:
                aligned = self.align_layer(aligned)
            return aligned
        
        # Normalize features
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute alignment via similarity
        if len(image_features.shape) == 2 and len(text_features.shape) == 2:
            # (batch, hidden_dim) x (batch, hidden_dim)
            similarity = torch.matmul(image_features_norm, text_features_norm.t())
            # Use diagonal elements (matching pairs)
            alignment_weight = torch.diag(similarity).unsqueeze(-1)
        else:
            # Handle sequence case
            alignment_weight = torch.sum(image_features_norm * text_features_norm, dim=-1, keepdim=True)
        
        # Fuse features with alignment weighting
        aligned = image_features + alignment_weight * text_features
        # Additional alignment: combine both modalities
        aligned = aligned + image_features * 0.5 + text_features * 0.5
        
        return aligned
        
    def forward(self, x, image_input=None, text_input=None):
        """
        Forward pass for alignment.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) - fallback input
            image_input: Optional image tensor (batch, C, H, W) or (batch, seq_len, C, H, W)
            text_input: Optional text input (list of strings or tokenized tensor)
        Returns:
            aligned: Aligned tensor of shape (batch, seq_len, hidden_dim)
        """
        if image_input is not None and text_input is not None:
            # Dual modality alignment with CLIP
            image_features = self.encode_image(image_input)
            text_features = self.encode_text(text_input)
            aligned = self.align_modalities(image_features, text_features)
        elif image_input is not None:
            # Image-only encoding
            aligned = self.encode_image(image_input)
        elif text_input is not None:
            # Text-only encoding
            aligned = self.encode_text(text_input)
        else:
            # Fallback: simple projection
            if self.use_clip:
                # Convert to text-like features
                text_repr = [f"feature_{i}" for i in range(x.shape[-1])]
                aligned = self.encode_text(text_repr)
                # Average over sequence
                if len(x.shape) == 3:
                    aligned = aligned.mean(dim=0, keepdim=True).repeat(x.shape[0], x.shape[1], 1)
            else:
                aligned = self.align_layer(x)
        
        # Normalize and apply dropout
        aligned = self.norm(aligned)
        aligned = self.dropout(aligned)
        
        return aligned

