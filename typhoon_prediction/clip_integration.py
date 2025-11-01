"""
CLIP Integration Example for Typhoon Prediction

This module demonstrates how to use CLIP for multi-modality alignment
in typhoon prediction tasks, aligning satellite imagery with weather metadata.
"""

import torch
import sys
import os

# Add CLIP to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CLIP'))
try:
    import clip
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

from models.alignment import DualMultiModalityAlignment


def example_image_text_alignment():
    """Example of aligning satellite imagery with text descriptions."""
    if not CLIP_AVAILABLE:
        print("CLIP not available. Cannot run example.")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize alignment module
    align_module = DualMultiModalityAlignment(
        input_dim=128,
        hidden_dim=256,
        clip_model_name="ViT-B/32",
        use_clip=True,
        device=device
    )
    
    # Example: Satellite image (simulated)
    # In practice, this would be real satellite imagery from Himawari-8
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Example: Weather metadata as text descriptions
    text_descriptions = [
        "typhoon center latitude 20.5 longitude 130.2 wind speed 45 m/s",
        "tropical cyclone intensity category 3 pressure 945 hPa"
    ]
    
    # Align image and text features
    aligned_features = align_module(
        x=None,
        image_input=image,
        text_input=text_descriptions
    )
    
    print(f"Aligned features shape: {aligned_features.shape}")
    print(f"Successfully aligned {batch_size} image-text pairs")
    
    return aligned_features


def example_sequence_alignment():
    """Example of aligning sequences of images with sequences of text."""
    if not CLIP_AVAILABLE:
        print("CLIP not available. Cannot run example.")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize alignment module
    align_module = DualMultiModalityAlignment(
        input_dim=128,
        hidden_dim=256,
        clip_model_name="ViT-B/32",
        use_clip=True,
        device=device
    )
    
    # Example: Sequence of satellite images (e.g., 24 time steps)
    seq_len = 24
    batch_size = 2
    image_sequence = torch.randn(batch_size, seq_len, 3, 224, 224).to(device)
    
    # Example: Sequence of weather descriptions
    text_sequence = [
        f"typhoon observation at time step {i}: wind speed {40+i} m/s"
        for i in range(seq_len)
    ]
    
    # Align sequences
    aligned_sequence = align_module(
        x=None,
        image_input=image_sequence,
        text_input=text_sequence
    )
    
    print(f"Aligned sequence shape: {aligned_sequence.shape}")
    print(f"Successfully aligned {batch_size} sequences of length {seq_len}")
    
    return aligned_sequence


def example_fallback_mode():
    """Example using fallback mode when CLIP is not available."""
    # Initialize without CLIP
    align_module = DualMultiModalityAlignment(
        input_dim=128,
        hidden_dim=256,
        use_clip=False
    )
    
    # Regular tensor input
    batch_size = 2
    seq_len = 24
    x = torch.randn(batch_size, seq_len, 128)
    
    aligned = align_module(x)
    print(f"Fallback mode aligned features shape: {aligned.shape}")
    
    return aligned


if __name__ == "__main__":
    print("=" * 80)
    print("CLIP Integration Examples for Typhoon Prediction")
    print("=" * 80)
    
    print("\n[1] Image-Text Alignment Example:")
    if CLIP_AVAILABLE:
        example_image_text_alignment()
    else:
        print("   Skipping - CLIP not available")
    
    print("\n[2] Sequence Alignment Example:")
    if CLIP_AVAILABLE:
        example_sequence_alignment()
    else:
        print("   Skipping - CLIP not available")
    
    print("\n[3] Fallback Mode Example:")
    example_fallback_mode()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)

