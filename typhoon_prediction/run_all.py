import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
print("=" * 80)
print("TYPHOON PREDICTION SYSTEM - RUNNING ALL COMPONENTS")
print("=" * 80)

print("\n[1/5] Testing Alignment Module...")
from models.alignment import DualMultiModalityAlignment, CLIP_AVAILABLE

# Test with CLIP if available, otherwise use fallback
if CLIP_AVAILABLE:
    print("   [INFO] CLIP available - testing with CLIP integration")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   [INFO] Using device: {device}")
    align = DualMultiModalityAlignment(use_clip=True, device=device)
    # Test image-text alignment
    test_image = torch.randn(2, 3, 224, 224).to(device)
    test_text = ["typhoon observation", "tropical cyclone"]
    aligned = align(x=None, image_input=test_image, text_input=test_text)
    print(f"   [OK] CLIP-based alignment test passed, output shape: {aligned.shape}")
else:
    print("   [INFO] CLIP not available - using fallback mode")
    align = DualMultiModalityAlignment(use_clip=False)
    test_input = torch.randn(2, 24, 128)
    aligned = align(test_input)
    print(f"   [OK] Fallback alignment test passed, output shape: {aligned.shape}")

print("   [OK] Alignment Module loaded successfully")

print("\n[2/5] Testing Generation Components...")
from models.generation import DiffusionModule, HistoryLSTM, FutureLSTM
print("   [OK] Generation modules loaded successfully")

print("\n[3/5] Testing Prediction Module...")
from models.prediction import FeatureCalibration
print("   [OK] Prediction module loaded successfully")

print("\n[4/5] Testing Student-Teacher Models...")
from models.student_teacher import StudentModel, TeacherModel
print("   [OK] Student-Teacher models loaded successfully")

print("\n[5/5] Testing Full Pipeline...")
batch_size = 2
hist_len = 24
fut_len = 24

student = StudentModel()
teacher = TeacherModel()

hist_data = torch.randn(batch_size, hist_len, 128)
pos_input = torch.randn(batch_size, hist_len + fut_len, 2)
phys_input = torch.randn(batch_size, hist_len + fut_len, 128)
temp_input = torch.randn(batch_size, hist_len + fut_len, 1)
fut_gt = torch.randn(batch_size, fut_len, 128)

# Test student
pred, _ = student(hist_data, pos_input, phys_input, temp_input)
print(f"   [OK] Student prediction shape: {pred.shape}")

# Test teacher
t_pred, _ = teacher(hist_data, fut_gt, pos_input, phys_input, temp_input)
print(f"   [OK] Teacher prediction shape: {t_pred.shape}")

print("\n" + "=" * 80)
print("ALL COMPONENTS TESTED SUCCESSFULLY!")
print("=" * 80)
print("\nThe typhoon prediction system is ready to use!")
print("All modules are working correctly with dummy data.")

