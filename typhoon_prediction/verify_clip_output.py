"""Verify CLIP aligned data"""
import numpy as np
import json
import h5py
import os

output_dir = r"D:\data_CLIP"

print("="*80)
print("CLIP Aligned Data Verification")
print("="*80)

# Check files
files = ['aligned_features.npy', 'aligned_features.h5', 'metadata.json', 'summary.json']
for f in files:
    path = os.path.join(output_dir, f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) / (1024*1024) if exists else 0
    print(f"{f}: {'EXISTS' if exists else 'MISSING'} ({size:.2f} MB)")

# Load and check features
if os.path.exists(os.path.join(output_dir, 'aligned_features.npy')):
    features = np.load(os.path.join(output_dir, 'aligned_features.npy'))
    print(f"\nFeatures shape: {features.shape}")
    print(f"Features dtype: {features.dtype}")
    print(f"Features stats: mean={features.mean():.4f}, std={features.std():.4f}, min={features.min():.4f}, max={features.max():.4f}")

# Load summary
if os.path.exists(os.path.join(output_dir, 'summary.json')):
    with open(os.path.join(output_dir, 'summary.json')) as f:
        summary = json.load(f)
    print(f"\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

# Check HDF5
if os.path.exists(os.path.join(output_dir, 'aligned_features.h5')):
    with h5py.File(os.path.join(output_dir, 'aligned_features.h5'), 'r') as f:
        print(f"\nHDF5 contents:")
        for key in f.keys():
            print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")

print("\n" + "="*80)
print("Verification Complete!")
print("="*80)

