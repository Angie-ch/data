import h5py
import numpy as np
import json

print("="*80)
print("Inspecting Output Files")
print("="*80)

# Check NumPy file
print("\n1. NumPy Array (.npy):")
arr = np.load('D:/typhoon_aligned/aligned_features.npy')
print(f"   Shape: {arr.shape}")
print(f"   Dtype: {arr.dtype}")
print(f"   Min: {arr.min():.4f}, Max: {arr.max():.4f}, Mean: {arr.mean():.4f}")

# Check HDF5 file
print("\n2. HDF5 File (.h5):")
with h5py.File('D:/typhoon_aligned/aligned_features.h5', 'r') as f:
    print(f"   Datasets: {list(f.keys())}")
    feat = f['features']
    print(f"   Features shape: {feat.shape}")
    print(f"   Features dtype: {feat.dtype}")
    print(f"   Features compression: {feat.compression}")
    
    if 'typhoon_ids' in f:
        print(f"\n   Metadata arrays:")
        print(f"   - typhoon_ids: {f['typhoon_ids'].shape} ({f['typhoon_ids'].dtype})")
        print(f"   - typhoon_names: {f['typhoon_names'].shape} ({f['typhoon_names'].dtype})")
        print(f"   - times: {f['times'].shape} ({f['times'].dtype})")
        print(f"   - years: {f['years'].shape} ({f['years'].dtype})")
        print(f"   - lats: {f['lats'].shape} ({f['lats'].dtype})")
        print(f"   - lons: {f['lons'].shape} ({f['lons'].dtype})")
        print(f"   - winds: {f['winds'].shape} ({f['winds'].dtype})")
        
        print(f"\n   Sample data (first 3):")
        print(f"   - typhoon_ids: {[x.decode() for x in f['typhoon_ids'][:3]]}")
        print(f"   - typhoon_names: {[x.decode() for x in f['typhoon_names'][:3]]}")
        print(f"   - years: {f['years'][:3].tolist()}")
        print(f"   - lats: {f['lats'][:3].tolist()}")
        print(f"   - lons: {f['lons'][:3].tolist()}")
        print(f"   - winds: {f['winds'][:3].tolist()}")

# Check summary
print("\n3. Summary JSON:")
with open('D:/typhoon_aligned/summary.json', 'r') as f:
    summary = json.load(f)
    for key, value in summary.items():
        print(f"   {key}: {value}")

# Check metadata sample
print("\n4. Metadata JSON (first 3 records):")
with open('D:/typhoon_aligned/metadata.json', 'r') as f:
    metadata = json.load(f)
    for i, record in enumerate(metadata[:3]):
        print(f"\n   Record {i+1}:")
        for key, value in record.items():
            if key == 'text_description':
                print(f"     {key}: {value[:80]}...")
            else:
                print(f"     {key}: {value}")

print(f"\n   Total metadata records: {len(metadata)}")
print("\n" + "="*80)

