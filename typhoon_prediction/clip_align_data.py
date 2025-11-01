"""
CLIP Alignment Script for Typhoon Data
Processes fused Himawari-8 and ERA5 data organized by typhoon ID
Performs visual + text alignment using CLIP
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import h5py
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import json
import glob

# Try to import xarray/netCDF4 for NetCDF support
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    try:
        import netCDF4
        HAS_NETCDF4 = True
    except ImportError:
        HAS_NETCDF4 = False
        print("Warning: Neither xarray nor netCDF4 found. NetCDF files may not load correctly.")

# Add CLIP to path - use Desktop location
clip_path = os.path.join(os.path.dirname(__file__), 'CLIP')
if not os.path.exists(clip_path):
    # Try alternative path
    clip_path = r"C:\Users\fyp\Desktop\FYP\typhoon_prediction\CLIP"

if os.path.exists(clip_path):
    sys.path.insert(0, clip_path)
    print(f"CLIP path added to sys.path: {clip_path}")
else:
    raise FileNotFoundError(f"CLIP directory not found at: {clip_path}")

try:
    import clip
    from clip import load, tokenize
    print(f"CLIP imported successfully from: {clip.__file__}")
except ImportError:
    print("Warning: CLIP not found. Please ensure CLIP is installed.")
    raise


class TyphoonCLIPAligner:
    """
    CLIP aligner for typhoon data organized by typhoon ID
    Performs visual + text alignment
    """
    
    def __init__(self, device='cuda', clip_model='ViT-B/32', batch_size=32):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        
        # Verify GPU
        if self.device == 'cuda':
            print(f"Using device: {self.device}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print(f"Using device: {self.device} (CPU mode)")
        
        # Load CLIP model
        print(f"Loading CLIP model: {clip_model}")
        try:
            self.clip_model, self.preprocess = load(clip_model, device=self.device)
            self.clip_model.eval()
            self.visual_dim = self.clip_model.visual.output_dim
            self.text_dim = self.clip_model.transformer.width
            print(f"CLIP loaded: visual_dim={self.visual_dim}, text_dim={self.text_dim}")
            print(f"Batch size: {batch_size}")
            
            # Warm up GPU
            if self.device == 'cuda':
                print("Warming up GPU...")
                dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
                dummy_text = tokenize(["test"]).to(self.device)
                with torch.no_grad():
                    _ = self.clip_model.encode_image(dummy_image)
                    _ = self.clip_model.encode_text(dummy_text)
                torch.cuda.synchronize()
                print("GPU warmed up!")
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            raise
    
    def find_data_files_in_folder(self, folder_path):
        """Find all data files in a typhoon ID folder"""
        data_files = []
        
        if not os.path.exists(folder_path):
            return data_files
        
        # Search for common formats (NetCDF files are .nc)
        patterns = ['*.nc', '*.h5', '*.hdf5', '*.npy', '*.npz']
        
        for pattern in patterns:
            found = list(Path(folder_path).glob(pattern))
            data_files.extend(found)
        
        return sorted(list(set(data_files)))
    
    def load_image_data(self, file_path):
        """Load image/data file"""
        try:
            if file_path.suffix == '.nc':
                # NetCDF file
                if HAS_XARRAY:
                    try:
                        ds = xr.open_dataset(file_path)
                        # Try to find image-like variables - prefer 2D variables
                        best_var = None
                        best_var_dims = None
                        
                        for var_name in ds.data_vars:
                            var = ds[var_name]
                            dims = var.dims
                            shape = var.shape
                            
                            # Prefer 2D variables (lat, lon) or (y, x)
                            if len(shape) == 2:
                                if best_var is None or len(best_var_dims) > 2:
                                    best_var = var_name
                                    best_var_dims = dims
                            # Accept 3D if spatial dimensions are reasonable
                            elif len(shape) == 3:
                                # Check if it's (time, lat, lon) or (lat, lon, time)
                                if 'time' in dims or 'time' in str(dims).lower():
                                    # Take first time slice
                                    if best_var is None:
                                        best_var = var_name
                                        best_var_dims = dims
                                elif shape[2] < 100:  # Likely (H, W, C) with small C
                                    if best_var is None:
                                        best_var = var_name
                                        best_var_dims = dims
                            # 4D+ - skip or take first slice
                            elif len(shape) == 4:
                                if best_var is None:
                                    best_var = var_name
                                    best_var_dims = dims
                        
                        if best_var is not None:
                            data = ds[best_var].values
                            # Handle different dimensions
                            if len(data.shape) == 2:
                                ds.close()
                                return data  # Perfect: (H, W)
                            elif len(data.shape) == 3:
                                # Check if it's (time, H, W) or (H, W, C)
                                if data.shape[0] < 10:  # Likely time dimension
                                    data = data[0]  # Take first time slice -> (H, W)
                                elif data.shape[2] < 100:  # Likely (H, W, C)
                                    data = np.mean(data, axis=2)  # Average channels -> (H, W)
                                else:
                                    data = data[:, :, 0]  # Take first channel -> (H, W)
                                ds.close()
                                return data
                            elif len(data.shape) == 4:
                                # (time, lat, lon, level) or similar - take first slice
                                data = data[0, :, :, 0] if data.shape[0] < 100 else data[:, :, 0, 0]
                                ds.close()
                                return data
                        
                        # Fallback: try first variable and extract 2D slice
                        if len(ds.data_vars) > 0:
                            var_name = list(ds.data_vars.keys())[0]
                            data = ds[var_name].values
                            # Extract 2D slice
                            while len(data.shape) > 2:
                                if data.shape[0] < 10:
                                    data = data[0]
                                else:
                                    data = np.mean(data, axis=0) if len(data.shape) == 3 else data[:, :, 0]
                            ds.close()
                            return data
                        
                        ds.close()
                    except Exception as e:
                        print(f"Error with xarray loading {file_path.name}: {e}")
                elif HAS_NETCDF4:
                    try:
                        nc = netCDF4.Dataset(file_path, 'r')
                        # Try common variable names first
                        preferred_names = ['fused', 'data', 'image', 'temperature', 'precipitation']
                        data = None
                        
                        for var_name in preferred_names:
                            if var_name in nc.variables:
                                var = nc.variables[var_name]
                                if len(var.shape) >= 2:
                                    data = var[:]
                                    break
                        
                        # If not found, try first variable
                        if data is None:
                            for var_name in nc.variables:
                                var = nc.variables[var_name]
                                if len(var.shape) >= 2:
                                    data = var[:]
                                    break
                        
                        if data is not None:
                            # Extract 2D slice
                            while len(data.shape) > 2:
                                if data.shape[0] < 10:
                                    data = data[0]
                                else:
                                    data = np.mean(data, axis=0) if len(data.shape) == 3 else data[:, :, 0]
                            nc.close()
                            return data
                        
                        nc.close()
                    except Exception as e:
                        print(f"Error with netCDF4 loading {file_path.name}: {e}")
                else:
                    print(f"Cannot load NetCDF file: {file_path}. Install xarray or netCDF4.")
                    return None
            elif file_path.suffix in ['.h5', '.hdf5']:
                with h5py.File(file_path, 'r') as f:
                    # Try common keys
                    for key in ['data', 'image', 'features', 'array', 'fused']:
                        if key in f:
                            data = f[key][:]
                            # Handle different shapes
                            if len(data.shape) >= 2:
                                return data
                    # Get first dataset
                    if len(f.keys()) > 0:
                        key = list(f.keys())[0]
                        return f[key][:]
            elif file_path.suffix == '.npy':
                return np.load(file_path)
            elif file_path.suffix == '.npz':
                npz = np.load(file_path)
                # Return first array or 'data' if exists
                if 'data' in npz:
                    return npz['data']
                else:
                    return npz[list(npz.keys())[0]]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def process_image_for_clip(self, image_data):
        """Process image data for CLIP input"""
        if image_data is None:
            return None
        
        # Handle different array shapes - ensure we get a 2D spatial array
        if len(image_data.shape) == 2:
            # Grayscale -> RGB: shape (H, W) -> (H, W, 3)
            image_data = np.stack([image_data] * 3, axis=-1)
        elif len(image_data.shape) == 3:
            # Handle (H, W, C) or (C, H, W) formats
            h, w, c = image_data.shape[0], image_data.shape[1], image_data.shape[2]
            
            # Check if it's (H, W, C) format
            if h > 10 and w > 10 and c < 100:  # Likely (H, W, C)
                if c == 1:
                    image_data = np.repeat(image_data, 3, axis=2)
                elif c == 3:
                    pass  # Already RGB
                elif c > 3:
                    # Take first 3 channels or average
                    image_data = image_data[:, :, :3]
            # Check if it's (C, H, W) format
            elif c > 10 and h < 1000 and w < 1000:  # Likely (C, H, W)
                # Take first channel or average channels
                if image_data.shape[0] == 1:
                    image_data = np.repeat(image_data[0], 3, axis=0)
                    image_data = image_data.T  # Transpose to (H, W, 3)
                else:
                    # Average or take first 3 channels
                    image_data = np.mean(image_data[:min(3, image_data.shape[0])], axis=0)
                    if len(image_data.shape) == 2:
                        image_data = np.stack([image_data] * 3, axis=-1)
                    else:
                        image_data = np.stack([image_data] * 3, axis=-1) if len(image_data.shape) == 2 else image_data
        elif len(image_data.shape) == 4:
            # (B, H, W, C) or (B, C, H, W) - take first
            image_data = image_data[0]
            # Recursively process if still multi-dimensional
            if len(image_data.shape) == 3:
                return self.process_image_for_clip(image_data)
        
        # Ensure we have (H, W, 3) format
        if len(image_data.shape) != 3 or image_data.shape[2] != 3:
            # Force to grayscale then RGB
            if len(image_data.shape) == 2:
                image_data = np.stack([image_data] * 3, axis=-1)
            else:
                # Take first slice or average
                if len(image_data.shape) == 3:
                    if image_data.shape[0] < 100:  # Likely (C, H, W)
                        image_data = np.mean(image_data, axis=0)
                    else:  # Likely (H, W, C)
                        image_data = np.mean(image_data, axis=2)
                image_data = np.stack([image_data] * 3, axis=-1)
        
        # Normalize to [0, 1]
        if image_data.max() > 1.0:
            image_data = image_data / 255.0
        
        # Ensure values are in [0, 1]
        image_data = np.clip(image_data, 0, 1)
        
        # Resize to 224x224 for CLIP
        if image_data.shape[0] != 224 or image_data.shape[1] != 224:
            from PIL import Image as PILImage
            img = PILImage.fromarray((image_data * 255).astype(np.uint8))
            img = img.resize((224, 224))
            image_data = np.array(img) / 255.0
        
        # Verify final shape is (224, 224, 3)
        assert image_data.shape == (224, 224, 3), f"Expected shape (224, 224, 3), got {image_data.shape}"
        
        # Convert to tensor (keep on CPU for batching)
        image_data = (image_data * 255).astype(np.uint8)
        image = Image.fromarray(image_data)
        image_tensor = self.preprocess(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
        
        # Verify tensor shape
        assert image_tensor.shape == (1, 3, 224, 224), f"Expected tensor shape (1, 3, 224, 224), got {image_tensor.shape}"
        
        return image_tensor
    
    def create_text_description(self, row):
        """Create text description from CSV row"""
        parts = []
        
        if 'typhoon_name' in row and pd.notna(row['typhoon_name']):
            parts.append(f"typhoon {row['typhoon_name']}")
        if 'lat' in row and pd.notna(row['lat']):
            parts.append(f"latitude {row['lat']:.2f} degrees")
        if 'lon' in row and pd.notna(row['lon']):
            parts.append(f"longitude {row['lon']:.2f} degrees")
        if 'wind' in row and pd.notna(row['wind']):
            parts.append(f"wind speed {row['wind']:.0f} knots")
        if 'time' in row:
            parts.append(f"time {row['time']}")
        if 'year' in row:
            parts.append(f"year {row['year']}")
        
        return ", ".join(parts) if parts else "typhoon data"
    
    def align_typhoon_data(self, data_dir, csv_path, output_dir, max_typhoons=None):
        """
        Main alignment function
        Processes data organized by typhoon ID folders
        
        Args:
            data_dir: Directory containing typhoon ID folders
            csv_path: Path to CSV file with typhoon track data
            output_dir: Directory to save aligned features
            max_typhoons: Maximum number of typhoons to process (None = all, for test mode)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load CSV
        print(f"\nLoading CSV from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} CSV records")
        
        # Group by typhoon_id
        grouped = df.groupby('typhoon_id')
        total_typhoons = len(grouped)
        print(f"Found {total_typhoons} unique typhoon IDs")
        
        # Apply test mode limit if specified
        if max_typhoons is not None:
            print(f"\n[TEST MODE] Processing only first {max_typhoons} typhoons")
            typhoon_list = list(grouped)[:max_typhoons]
        else:
            typhoon_list = grouped
        
        aligned_features = []
        metadata = []
        # Organized by typhoon for LSTM sequences
        typhoon_sequences = {}  # {typhoon_id: [(feature, metadata), ...]}
        
        # Process each typhoon
        for typhoon_id, group_df in tqdm(typhoon_list, desc="Processing typhoons", total=min(max_typhoons or total_typhoons, total_typhoons)):
            typhoon_folder = os.path.join(data_dir, str(typhoon_id))
            
            if not os.path.exists(typhoon_folder):
                print(f"\nWarning: Folder not found for {typhoon_id}, skipping...")
                continue
            
            # Find data files in this typhoon's folder
            data_files = self.find_data_files_in_folder(typhoon_folder)
            
            if len(data_files) == 0:
                print(f"\nWarning: No data files found in {typhoon_folder}, skipping...")
                continue
            
            print(f"\nProcessing {typhoon_id}: {len(data_files)} files, {len(group_df)} CSV records")
            
            # Match data files with CSV entries by time
            # Extract time from filename (format: era5_merged_YYYYMMDD_YYYYMMDD_HHMM_fused.nc)
            def extract_time_from_filename(filename):
                """Extract time string from filename"""
                name = filename.stem
                # Look for pattern like 20180208_0600 or 20180208_20180208_0600
                parts = name.split('_')
                date_part = None
                time_part = None
                for part in parts:
                    if len(part) == 8 and part.isdigit():  # YYYYMMDD
                        date_part = part
                    elif len(part) == 4 and part.isdigit():  # HHMM
                        time_part = part
                if date_part and time_part:
                    return date_part + time_part  # YYYYMMDDHHMM
                return None
            
            # Match CSV time format (08020600 = MMDDHHMM -> 20180208060200)
            def csv_time_to_filename_time(csv_row):
                """Convert CSV time format to match filename time"""
                csv_time_str = str(int(csv_row.get('time', 0))) if pd.notna(csv_row.get('time')) else None
                year = csv_row.get('year')
                
                if csv_time_str and year and len(csv_time_str) == 8:  # MMDDHHMM
                    month = csv_time_str[:2]
                    day = csv_time_str[2:4]
                    hour = csv_time_str[4:6]
                    minute = csv_time_str[6:8]
                    # Create YYYYMMDDHHMM format
                    return f"{year}{month}{day}{hour}{minute}"
                return None
            
            # Create mapping of filename times to files
            file_time_map = {}
            for data_file in data_files:
                time_key = extract_time_from_filename(data_file)
                if time_key:
                    file_time_map[time_key] = data_file
            
            # Process CSV rows and match with files
            matched_pairs = []
            for idx, csv_row in group_df.iterrows():
                csv_time = csv_time_to_filename_time(csv_row)
                if csv_time:
                    # Try to find matching file
                    matching_file = None
                    # Try exact match first
                    if csv_time in file_time_map:
                        matching_file = file_time_map[csv_time]
                    else:
                        # Try partial match (just date and hour, allowing minute mismatch)
                        partial_time = csv_time[:10]  # YYYYMMDDHH
                        for time_key, file_path in file_time_map.items():
                            if time_key.startswith(partial_time):
                                matching_file = file_path
                                break
                    
                    if matching_file:
                        matched_pairs.append((csv_row, matching_file))
            
            # If no matches found, fall back to sequential matching
            if len(matched_pairs) == 0:
                num_to_process = min(len(data_files), len(group_df))
                for i in range(num_to_process):
                    matched_pairs.append((group_df.iloc[i % len(group_df)], data_files[i % len(data_files)]))
            
            # Process matched pairs in batches for efficiency
            batch_images = []
            batch_texts = []
            batch_metadata = []
            
            for csv_row, data_file in matched_pairs:
                try:
                    # Load image/data
                    image_data = self.load_image_data(data_file)
                    if image_data is None:
                        continue
                    
                    # Process image for CLIP
                    image_tensor = self.process_image_for_clip(image_data)
                    if image_tensor is None:
                        continue
                    
                    # Create text description
                    text_desc = self.create_text_description(csv_row)
                    
                    # Store for batch processing
                    batch_images.append(image_tensor)
                    batch_texts.append(text_desc)
                    batch_metadata.append({
                        'typhoon_id': str(typhoon_id),
                        'typhoon_name': str(csv_row.get('typhoon_name', '')),
                        'time': str(csv_row.get('time', '')),
                        'year': int(csv_row.get('year', 0)) if pd.notna(csv_row.get('year')) else 0,
                        'lat': float(csv_row.get('lat', 0)) if pd.notna(csv_row.get('lat')) else 0.0,
                        'lon': float(csv_row.get('lon', 0)) if pd.notna(csv_row.get('lon')) else 0.0,
                        'wind': float(csv_row.get('wind', 0)) if pd.notna(csv_row.get('wind')) else 0.0,
                        'data_file': str(data_file.name),
                        'data_file_path': str(data_file),
                        'text_description': text_desc
                    })
                    
                    # Process batch when full
                    if len(batch_images) >= self.batch_size:
                        # Batch encode images - concatenate along batch dimension
                        # Each img is (1, 3, 224, 224), we want (batch_size, 3, 224, 224)
                        image_batch = torch.cat(batch_images, dim=0).to(self.device)  # (batch_size, 3, 224, 224)
                        
                        text_batch_tokens = tokenize(batch_texts).to(self.device)
                        
                        with torch.no_grad():
                            visual_feats = self.clip_model.encode_image(image_batch)
                            text_feats = self.clip_model.encode_text(text_batch_tokens)
                        
                        # Normalize and align
                        visual_feats = visual_feats / visual_feats.norm(dim=-1, keepdim=True)
                        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                        aligned_feats = (visual_feats + text_feats) / 2
                        aligned_feats = aligned_feats / aligned_feats.norm(dim=-1, keepdim=True)
                        
                        # Move to CPU and store
                        aligned_feats_np = aligned_feats.cpu().numpy()
                        aligned_features.extend(aligned_feats_np)
                        metadata.extend(batch_metadata)
                        
                        # Store by typhoon for sequential organization
                        for feat, meta in zip(aligned_feats_np, batch_metadata):
                            tid = meta['typhoon_id']
                            if tid not in typhoon_sequences:
                                typhoon_sequences[tid] = []
                            typhoon_sequences[tid].append((feat, meta))
                        
                        # Clear batch and GPU cache
                        del image_batch, text_batch_tokens, visual_feats, text_feats, aligned_feats
                        batch_images = []
                        batch_texts = []
                        batch_metadata = []
                        
                        # Optional: Clear GPU cache periodically
                        if self.device == 'cuda' and len(aligned_features) % (self.batch_size * 10) == 0:
                            torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"\nError processing {data_file.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Process remaining items in batch
            if len(batch_images) > 0:
                # Concatenate along batch dimension
                image_batch = torch.cat(batch_images, dim=0).to(self.device)  # (N, 3, 224, 224)
                
                # Verify batch shape
                if image_batch.shape[1] != 3:
                    print(f"ERROR: Batch shape is {image_batch.shape}, expected (N, 3, 224, 224)")
                    raise ValueError(f"Invalid batch shape: {image_batch.shape}")
                
                text_batch_tokens = tokenize(batch_texts).to(self.device)
                
                with torch.no_grad():
                    visual_feats = self.clip_model.encode_image(image_batch)
                    text_feats = self.clip_model.encode_text(text_batch_tokens)
                
                visual_feats = visual_feats / visual_feats.norm(dim=-1, keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                aligned_feats = (visual_feats + text_feats) / 2
                aligned_feats = aligned_feats / aligned_feats.norm(dim=-1, keepdim=True)
                
                aligned_feats_np = aligned_feats.cpu().numpy()
                aligned_features.extend(aligned_feats_np)
                metadata.extend(batch_metadata)
                
                # Store by typhoon for sequential organization
                for feat, meta in zip(aligned_feats_np, batch_metadata):
                    tid = meta['typhoon_id']
                    if tid not in typhoon_sequences:
                        typhoon_sequences[tid] = []
                    typhoon_sequences[tid].append((feat, meta))
        
        # Save results
        print(f"\n{'='*80}")
        print(f"Saving {len(aligned_features)} aligned features to {output_dir}")
        print(f"{'='*80}")
        
        if aligned_features:
            # Save as numpy
            features_array = np.vstack(aligned_features)
            np.save(os.path.join(output_dir, 'aligned_features.npy'), features_array)
            print(f"[OK] Saved aligned_features.npy: shape {features_array.shape}")
            
            # Save metadata
            with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"[OK] Saved metadata.json: {len(metadata)} records")
            
            # Save as HDF5
            with h5py.File(os.path.join(output_dir, 'aligned_features.h5'), 'w') as f:
                f.create_dataset('features', data=features_array, compression='gzip')
                
                if metadata:
                    f.create_dataset('typhoon_ids',
                                   data=[m['typhoon_id'].encode('utf-8') for m in metadata],
                                   dtype=h5py.string_dtype())
                    f.create_dataset('typhoon_names',
                                   data=[m['typhoon_name'].encode('utf-8') for m in metadata],
                                   dtype=h5py.string_dtype())
                    f.create_dataset('times',
                                   data=[m['time'].encode('utf-8') for m in metadata],
                                   dtype=h5py.string_dtype())
                    f.create_dataset('years', data=[m['year'] for m in metadata])
                    f.create_dataset('lats', data=[m['lat'] for m in metadata])
                    f.create_dataset('lons', data=[m['lon'] for m in metadata])
                    f.create_dataset('winds', data=[m['wind'] for m in metadata])
            
            print(f"[OK] Saved aligned_features.h5")
            
            # Save organized by typhoon ID for LSTM sequences
            print(f"\nSaving organized by typhoon ID...")
            organized_dir = os.path.join(output_dir, 'organized_by_typhoon')
            os.makedirs(organized_dir, exist_ok=True)
            
            for typhoon_id, sequence in typhoon_sequences.items():
                typhoon_folder = os.path.join(organized_dir, str(typhoon_id))
                os.makedirs(typhoon_folder, exist_ok=True)
                
                # Sort by time to ensure sequential order
                sequence.sort(key=lambda x: (
                    x[1].get('year', 0),
                    float(x[1].get('time', '0').replace('.0', '')) if isinstance(x[1].get('time'), str) else float(x[1].get('time', 0))
                ))
                
                # Save features as numpy array (sequence)
                features_seq = np.array([feat for feat, _ in sequence])
                np.save(os.path.join(typhoon_folder, 'features.npy'), features_seq)
                
                # Save metadata as JSON
                metadata_seq = [meta for _, meta in sequence]
                with open(os.path.join(typhoon_folder, 'metadata.json'), 'w', encoding='utf-8') as f:
                    json.dump(metadata_seq, f, indent=2, ensure_ascii=False)
                
                # Save individual timestep files (optional, for easy access)
                timesteps_dir = os.path.join(typhoon_folder, 'timesteps')
                os.makedirs(timesteps_dir, exist_ok=True)
                
                for idx, (feat, meta) in enumerate(sequence):
                    timestep_file = os.path.join(timesteps_dir, f'timestep_{idx:06d}.npz')
                    np.savez_compressed(
                        timestep_file,
                        feature=feat,
                        time=meta.get('time', ''),
                        lat=meta.get('lat', 0.0),
                        lon=meta.get('lon', 0.0),
                        wind=meta.get('wind', 0.0),
                        year=meta.get('year', 0)
                    )
                
                print(f"  Saved {typhoon_id}: {len(sequence)} timesteps")
            
            print(f"[OK] Saved organized data to {organized_dir}")
            
            # Save summary
            summary = {
                'total_samples': len(aligned_features),
                'feature_dim': features_array.shape[1],
                'unique_typhoons': len(set(m['typhoon_id'] for m in metadata)),
                'output_dir': output_dir,
                'device': str(self.device),
                'clip_model': 'ViT-B/32',
                'alignment_type': 'visual + text',
                'test_mode': max_typhoons is not None,
                'max_typhoons_processed': max_typhoons if max_typhoons is not None else 'all'
            }
            
            with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n{'='*80}")
            print("CLIP Alignment Complete!")
            print(f"{'='*80}")
            print(f"Processed {len(aligned_features)} samples")
            print(f"Feature dimension: {features_array.shape[1]}")
            print(f"Unique typhoons: {summary['unique_typhoons']}")
            print(f"Output directory: {output_dir}")
            print(f"{'='*80}")
        else:
            print("No features were processed!")


def main():
    """Main function"""
    # Configuration
    data_dir = r"D:\typhoon_interpolated"
    csv_path = r"D:\typhoon_interpolated\interpolated_typhoon_tracks.csv"
    output_dir = r"D:\typhoon_aligned"
    
    # TEST MODE: Set to a number (e.g., 3) to process only first N typhoons, None to process all
    TEST_MODE = True  # Set to True to enable test mode
    MAX_TYPHOONS_TEST = 3  # Number of typhoons to process in test mode
    
    # Determine if test mode is active
    max_typhoons = MAX_TYPHOONS_TEST if TEST_MODE else None
    
    print("="*80)
    if TEST_MODE:
        print("CLIP Alignment for Typhoon Data (Visual + Text) - TEST MODE")
        print(f"Will process only first {MAX_TYPHOONS_TEST} typhoons")
    else:
        print("CLIP Alignment for Typhoon Data (Visual + Text)")
    print("="*80)
    print(f"CLIP location: {clip_path}")
    print(f"Input data directory: {data_dir}")
    print(f"  (organized by typhoon ID folders, e.g., {data_dir}\\2018039N08151)")
    print(f"CSV tracks file: {csv_path}")
    print(f"Output directory: {output_dir}")
    if TEST_MODE:
        print(f"Test mode: Processing {MAX_TYPHOONS_TEST} typhoons")
    print("="*80)
    
    # Verify paths exist
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    print(f"[OK] Data directory exists: {data_dir}")
    print(f"[OK] CSV file exists: {csv_path}")
    print(f"[OK] Output directory will be created: {output_dir}")
    print("="*80)
    
    # Create aligner with batch processing
    # Use larger batch size for GPU (32-64), smaller for CPU (8-16)
    use_cuda = torch.cuda.is_available()
    batch_size = 64 if use_cuda else 16
    
    aligner = TyphoonCLIPAligner(
        device='cuda' if use_cuda else 'cpu',
        batch_size=batch_size
    )
    
    # Run alignment
    aligner.align_typhoon_data(data_dir, csv_path, output_dir, max_typhoons=max_typhoons)


if __name__ == '__main__':
    main()
