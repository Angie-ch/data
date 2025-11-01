"""
Visualize timestamp-aligned typhoon sequences
Shows how features, metadata, and targets align across timesteps
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from data_loaders.lstm_loader import TyphoonSequenceDataset, create_dataloaders
import torch

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def visualize_typhoon_sequence(typhoon_folder, output_dir):
    """Visualize a single typhoon sequence"""
    print(f"\nVisualizing {typhoon_folder.name}...")
    
    # Load features
    features_file = typhoon_folder / 'features.npy'
    metadata_file = typhoon_folder / 'metadata.json'
    
    features = np.load(features_file)  # (T, 512)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    times = [float(m.get('time', '0').replace('.0', '')) for m in metadata]
    lats = [m.get('lat', 0) for m in metadata]
    lons = [m.get('lon', 0) for m in metadata]
    winds = [m.get('wind', 0) for m in metadata]
    years = [m.get('year', 0) for m in metadata]
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    typhoon_id = typhoon_folder.name
    typhoon_name = metadata[0].get('typhoon_name', 'Unknown')
    fig.suptitle(f'Typhoon Sequence: {typhoon_name} ({typhoon_id})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Feature evolution over time (PCA)
    ax = fig.add_subplot(gs[0, :2])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features.astype(np.float32))
    
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                        c=range(len(features)), cmap='viridis', 
                        s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('Feature Space Evolution (PCA)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Timestep')
    
    # 2. Feature statistics over time
    ax = fig.add_subplot(gs[0, 2])
    feature_means = features.mean(axis=1)
    feature_stds = features.std(axis=1)
    timesteps = np.arange(len(features))
    ax.plot(timesteps, feature_means, label='Mean', linewidth=2)
    ax.fill_between(timesteps, feature_means - feature_stds, 
                    feature_means + feature_stds, alpha=0.3, label='±Std')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Feature Value')
    ax.set_title('Feature Statistics Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Wind speed over time
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(range(len(winds)), winds, marker='o', markersize=3, 
           linewidth=2, color='red', alpha=0.7)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Wind Speed (knots)')
    ax.set_title('Wind Speed Evolution')
    ax.grid(True, alpha=0.3)
    
    # 4. Location track
    ax = fig.add_subplot(gs[1, 1])
    scatter = ax.scatter(lons, lats, c=range(len(lats)), cmap='coolwarm',
                        s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.plot(lons, lats, 'k--', alpha=0.3, linewidth=1)
    ax.plot(lons[0], lats[0], 'gs', markersize=15, label='Start', markeredgecolor='black')
    ax.plot(lons[-1], lats[-1], 'r*', markersize=20, label='End', markeredgecolor='black')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Location Track')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Timestep')
    
    # 5. Feature heatmap (sample dimensions)
    ax = fig.add_subplot(gs[1, 2])
    # Show first 50 feature dimensions
    sample_features = features[:, :50].T
    im = ax.imshow(sample_features, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Feature Dimension')
    ax.set_title('Feature Heatmap (First 50 dims)')
    plt.colorbar(im, ax=ax, label='Feature Value')
    
    # 6. Correlation between features and wind
    ax = fig.add_subplot(gs[2, 0])
    # Find most correlated features with wind
    correlations = []
    for i in range(min(100, features.shape[1])):  # Check first 100 features
        corr = np.corrcoef(features[:, i], winds)[0, 1]
        correlations.append(corr)
    
    top_indices = np.argsort(np.abs(correlations))[-20:][::-1]
    top_corrs = [correlations[i] for i in top_indices]
    
    ax.barh(range(len(top_corrs)), top_corrs, color='steelblue')
    ax.set_yticks(range(len(top_corrs)))
    ax.set_yticklabels([f'Dim {i}' for i in top_indices])
    ax.set_xlabel('Correlation with Wind Speed')
    ax.set_title('Top 20 Features Correlated with Wind')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 7. Timestep comparison (first, middle, last)
    ax = fig.add_subplot(gs[2, 1])
    timesteps_to_show = [0, len(features)//2, len(features)-1]
    for i, t in enumerate(timesteps_to_show):
        ax.plot(features[t, :100], label=f'Timestep {t}', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Feature Value')
    ax.set_title('Feature Comparison (First 100 dims)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Sequence statistics
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    stats_text = f"""
    Sequence Statistics:
    • Total Timesteps: {len(features)}
    • Feature Dimension: {features.shape[1]}
    • Start Time: {metadata[0].get('time', 'N/A')}
    • End Time: {metadata[-1].get('time', 'N/A')}
    • Year: {years[0]}
    • Wind Range: {min(winds):.1f} - {max(winds):.1f} knots
    • Lat Range: {min(lats):.2f} - {max(lats):.2f}°
    • Lon Range: {min(lons):.2f} - {max(lons):.2f}°
    """
    ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 9. Feature magnitude over time
    ax = fig.add_subplot(gs[3, 0])
    feature_norms = np.linalg.norm(features, axis=1)
    ax.plot(range(len(feature_norms)), feature_norms, linewidth=2, color='purple')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Feature Vector Norm')
    ax.set_title('Feature Magnitude Over Time')
    ax.grid(True, alpha=0.3)
    
    # 10. Wind vs Location
    ax = fig.add_subplot(gs[3, 1])
    scatter = ax.scatter(lons, lats, c=winds, cmap='YlOrRd',
                        s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.plot(lons, lats, 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Location Track (colored by wind speed)')
    plt.colorbar(scatter, ax=ax, label='Wind Speed (knots)')
    ax.grid(True, alpha=0.3)
    
    # 11. Time series of all variables
    ax = fig.add_subplot(gs[3, 2])
    ax2 = ax.twinx()
    
    line1 = ax.plot(range(len(winds)), winds, 'r-', label='Wind Speed', linewidth=2)
    line2 = ax2.plot(range(len(lats)), lats, 'b-', label='Latitude', linewidth=2, alpha=0.7)
    line3 = ax2.plot(range(len(lons)), np.array(lons) - np.array(lons).mean(), 
                     'g-', label='Lon (centered)', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Wind Speed (knots)', color='r')
    ax2.set_ylabel('Latitude / Longitude', color='b')
    ax.set_title('Multi-Variable Time Series')
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='b')
    
    plt.savefig(output_dir / f'{typhoon_id}_sequence_visualization.png', 
                dpi=300, bbox_inches='tight')
    print(f"  Saved: {typhoon_id}_sequence_visualization.png")
    plt.close()

def visualize_lstm_sequences(data_dir, output_dir):
    """Visualize how LSTM sequences are created"""
    print("\nVisualizing LSTM sequence creation...")
    
    # Create dataset
    dataset = TyphoonSequenceDataset(
        data_dir=data_dir,
        sequence_length=10,
        prediction_horizon=1,
        target_type='wind',
        mode='train'
    )
    
    # Get some samples
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('LSTM Sequence Visualization', fontsize=16, fontweight='bold')
    
    for plot_idx in range(4):
        if plot_idx >= len(dataset):
            break
        
        ax = axes[plot_idx // 2, plot_idx % 2]
        
        sequence, target, metadata = dataset[plot_idx]
        sequence_np = sequence.numpy()
        target_np = target.numpy()
        
        # Plot sequence features (mean over dimensions)
        seq_mean = sequence_np.mean(axis=1)
        ax.plot(range(len(seq_mean)), seq_mean, 'b-', linewidth=2, label='Sequence')
        
        # Mark prediction point
        ax.axvline(len(seq_mean) - 1, color='r', linestyle='--', linewidth=2, label='Last input')
        ax.axvline(len(seq_mean), color='g', linestyle='--', linewidth=2, label='Prediction')
        
        # Add target value
        ax.plot(len(seq_mean), target_np[0], 'go', markersize=10, label='Target')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Mean Feature Value')
        ax.set_title(f'Sequence {plot_idx}\nTyphoon: {metadata["typhoon_id"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lstm_sequences.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: lstm_sequences.png")
    plt.close()

def visualize_all_typhoons_overview(data_dir, output_dir):
    """Create overview of all typhoons"""
    print("\nCreating overview visualization...")
    
    organized_dir = Path(data_dir)
    typhoon_folders = sorted([f for f in organized_dir.iterdir() if f.is_dir()])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('All Typhoons Overview', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(typhoon_folders)))
    
    # 1. Sequence lengths
    ax = axes[0, 0]
    lengths = []
    names = []
    for folder in typhoon_folders:
        features_file = folder / 'features.npy'
        if features_file.exists():
            features = np.load(features_file)
            lengths.append(len(features))
            
            metadata_file = folder / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    meta = json.load(f)
                    names.append(meta[0].get('typhoon_name', folder.name))
            else:
                names.append(folder.name)
    
    bars = ax.bar(range(len(lengths)), lengths, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Sequence Length')
    ax.set_title('Sequence Lengths by Typhoon')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Wind speed ranges
    ax = axes[0, 1]
    wind_mins = []
    wind_maxs = []
    wind_means = []
    for folder in typhoon_folders:
        metadata_file = folder / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                meta = json.load(f)
                winds = [m.get('wind', 0) for m in meta]
                wind_mins.append(min(winds))
                wind_maxs.append(max(winds))
                wind_means.append(np.mean(winds))
    
    x_pos = np.arange(len(names))
    width = 0.35
    ax.bar(x_pos - width/2, wind_mins, width, label='Min', color='lightblue')
    ax.bar(x_pos, wind_means, width, label='Mean', color='steelblue')
    ax.bar(x_pos + width/2, wind_maxs, width, label='Max', color='darkblue')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Wind Speed (knots)')
    ax.set_title('Wind Speed Statistics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Location tracks (all on one map)
    ax = axes[1, 0]
    for i, folder in enumerate(typhoon_folders):
        metadata_file = folder / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                meta = json.load(f)
                lats = [m.get('lat', 0) for m in meta]
                lons = [m.get('lon', 0) for m in meta]
                name = meta[0].get('typhoon_name', folder.name)
                ax.plot(lons, lats, marker='o', markersize=2, linewidth=1.5,
                       label=name, color=colors[i], alpha=0.7)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('All Typhoon Tracks')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Feature statistics comparison
    ax = axes[1, 1]
    feature_means_all = []
    for folder in typhoon_folders:
        features_file = folder / 'features.npy'
        if features_file.exists():
            features = np.load(features_file)
            feature_means_all.append(features.mean(axis=1).mean())  # Mean over all timesteps
    
    ax.bar(range(len(feature_means_all)), feature_means_all, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Mean Feature Value')
    ax.set_title('Average Feature Values')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_typhoons_overview.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: all_typhoons_overview.png")
    plt.close()

def main():
    """Main function"""
    data_dir = r"D:\typhoon_aligned\organized_by_typhoon"
    output_dir = Path(r"D:\typhoon_aligned\visualizations\sequences")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Timestamp-Aligned Sequence Visualization")
    print("="*80)
    
    organized_dir = Path(data_dir)
    
    # Visualize each typhoon sequence
    typhoon_folders = sorted([f for f in organized_dir.iterdir() if f.is_dir()])
    for folder in typhoon_folders:
        visualize_typhoon_sequence(folder, output_dir)
    
    # Visualize LSTM sequences
    visualize_lstm_sequences(data_dir, output_dir)
    
    # Overview of all typhoons
    visualize_all_typhoons_overview(data_dir, output_dir)
    
    print("\n" + "="*80)
    print("Visualization Complete!")
    print(f"All plots saved to: {output_dir}")
    print("="*80)
    print("\nGenerated files:")
    print("  - {typhoon_id}_sequence_visualization.png (for each typhoon)")
    print("  - lstm_sequences.png")
    print("  - all_typhoons_overview.png")
    print("="*80)

if __name__ == '__main__':
    main()

