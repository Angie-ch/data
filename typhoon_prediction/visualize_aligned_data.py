"""
Visualization script for aligned typhoon CLIP features
Creates comprehensive visualizations of the aligned data
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import h5py
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(data_dir):
    """Load aligned features and metadata"""
    print("Loading data...")
    
    # Load HDF5 file
    h5_path = Path(data_dir) / 'aligned_features.h5'
    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]
        typhoon_ids = [x.decode() for x in f['typhoon_ids'][:]]
        typhoon_names = [x.decode() for x in f['typhoon_names'][:]]
        times = [x.decode() for x in f['times'][:]]
        years = f['years'][:]
        lats = f['lats'][:]
        lons = f['lons'][:]
        winds = f['winds'][:]
    
    # Load summary
    summary_path = Path(data_dir) / 'summary.json'
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"Loaded {len(features)} samples")
    print(f"Feature shape: {features.shape}")
    print(f"Unique typhoons: {summary['unique_typhoons']}")
    
    return {
        'features': features,
        'typhoon_ids': np.array(typhoon_ids),
        'typhoon_names': np.array(typhoon_names),
        'times': np.array(times),
        'years': years,
        'lats': lats,
        'lons': lons,
        'winds': winds,
        'summary': summary
    }

def plot_feature_statistics(data, output_dir):
    """Plot feature statistics"""
    print("\nCreating feature statistics plots...")
    
    features = data['features']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CLIP Aligned Feature Statistics', fontsize=16, fontweight='bold')
    
    # Feature value distribution
    ax = axes[0, 0]
    flat_features = features.flatten()
    ax.hist(flat_features, bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Feature Value Distribution')
    ax.axvline(flat_features.mean(), color='red', linestyle='--', 
               label=f'Mean: {flat_features.mean():.4f}')
    ax.legend()
    
    # Per-sample feature statistics
    ax = axes[0, 1]
    sample_means = features.mean(axis=1)
    sample_stds = features.std(axis=1)
    ax.scatter(sample_means, sample_stds, alpha=0.5, s=20)
    ax.set_xlabel('Mean Feature Value per Sample')
    ax.set_ylabel('Std Feature Value per Sample')
    ax.set_title('Sample-level Feature Statistics')
    ax.grid(True, alpha=0.3)
    
    # Feature dimension statistics
    ax = axes[1, 0]
    dim_means = features.mean(axis=0)
    dim_stds = features.std(axis=0)
    ax.plot(dim_means, label='Mean', alpha=0.7)
    ax.plot(dim_stds, label='Std', alpha=0.7)
    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Value')
    ax.set_title('Feature Dimension Statistics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Feature magnitude distribution
    ax = axes[1, 1]
    feature_norms = np.linalg.norm(features, axis=1)
    ax.hist(feature_norms, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Feature Vector Norm')
    ax.set_ylabel('Frequency')
    ax.set_title('Feature Vector Magnitude Distribution')
    ax.axvline(feature_norms.mean(), color='red', linestyle='--',
               label=f'Mean: {feature_norms.mean():.4f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'feature_statistics.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: feature_statistics.png")
    plt.close()

def plot_typhoon_tracks(data, output_dir):
    """Plot typhoon tracks on map"""
    print("\nCreating typhoon track plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Typhoon Tracks Visualization', fontsize=16, fontweight='bold')
    
    unique_typhoons = np.unique(data['typhoon_ids'])
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_typhoons)))
    
    # Track map
    ax = axes[0, 0]
    for i, typhoon_id in enumerate(unique_typhoons):
        mask = data['typhoon_ids'] == typhoon_id
        name = data['typhoon_names'][mask][0]
        ax.plot(data['lons'][mask], data['lats'][mask], 
               marker='o', markersize=3, linewidth=1.5, 
               label=f"{name} ({typhoon_id})", color=colors[i], alpha=0.7)
        # Mark start
        ax.plot(data['lons'][mask][0], data['lats'][mask][0], 
               marker='s', markersize=10, color=colors[i], markeredgecolor='black')
        # Mark end
        ax.plot(data['lons'][mask][-1], data['lats'][mask][-1], 
               marker='*', markersize=15, color=colors[i], markeredgecolor='black')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Typhoon Tracks (Latitude vs Longitude)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Wind speed over time (by typhoon)
    ax = axes[0, 1]
    for i, typhoon_id in enumerate(unique_typhoons):
        mask = data['typhoon_ids'] == typhoon_id
        name = data['typhoon_names'][mask][0]
        # Use index as time proxy
        indices = np.arange(len(data['winds'][mask]))
        ax.plot(indices, data['winds'][mask], 
               marker='o', markersize=2, linewidth=1.5,
               label=f"{name}", color=colors[i], alpha=0.7)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Wind Speed (knots)')
    ax.set_title('Wind Speed Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Location distribution
    ax = axes[1, 0]
    scatter = ax.scatter(data['lons'], data['lats'], 
                        c=data['winds'], s=20, alpha=0.6, 
                        cmap='viridis', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Location Distribution (colored by wind speed)')
    plt.colorbar(scatter, ax=ax, label='Wind Speed (knots)')
    ax.grid(True, alpha=0.3)
    
    # Wind speed histogram
    ax = axes[1, 1]
    for i, typhoon_id in enumerate(unique_typhoons):
        mask = data['typhoon_ids'] == typhoon_id
        name = data['typhoon_names'][mask][0]
        ax.hist(data['winds'][mask], bins=30, alpha=0.6, 
               label=f"{name}", color=colors[i], edgecolor='black')
    
    ax.set_xlabel('Wind Speed (knots)')
    ax.set_ylabel('Frequency')
    ax.set_title('Wind Speed Distribution by Typhoon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'typhoon_tracks.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: typhoon_tracks.png")
    plt.close()

def plot_feature_reduction(data, output_dir):
    """Plot PCA and t-SNE reductions"""
    print("\nCreating feature space visualizations...")
    
    features = data['features'].astype(np.float32)  # Convert to float32 for sklearn
    unique_typhoons = np.unique(data['typhoon_ids'])
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_typhoons)))
    
    # PCA
    print("  Computing PCA...")
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    explained_var = pca.explained_variance_ratio_.sum()
    
    # t-SNE (sample subset if too large)
    print("  Computing t-SNE (this may take a while)...")
    n_samples = len(features)
    if n_samples > 1000:
        print(f"  Sampling {min(1000, n_samples)} points for t-SNE...")
        indices = np.random.choice(n_samples, min(1000, n_samples), replace=False)
        features_subset = features[indices]
        typhoon_ids_subset = data['typhoon_ids'][indices]
    else:
        features_subset = features
        typhoon_ids_subset = data['typhoon_ids']
        indices = np.arange(len(features))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_tsne = tsne.fit_transform(features_subset)
    
    # Plot PCA
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax = axes[0]
    for i, typhoon_id in enumerate(unique_typhoons):
        mask = data['typhoon_ids'] == typhoon_id
        name = data['typhoon_names'][mask][0]
        ax.scatter(features_pca[mask, 0], features_pca[mask, 1],
                  c=[colors[i]], label=f"{name}", alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title(f'PCA Visualization (Total Explained Variance: {explained_var:.2%})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot t-SNE
    ax = axes[1]
    for i, typhoon_id in enumerate(unique_typhoons):
        mask_subset = typhoon_ids_subset == typhoon_id
        if mask_subset.sum() > 0:
            name = data['typhoon_names'][data['typhoon_ids'] == typhoon_id][0]
            ax.scatter(features_tsne[mask_subset, 0], features_tsne[mask_subset, 1],
                      c=[colors[i]], label=f"{name}", alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE Visualization (2D embedding)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'feature_reduction.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: feature_reduction.png")
    plt.close()

def plot_typhoon_comparison(data, output_dir):
    """Compare typhoons across different dimensions"""
    print("\nCreating typhoon comparison plots...")
    
    unique_typhoons = np.unique(data['typhoon_ids'])
    n_typhoons = len(unique_typhoons)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Typhoon Comparison Analysis', fontsize=16, fontweight='bold')
    
    # Feature similarity matrix
    ax = axes[0, 0]
    feature_means = []
    typhoon_labels = []
    for typhoon_id in unique_typhoons:
        mask = data['typhoon_ids'] == typhoon_id
        name = data['typhoon_names'][mask][0]
        feature_means.append(data['features'][mask].mean(axis=0))
        typhoon_labels.append(f"{name}\n({typhoon_id})")
    
    feature_means = np.array(feature_means)
    similarity_matrix = np.corrcoef(feature_means)
    
    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(n_typhoons))
    ax.set_yticks(range(n_typhoons))
    ax.set_xticklabels(typhoon_labels, rotation=45, ha='right')
    ax.set_yticklabels(typhoon_labels)
    ax.set_title('Feature Similarity Matrix (Correlation)')
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Sample counts
    ax = axes[0, 1]
    sample_counts = [np.sum(data['typhoon_ids'] == tid) for tid in unique_typhoons]
    names = [data['typhoon_names'][data['typhoon_ids'] == tid][0] for tid in unique_typhoons]
    bars = ax.bar(range(n_typhoons), sample_counts, color=plt.cm.tab10(np.linspace(0, 1, n_typhoons)))
    ax.set_xticks(range(n_typhoons))
    ax.set_xticklabels([f"{n}\n({tid})" for n, tid in zip(names, unique_typhoons)], rotation=45, ha='right')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Count per Typhoon')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Wind speed comparison
    ax = axes[1, 0]
    wind_data = [data['winds'][data['typhoon_ids'] == tid] for tid in unique_typhoons]
    bp = ax.boxplot(wind_data, labels=[data['typhoon_names'][data['typhoon_ids'] == tid][0] for tid in unique_typhoons],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.tab10(np.linspace(0, 1, n_typhoons))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Wind Speed (knots)')
    ax.set_title('Wind Speed Distribution Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Feature distance matrix
    ax = axes[1, 1]
    distance_matrix = np.zeros((n_typhoons, n_typhoons))
    for i, tid1 in enumerate(unique_typhoons):
        for j, tid2 in enumerate(unique_typhoons):
            mask1 = data['typhoon_ids'] == tid1
            mask2 = data['typhoon_ids'] == tid2
            mean1 = data['features'][mask1].mean(axis=0)
            mean2 = data['features'][mask2].mean(axis=0)
            # Cosine distance
            distance = 1 - np.dot(mean1, mean2) / (np.linalg.norm(mean1) * np.linalg.norm(mean2))
            distance_matrix[i, j] = distance
    
    im = ax.imshow(distance_matrix, cmap='viridis_r', aspect='auto')
    ax.set_xticks(range(n_typhoons))
    ax.set_yticks(range(n_typhoons))
    ax.set_xticklabels(typhoon_labels, rotation=45, ha='right')
    ax.set_yticklabels(typhoon_labels)
    ax.set_title('Feature Distance Matrix (Cosine Distance)')
    plt.colorbar(im, ax=ax, label='Distance')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'typhoon_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: typhoon_comparison.png")
    plt.close()

def plot_summary_dashboard(data, output_dir):
    """Create a summary dashboard"""
    print("\nCreating summary dashboard...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Typhoon CLIP Alignment - Summary Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    summary = data['summary']
    unique_typhoons = np.unique(data['typhoon_ids'])
    
    # Summary text
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    summary_text = f"""
    Dataset Summary:
    • Total Samples: {summary['total_samples']:,}
    • Feature Dimension: {summary['feature_dim']}
    • Unique Typhoons: {summary['unique_typhoons']}
    • Model: {summary['clip_model']}
    • Alignment: {summary['alignment_type']}
    • Device: {summary['device']}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Sample distribution pie
    ax = fig.add_subplot(gs[0, 1])
    sample_counts = [np.sum(data['typhoon_ids'] == tid) for tid in unique_typhoons]
    names = [data['typhoon_names'][data['typhoon_ids'] == tid][0] for tid in unique_typhoons]
    colors_pie = plt.cm.tab10(np.linspace(0, 1, len(unique_typhoons)))
    ax.pie(sample_counts, labels=names, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax.set_title('Sample Distribution')
    
    # Feature stats
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    features = data['features']
    stats_text = f"""
    Feature Statistics:
    • Mean: {features.mean():.4f}
    • Std: {features.std():.4f}
    • Min: {features.min():.4f}
    • Max: {features.max():.4f}
    • Mean Norm: {np.linalg.norm(features, axis=1).mean():.4f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Wind speed over time (all typhoons)
    ax = fig.add_subplot(gs[1, :])
    for i, typhoon_id in enumerate(unique_typhoons):
        mask = data['typhoon_ids'] == typhoon_id
        name = data['typhoon_names'][mask][0]
        indices = np.arange(len(data['winds'][mask]))
        ax.plot(indices, data['winds'][mask], marker='o', markersize=2, 
               linewidth=1.5, label=f"{name}", alpha=0.7, color=colors_pie[i])
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Wind Speed (knots)')
    ax.set_title('Wind Speed Evolution Over Time')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Location scatter
    ax = fig.add_subplot(gs[2, 0])
    for i, typhoon_id in enumerate(unique_typhoons):
        mask = data['typhoon_ids'] == typhoon_id
        name = data['typhoon_names'][mask][0]
        ax.scatter(data['lons'][mask], data['lats'][mask], 
                  c=[colors_pie[i]], label=name, alpha=0.6, s=15, edgecolors='black', linewidth=0.3)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Typhoon Tracks')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Wind speed histogram
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(data['winds'], bins=40, alpha=0.7, edgecolor='black', color='steelblue')
    ax.set_xlabel('Wind Speed (knots)')
    ax.set_ylabel('Frequency')
    ax.set_title('Wind Speed Distribution')
    ax.axvline(data['winds'].mean(), color='red', linestyle='--', 
              label=f'Mean: {data["winds"].mean():.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Feature value distribution
    ax = fig.add_subplot(gs[2, 2])
    flat_features = data['features'].flatten()
    ax.hist(flat_features, bins=60, alpha=0.7, edgecolor='black', color='coral')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Feature Value Distribution')
    ax.axvline(flat_features.mean(), color='red', linestyle='--',
              label=f'Mean: {flat_features.mean():.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(Path(output_dir) / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: summary_dashboard.png")
    plt.close()

def plot_visual_text_alignment(data, output_dir, data_dir):
    """Visualize fused visual + text alignment"""
    print("\nCreating visual + text fused alignment plots...")
    
    # Load metadata with text descriptions
    metadata_path = Path(data_dir) / 'metadata.json'
    if not metadata_path.exists():
        print("Warning: metadata.json not found, skipping text analysis")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    features = data['features']  # These are the fused (visual + text) features
    winds = data['winds']
    lats = data['lats']
    lons = data['lons']
    unique_typhoons = np.unique(data['typhoon_ids'])
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    fig.suptitle('Fused Visual + Text Alignment Analysis', fontsize=18, fontweight='bold')
    
    # 1. Text description keyword analysis
    ax = fig.add_subplot(gs[0, 0])
    text_descriptions = [m.get('text_description', '') for m in metadata]
    
    all_words = []
    for desc in text_descriptions:
        words = re.findall(r'\b\w+\b', desc.lower())
        all_words.extend(words)
    
    stop_words = {'typhoon', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'degrees'}
    filtered_words = [w for w in all_words if w not in stop_words and len(w) > 2]
    
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(15)
    
    if top_words:
        words, counts = zip(*top_words)
        ax.barh(range(len(words)), counts, color='steelblue')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('Frequency')
        ax.set_title('Top Keywords in Text Descriptions')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 2. Feature correlation with text content (wind speed)
    ax = fig.add_subplot(gs[0, 1])
    # Find features most correlated with wind (text mentions wind speed)
    wind_correlations = []
    for i in range(min(100, features.shape[1])):
        corr = np.corrcoef(features[:, i], winds)[0, 1]
        wind_correlations.append(abs(corr))
    
    top_indices = np.argsort(wind_correlations)[-20:][::-1]
    top_corrs = [wind_correlations[i] for i in top_indices]
    
    ax.barh(range(len(top_corrs)), top_corrs, color='orange')
    ax.set_yticks(range(len(top_corrs)))
    ax.set_yticklabels([f'Dim {i}' for i in top_indices])
    ax.set_xlabel('|Correlation with Wind Speed|')
    ax.set_title('Features Correlated with Wind (Text Info)')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Text description length vs feature magnitude
    ax = fig.add_subplot(gs[0, 2])
    desc_lengths = [len(desc.split()) for desc in text_descriptions]
    feature_magnitudes = np.linalg.norm(features, axis=1)
    
    scatter = ax.scatter(desc_lengths, feature_magnitudes, alpha=0.5, s=20, 
                        c=winds, cmap='viridis', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Text Description Length (words)')
    ax.set_ylabel('Feature Vector Magnitude')
    ax.set_title('Text Length vs Feature Magnitude\n(colored by wind speed)')
    plt.colorbar(scatter, ax=ax, label='Wind Speed (knots)')
    ax.grid(True, alpha=0.3)
    
    # 4. Feature similarity matrix (showing alignment quality)
    ax = fig.add_subplot(gs[1, 0])
    sample_size = min(50, len(features))
    sample_indices = np.random.choice(len(features), sample_size, replace=False)
    sample_features = features[sample_indices]
    
    # Compute cosine similarity
    similarity_matrix = np.zeros((sample_size, sample_size))
    for i in range(sample_size):
        for j in range(sample_size):
            feat1 = sample_features[i] / (np.linalg.norm(sample_features[i]) + 1e-8)
            feat2 = sample_features[j] / (np.linalg.norm(sample_features[j]) + 1e-8)
            similarity_matrix[i, j] = np.dot(feat1, feat2)
    
    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')
    ax.set_title('Fused Feature Similarity Matrix\n(Visual + Text Alignment)')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    # 5. Feature dimensions activated by high/low wind (text info)
    ax = fig.add_subplot(gs[1, 1])
    high_wind_mask = winds > np.percentile(winds, 75)
    low_wind_mask = winds < np.percentile(winds, 25)
    
    if high_wind_mask.sum() > 0 and low_wind_mask.sum() > 0:
        high_wind_features = features[high_wind_mask].mean(axis=0)
        low_wind_features = features[low_wind_mask].mean(axis=0)
        
        diff = high_wind_features - low_wind_features
        top_dims = np.argsort(np.abs(diff))[-20:][::-1]
        
        colors_bar = ['green' if d > 0 else 'red' for d in diff[top_dims]]
        ax.barh(range(len(top_dims)), diff[top_dims], color=colors_bar)
        ax.set_yticks(range(len(top_dims)))
        ax.set_yticklabels([f'Dim {d}' for d in top_dims])
        ax.set_xlabel('Feature Difference (High - Low Wind)')
        ax.set_title('Feature Dimensions Activated by Wind Speed')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    # 6. Sample text descriptions with feature stats
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    
    sample_texts = []
    for i in range(min(6, len(metadata))):
        desc = metadata[i].get('text_description', '')[:55]
        feat_mean = features[i].mean()
        feat_std = features[i].std()
        wind_val = metadata[i].get('wind', 0)
        lat_val = metadata[i].get('lat', 0)
        lon_val = metadata[i].get('lon', 0)
        sample_texts.append(
            f"[{i+1}] {desc}...\n"
            f"    Feature: μ={feat_mean:.4f}, σ={feat_std:.4f}\n"
            f"    Wind: {wind_val}kt, Loc: ({lat_val:.1f}°, {lon_val:.1f}°)\n"
        )
    
    text_content = "\n".join(sample_texts)
    ax.text(0.05, 0.95, text_content, fontsize=8.5, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Sample Text Descriptions\nwith Fused Feature Statistics')
    
    # 7. Temporal alignment: Features + Text + Wind for one typhoon
    ax = fig.add_subplot(gs[2, :])
    if len(unique_typhoons) > 0:
        typhoon_mask = data['typhoon_ids'] == unique_typhoons[0]
        typhoon_features = features[typhoon_mask]
        typhoon_metadata = [m for m, mask in zip(metadata, typhoon_mask) if mask]
        typhoon_winds = winds[typhoon_mask]
        typhoon_lats = lats[typhoon_mask]
        typhoon_lons = lons[typhoon_mask]
        
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        
        timesteps = np.arange(len(typhoon_features))
        
        # Feature mean
        feature_means = typhoon_features.mean(axis=1)
        line1 = ax.plot(timesteps, feature_means, 'b-', linewidth=2.5, 
                       label='Fused Feature Mean (Visual+Text)', alpha=0.8)
        
        # Wind speed
        line2 = ax2.plot(timesteps, typhoon_winds, 'r-', linewidth=2, 
                        label='Wind Speed', alpha=0.8)
        
        # Feature magnitude
        feature_norms = np.linalg.norm(typhoon_features, axis=1)
        line3 = ax3.plot(timesteps, feature_norms, 'g--', linewidth=2, 
                        label='Feature Magnitude', alpha=0.7)
        
        # Add text annotations at key points
        key_indices = np.linspace(0, len(timesteps)-1, min(8, len(timesteps)), dtype=int)
        for idx in key_indices:
            desc = typhoon_metadata[idx].get('text_description', '')[:35]
            ax.axvline(idx, color='gray', linestyle=':', alpha=0.4, linewidth=1)
            ax.text(idx, feature_means[idx], f'{desc}...', rotation=90, 
                   fontsize=7, alpha=0.6, verticalalignment='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        ax.set_xlabel('Timestep', fontsize=11)
        ax.set_ylabel('Mean Feature Value', color='b', fontsize=11)
        ax2.set_ylabel('Wind Speed (knots)', color='r', fontsize=11)
        ax3.set_ylabel('Feature Magnitude', color='g', fontsize=11)
        
        typhoon_name = typhoon_metadata[0].get('typhoon_name', unique_typhoons[0])
        ax.set_title(f'Fused Visual + Text Alignment Over Time\nTyphoon: {typhoon_name} ({unique_typhoons[0]})', 
                    fontsize=12, fontweight='bold')
        
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax3.tick_params(axis='y', labelcolor='g')
    
    # 8. Feature vs Wind relationship (showing text alignment)
    ax = fig.add_subplot(gs[3, 0])
    feature_means = features.mean(axis=1)
    scatter = ax.scatter(feature_means, winds, alpha=0.5, s=20, 
                        c=lats, cmap='coolwarm', edgecolors='black', linewidth=0.5)
    
    # Fit polynomial
    z = np.polyfit(feature_means, winds, 2)
    p = np.poly1d(z)
    ax.plot(sorted(feature_means), p(sorted(feature_means)), "r--", 
           alpha=0.8, linewidth=2, label='Polynomial Fit')
    
    ax.set_xlabel('Mean Feature Value')
    ax.set_ylabel('Wind Speed (knots)')
    ax.set_title('Feature-Wind Relationship\n(colored by latitude)')
    plt.colorbar(scatter, ax=ax, label='Latitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 9. Feature space colored by text keywords
    ax = fig.add_subplot(gs[3, 1])
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features.astype(np.float32))
    
    # Color by wind speed (from text)
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                        c=winds, cmap='YlOrRd', s=15, alpha=0.6, 
                        edgecolors='black', linewidth=0.3)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_title('Feature Space (colored by wind from text)')
    plt.colorbar(scatter, ax=ax, label='Wind Speed (knots)')
    ax.grid(True, alpha=0.3)
    
    # 10. Alignment quality metrics
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    
    # Compute alignment metrics
    feature_mean = features.mean()
    feature_std = features.std()
    feature_norm_mean = np.linalg.norm(features, axis=1).mean()
    
    # Correlation between features and wind
    wind_corr = np.corrcoef(feature_means, winds)[0, 1]
    
    # Feature consistency (low std means aligned)
    feature_consistency = 1 / (1 + features.std(axis=1).mean())
    
    metrics_text = f"""
    Alignment Quality Metrics:
    
    Feature Statistics:
    • Mean: {feature_mean:.4f}
    • Std: {feature_std:.4f}
    • Mean Norm: {feature_norm_mean:.4f}
    
    Text-Feature Alignment:
    • Wind Correlation: {wind_corr:.4f}
    • Feature Consistency: {feature_consistency:.4f}
    
    Sample Count:
    • Total: {len(features):,}
    • Typhoons: {len(unique_typhoons)}
    • Avg per Typhoon: {len(features)/len(unique_typhoons):.0f}
    
    Text Coverage:
    • Avg Description Length: {np.mean([len(d.split()) for d in text_descriptions]):.1f} words
    • Unique Keywords: {len(set(all_words)):,}
    """
    
    ax.text(0.05, 0.95, metrics_text, fontsize=9.5, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.set_title('Alignment Quality Metrics')
    
    plt.savefig(Path(output_dir) / 'visual_text_alignment.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: visual_text_alignment.png")
    plt.close()

def main():
    """Main function"""
    data_dir = r"D:\typhoon_aligned"
    output_dir = Path(data_dir) / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("Typhoon CLIP Alignment Visualization")
    print("="*80)
    
    # Load data
    data = load_data(data_dir)
    
    # Create visualizations
    plot_feature_statistics(data, output_dir)
    plot_typhoon_tracks(data, output_dir)
    plot_feature_reduction(data, output_dir)
    plot_typhoon_comparison(data, output_dir)
    plot_summary_dashboard(data, output_dir)
    plot_visual_text_alignment(data, output_dir, data_dir)
    
    print("\n" + "="*80)
    print("Visualization Complete!")
    print(f"All plots saved to: {output_dir}")
    print("="*80)
    print("\nGenerated files:")
    print("  - feature_statistics.png")
    print("  - typhoon_tracks.png")
    print("  - feature_reduction.png")
    print("  - typhoon_comparison.png")
    print("  - summary_dashboard.png")
    print("  - visual_text_alignment.png")
    print("="*80)

if __name__ == '__main__':
    main()

