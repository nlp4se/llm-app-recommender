import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse
import re
import pickle
import hashlib
import json
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Define consistent colors for models (matching rq2 and rq4)
MODEL_COLORS = {
    'openai': '#FF6B6B',      # Red
    'anthropic': '#4ECDC4',    # Teal
    'gemini': '#45B7D1',       # Blue
    'mistral': '#96CEB4',      # Green
    'perplexity': '#FFEAA7'    # Yellow
}

def generate_cache_key(llm_criteria, standard_criteria, model_name='all-MiniLM-L6-v2'):
    """
    Generate a unique cache key based on the criteria content and model.
    """
    # Create a hash of the criteria content
    criteria_hash = hashlib.md5()
    
    # Add LLM criteria to hash
    for crit in llm_criteria:
        criteria_hash.update(crit.encode('utf-8'))
    
    # Add standard criteria to hash
    for crit in standard_criteria:
        criteria_hash.update(crit.encode('utf-8'))
    
    # Add model name to hash
    criteria_hash.update(model_name.encode('utf-8'))
    
    return criteria_hash.hexdigest()

def get_cache_path(output_dir, cache_key):
    """
    Get the path for cached embeddings.
    """
    cache_dir = os.path.join(output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'embeddings_{cache_key}.pkl')

def load_cached_embeddings(cache_path):
    """
    Load cached embeddings if they exist.
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"Loaded cached embeddings from {cache_path}")
            return cached_data['embeddings'], cached_data['tsne_coords']
        except Exception as e:
            print(f"Error loading cached embeddings: {e}")
            return None, None
    return None, None

def save_cached_embeddings(cache_path, embeddings, tsne_coords):
    """
    Save embeddings to cache.
    """
    try:
        cached_data = {
            'embeddings': embeddings,
            'tsne_coords': tsne_coords
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"Saved embeddings to cache: {cache_path}")
    except Exception as e:
        print(f"Error saving embeddings to cache: {e}")

def load_ranking_criteria_data(input_file, rc_file):
    """
    Load and prepare the ranking criteria data.
    """
    # Load main data
    df = pd.read_csv(input_file)
    
    # Load standard ranking criteria
    rc_df = pd.read_csv(rc_file, sep=';')
    
    return df, rc_df

def prepare_criteria_texts(df, rc_df):
    """
    Prepare criteria texts for embedding and visualization.
    """
    # Prepare LLM-generated criteria
    llm_criteria = []
    llm_metadata = []
    
    for _, row in df.iterrows():
        if pd.notna(row['n']) and pd.notna(row['d']):
            criteria_text = f"{row['n']}: {row['d']}"
            llm_criteria.append(criteria_text)
            llm_metadata.append({
                'model': row['model'],
                'feature': row['feature'],
                'run': row['run'],
                'name': row['n'],
                'description': row['d']
            })
    
    # Prepare standard ranking criteria
    standard_criteria = []
    standard_metadata = []
    
    for _, row in rc_df.iterrows():
        criteria_text = f"{row['name']}: {row['description']}"
        standard_criteria.append(criteria_text)
        standard_metadata.append({
            'id': row['id'],
            'name': row['name'],
            'description': row['description'],
            'type': row['type']
        })
    
    return llm_criteria, llm_metadata, standard_criteria, standard_metadata

def create_embeddings_and_visualization(llm_criteria, llm_metadata, standard_criteria, standard_metadata, output_dir, use_cache=True):
    """
    Create embeddings and generate the 2D visualization with caching support.
    """
    # Generate cache key
    cache_key = generate_cache_key(llm_criteria, standard_criteria)
    cache_path = get_cache_path(output_dir, cache_key)
    
    # Try to load from cache first
    if use_cache:
        cached_embeddings, cached_tsne_coords = load_cached_embeddings(cache_path)
        if cached_embeddings is not None and cached_tsne_coords is not None:
            print("Using cached embeddings and t-SNE coordinates")
            embeddings = cached_embeddings
            coords_2d = cached_tsne_coords
        else:
            # Create new embeddings
            print("Loading sentence embedding model...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Combine all criteria for embedding
            all_criteria = llm_criteria + standard_criteria
            
            print("Creating embeddings...")
            embeddings = model.encode(all_criteria)
            
            # Apply t-SNE for dimensionality reduction
            print("Applying t-SNE dimensionality reduction...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            coords_2d = tsne.fit_transform(embeddings)
            
            # Save to cache
            save_cached_embeddings(cache_path, embeddings, coords_2d)
    else:
        # Create new embeddings without using cache
        print("Loading sentence embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Combine all criteria for embedding
        all_criteria = llm_criteria + standard_criteria
        
        print("Creating embeddings...")
        embeddings = model.encode(all_criteria)
        
        # Apply t-SNE for dimensionality reduction
        print("Applying t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        coords_2d = tsne.fit_transform(embeddings)
    
    # Split coordinates for LLM and standard criteria
    llm_coords = coords_2d[:len(llm_criteria)]
    standard_coords = coords_2d[len(llm_criteria):]
    
    # Keep ALL standard criteria (centroids) - no filtering
    standard_coords_filtered = standard_coords
    standard_metadata_filtered = standard_metadata
    
    # Calculate distance threshold based on distances from LLM criteria to nearest standard criteria
    llm_to_standard_distances = []
    for llm_coord in llm_coords:
        # Calculate distance to each standard criteria
        distances_to_standards = []
        for standard_coord in standard_coords:
            distance = np.sqrt((llm_coord[0] - standard_coord[0])**2 + (llm_coord[1] - standard_coord[1])**2)
            distances_to_standards.append(distance)
        # Find the minimum distance to any standard criteria
        min_distance = min(distances_to_standards)
        llm_to_standard_distances.append(min_distance)
    
    # Use 90th percentile as threshold for LLM criteria filtering
    distance_threshold = np.percentile(llm_to_standard_distances, 90)
    
    # Filter LLM criteria: only keep those close to at least one standard criteria
    llm_coords_filtered = []
    llm_metadata_filtered = []
    for i, (coord, metadata) in enumerate(zip(llm_coords, llm_metadata)):
        # Calculate distance to nearest standard criteria
        distances_to_standards = []
        for standard_coord in standard_coords:
            distance = np.sqrt((coord[0] - standard_coord[0])**2 + (coord[1] - standard_coord[1])**2)
            distances_to_standards.append(distance)
        min_distance = min(distances_to_standards)
        
        # Keep if close to at least one standard criteria
        if min_distance <= distance_threshold:
            llm_coords_filtered.append(coord)
            llm_metadata_filtered.append(metadata)
    
    print(f"Filtered out {len(llm_coords) - len(llm_coords_filtered)} LLM outliers (distance threshold: {distance_threshold:.3f})")
    print(f"Kept all {len(standard_coords_filtered)} standard criteria (centroids)")
    
    # Create the visualization
    plt.figure(figsize=(16, 12))
    
    # Plot LLM criteria as colored nodes
    for i, (coord, metadata) in enumerate(zip(llm_coords_filtered, llm_metadata_filtered)):
        model = metadata['model']
        color = MODEL_COLORS.get(model, '#808080')  # Gray for unknown models
        
        plt.scatter(coord[0], coord[1], 
                   c=[color], s=50, alpha=0.7, 
                   edgecolors='black', linewidth=0.5)
    
    # Plot ALL standard criteria as large centroids (no filtering)
    for i, (coord, metadata) in enumerate(zip(standard_coords_filtered, standard_metadata_filtered)):
        plt.scatter(coord[0], coord[1], 
                   c=['#FF6B35'], s=200, alpha=0.8, 
                   marker='*', edgecolors='black', linewidth=1.5)
        
        # Custom offsets for edge labels
        label_offsets = {
            "Customer Support": (-60, 20),   # Move left and up
            "Update Frequency": (40, -20),   # Move right and down
            # Add more if needed
        }
        offset = label_offsets.get(metadata['name'], (5, 5))
        
        plt.annotate(metadata['name'], 
                    (coord[0], coord[1]), 
                    xytext=offset, textcoords='offset points',
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Create legend with larger fonts
    legend_elements = []
    for model, color in MODEL_COLORS.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        label=model.title()))
    
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                                    markerfacecolor='#FF6B35', markersize=15, 
                                    label='Standard Criteria'))
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=16)  # Increased from 12 to 16
    plt.title('Ranking Criteria Distribution: LLM vs Standard Criteria', fontsize=20, fontweight='bold')  # Increased from 16 to 20
    plt.xlabel('t-SNE Dimension 1', fontsize=18)  # Increased from 14 to 18
    plt.ylabel('t-SNE Dimension 2', fontsize=18)  # Increased from 14 to 18
    plt.grid(True, alpha=0.3)
    
    # Increase tick label font sizes
    plt.xticks(fontsize=14)  # Added tick font size
    plt.yticks(fontsize=14)  # Added tick font size
    
    # Add extra margins to ensure all labels are visible
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    x_margin = (x_max - x_min) * 0.1  # 10% margin
    y_margin = (y_max - y_min) * 0.1
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'ranking_criteria_distribution.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")
    
    return llm_coords, standard_coords, embeddings

def calculate_similarity_analysis(llm_criteria, llm_metadata, standard_criteria, standard_metadata, embeddings, output_dir):
    """
    Calculate similarity between LLM criteria and standard criteria.
    """
    print("Calculating similarity analysis...")
    
    # Split embeddings
    llm_embeddings = embeddings[:len(llm_criteria)]
    standard_embeddings = embeddings[len(llm_criteria):]
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(llm_embeddings, standard_embeddings)
    
    # Find best matches for each LLM criteria
    best_matches = []
    for i, (llm_crit, llm_meta) in enumerate(zip(llm_criteria, llm_metadata)):
        similarities = similarity_matrix[i]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_standard = standard_metadata[best_idx]
        
        best_matches.append({
            'model': llm_meta['model'],
            'feature': llm_meta['feature'],
            'llm_criteria': llm_crit,
            'best_match': f"{best_standard['name']}: {best_standard['description']}",
            'similarity_score': best_score,
            'standard_criteria_id': best_standard['id']
        })
    
    # Create summary statistics
    match_df = pd.DataFrame(best_matches)
    
    # Model-wise analysis
    model_analysis = match_df.groupby('model').agg({
        'similarity_score': ['mean', 'std', 'count'],
        'standard_criteria_id': lambda x: x.value_counts().to_dict()
    }).round(3)
    
    # Feature-wise analysis
    feature_analysis = match_df.groupby('feature').agg({
        'similarity_score': ['mean', 'std', 'count'],
        'model': lambda x: x.value_counts().to_dict()
    }).round(3)
    
    # Standard criteria coverage
    coverage_analysis = match_df['standard_criteria_id'].value_counts()
    
    # Save detailed results
    match_df.to_csv(os.path.join(output_dir, 'similarity_analysis_detailed.csv'), index=False)
    
    return match_df, model_analysis, feature_analysis, coverage_analysis

def calculate_internal_consistency(df, embeddings, llm_metadata, output_dir):
    """
    Calculate internal consistency for each LLM.
    Internal consistency: How consistent are ranking criteria within each LLM?
    """
    print("Calculating internal consistency...")
    
    # Group by model and calculate pairwise similarities within each model
    internal_consistency_results = []
    
    # Get LLM embeddings (first part of embeddings array)
    llm_embeddings = embeddings[:len(llm_metadata)]
    
    for model in df['model'].unique():
        model_mask = [meta['model'] == model for meta in llm_metadata]
        model_embeddings = llm_embeddings[model_mask]
        model_metadata = [meta for i, meta in enumerate(llm_metadata) if model_mask[i]]
        
        if len(model_embeddings) < 2:
            continue
            
        # Calculate pairwise similarities within the model
        similarity_matrix = cosine_similarity(model_embeddings)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        internal_consistency_results.append({
            'model': model,
            'num_criteria': len(model_embeddings),
            'mean_internal_similarity': np.mean(upper_triangle),
            'std_internal_similarity': np.std(upper_triangle),
            'min_internal_similarity': np.min(upper_triangle),
            'max_internal_similarity': np.max(upper_triangle),
            'internal_consistency_score': np.mean(upper_triangle)  # Higher = more consistent
        })
    
    internal_df = pd.DataFrame(internal_consistency_results)
    internal_df.to_csv(os.path.join(output_dir, 'internal_consistency.csv'), index=False)
    
    return internal_df

def calculate_external_consistency(df, embeddings, llm_metadata, output_dir):
    """
    Calculate external consistency between LLMs.
    External consistency: How consistent are ranking criteria between different LLMs?
    """
    print("Calculating external consistency...")
    
    # Get LLM embeddings (first part of embeddings array)
    llm_embeddings = embeddings[:len(llm_metadata)]
    
    # Get embeddings for each model
    model_embeddings = {}
    for model in df['model'].unique():
        model_mask = [meta['model'] == model for meta in llm_metadata]
        model_embeddings[model] = llm_embeddings[model_mask]
    
    # Calculate pairwise similarities between models
    external_consistency_results = []
    model_pairs = []
    
    for i, model1 in enumerate(df['model'].unique()):
        for j, model2 in enumerate(df['model'].unique()):
            if i < j:  # Avoid duplicate pairs
                if model1 in model_embeddings and model2 in model_embeddings:
                    emb1 = model_embeddings[model1]
                    emb2 = model_embeddings[model2]
                    
                    # Calculate cross-model similarities
                    cross_similarity = cosine_similarity(emb1, emb2)
                    
                    external_consistency_results.append({
                        'model1': model1,
                        'model2': model2,
                        'num_criteria_model1': len(emb1),
                        'num_criteria_model2': len(emb2),
                        'mean_cross_similarity': np.mean(cross_similarity),
                        'std_cross_similarity': np.std(cross_similarity),
                        'min_cross_similarity': np.min(cross_similarity),
                        'max_cross_similarity': np.max(cross_similarity),
                        'external_consistency_score': np.mean(cross_similarity)
                    })
                    model_pairs.append((model1, model2))
    
    external_df = pd.DataFrame(external_consistency_results)
    external_df.to_csv(os.path.join(output_dir, 'external_consistency.csv'), index=False)
    
    return external_df, model_pairs

def create_consistency_tables(internal_df, external_df, output_dir):
    """
    Create comprehensive tables for internal and external consistency.
    """
    print("Creating consistency tables...")
    
    # 1. Internal Consistency Summary Table
    internal_summary = internal_df.copy()
    internal_summary['consistency_level'] = pd.cut(
        internal_summary['internal_consistency_score'], 
        bins=[0, 0.3, 0.6, 1.0], 
        labels=['Low', 'Medium', 'High']
    )
    
    # 2. External Consistency Matrix
    models = sorted(internal_df['model'].unique())
    external_matrix = pd.DataFrame(index=models, columns=models)
    
    for _, row in external_df.iterrows():
        external_matrix.loc[row['model1'], row['model2']] = row['external_consistency_score']
        external_matrix.loc[row['model2'], row['model1']] = row['external_consistency_score']
    
    # Fill diagonal with internal consistency scores
    for model in models:
        internal_score = internal_df[internal_df['model'] == model]['internal_consistency_score'].iloc[0]
        external_matrix.loc[model, model] = internal_score
    
    # 3. Create formatted tables
    with open(os.path.join(output_dir, 'consistency_tables.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RANKING CRITERIA CONSISTENCY ANALYSIS TABLES\n")
        f.write("=" * 80 + "\n\n")
        
        # Internal Consistency Table
        f.write("INTERNAL CONSISTENCY (Within each LLM)\n")
        f.write("-" * 50 + "\n")
        f.write("Model | Criteria Count | Mean Similarity | Std Dev | Consistency Level\n")
        f.write("-" * 50 + "\n")
        for _, row in internal_summary.iterrows():
            f.write(f"{row['model']:10} | {row['num_criteria']:13} | {row['internal_consistency_score']:14.3f} | {row['std_internal_similarity']:7.3f} | {row['consistency_level']:16}\n")
        f.write("\n")
        
        # External Consistency Table
        f.write("EXTERNAL CONSISTENCY (Between LLMs)\n")
        f.write("-" * 50 + "\n")
        f.write("Model Pair | Mean Cross-Similarity | Std Dev | Consistency Level\n")
        f.write("-" * 50 + "\n")
        for _, row in external_df.iterrows():
            consistency_level = 'High' if row['external_consistency_score'] > 0.6 else 'Medium' if row['external_consistency_score'] > 0.3 else 'Low'
            f.write(f"{row['model1']}-{row['model2']:10} | {row['external_consistency_score']:20.3f} | {row['std_cross_similarity']:7.3f} | {consistency_level:16}\n")
        f.write("\n")
        
        # Summary Statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average Internal Consistency: {internal_df['internal_consistency_score'].mean():.3f}\n")
        f.write(f"Average External Consistency: {external_df['external_consistency_score'].mean():.3f}\n")
        f.write(f"Most Internally Consistent: {internal_df.loc[internal_df['internal_consistency_score'].idxmax(), 'model']}\n")
        f.write(f"Most Externally Consistent Pair: {external_df.loc[external_df['external_consistency_score'].idxmax(), 'model1']}-{external_df.loc[external_df['external_consistency_score'].idxmax(), 'model2']}\n")
    
    # Save tables as CSV
    internal_summary.to_csv(os.path.join(output_dir, 'internal_consistency_summary.csv'), index=False)
    external_matrix.to_csv(os.path.join(output_dir, 'external_consistency_matrix.csv'))
    
    return internal_summary, external_matrix

def create_consistency_visualizations(internal_df, external_df, output_dir):
    """
    Create visualizations for internal and external consistency.
    """
    print("Creating consistency visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Internal Consistency Bar Chart
    axes[0, 0].bar(internal_df['model'], internal_df['internal_consistency_score'], 
                   color=[MODEL_COLORS.get(model, '#808080') for model in internal_df['model']])
    axes[0, 0].set_title('Internal Consistency by Model', fontweight='bold')
    axes[0, 0].set_ylabel('Internal Consistency Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. External Consistency Heatmap
    models = sorted(internal_df['model'].unique())
    external_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    
    # Initialize with NaN
    external_matrix.fillna(np.nan, inplace=True)
    
    # Fill external consistency scores
    for _, row in external_df.iterrows():
        external_matrix.loc[row['model1'], row['model2']] = row['external_consistency_score']
        external_matrix.loc[row['model2'], row['model1']] = row['external_consistency_score']
    
    # Fill diagonal with internal consistency scores
    for model in models:
        internal_score = internal_df[internal_df['model'] == model]['internal_consistency_score'].iloc[0]
        external_matrix.loc[model, model] = internal_score
    
    # Convert to numeric, replacing any remaining non-numeric values with NaN
    external_matrix = external_matrix.apply(pd.to_numeric, errors='coerce')
    
    # Create heatmap only if we have valid data
    if not external_matrix.isna().all().all():
        sns.heatmap(external_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                    center=0.5, ax=axes[0, 1], cbar_kws={'label': 'Consistency Score'})
        axes[0, 1].set_title('Consistency Matrix (Internal + External)', fontweight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, 'No valid consistency data available', 
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Consistency Matrix (No Data)', fontweight='bold')
    
    # 3. Consistency Distribution
    all_internal_scores = internal_df['internal_consistency_score'].values
    all_external_scores = external_df['external_consistency_score'].values
    
    axes[1, 0].hist(all_internal_scores, alpha=0.7, label='Internal', bins=10, color='blue')
    axes[1, 0].hist(all_external_scores, alpha=0.7, label='External', bins=10, color='red')
    axes[1, 0].set_title('Distribution of Consistency Scores', fontweight='bold')
    axes[1, 0].set_xlabel('Consistency Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # 4. Model Comparison
    x = np.arange(len(internal_df))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, internal_df['internal_consistency_score'], width, 
                   label='Internal', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, external_df['external_consistency_score'].mean(), width, 
                   label='External (Avg)', color='red', alpha=0.7)
    axes[1, 1].set_title('Internal vs External Consistency', fontweight='bold')
    axes[1, 1].set_ylabel('Consistency Score')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(internal_df['model'], rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'consistency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_standard_criteria_coverage_bar(coverage_analysis, standard_metadata, output_dir):
    """
    Plot and save the coverage of standard ranking criteria as a standalone bar plot.
    """
    import matplotlib.pyplot as plt

    # Map IDs to names using standard_metadata
    id_to_name = {meta['id']: meta['name'] for meta in standard_metadata}
    names = [id_to_name.get(idx, str(idx)) for idx in coverage_analysis.index]
    counts = coverage_analysis.values

    plt.figure(figsize=(14, 7))
    bars = plt.bar(names, counts, color="#3498db", edgecolor='black')  # Blue bars

    plt.ylabel('Number of LLM ranking criteria', fontsize=16)
    plt.xlabel('Standard Ranking Criteria', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)

    # Add count labels inside bars, in white
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, str(count),
                 ha='center', va='center', color='white', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'standard_criteria_coverage_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_criteria_distribution_pie(match_df, output_dir):
    """
    Plot and save the distribution of criteria by model as a standalone pie chart.
    """
    import matplotlib.pyplot as plt

    # Mapping for capitalization
    model_label_map = {
        'openai': 'OpenAI',
        'anthropic': 'Anthropic',
        'mistral': 'Mistral',
        'gemini': 'Gemini',
        'perplexity': 'Perplexity'
    }

    model_criteria_count = match_df['model'].value_counts()
    model_labels = [model_label_map.get(model, model.title()) for model in model_criteria_count.index]
    model_colors = [MODEL_COLORS.get(model, '#808080') for model in model_criteria_count.index]

    plt.figure(figsize=(8, 8))
    plt.pie(model_criteria_count.values, labels=model_labels,
            autopct='%1.1f%%', colors=model_colors, textprops={'fontsize': 16})

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_criteria_distribution_pie.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_additional_visualizations(match_df, model_analysis, feature_analysis, coverage_analysis, output_dir, standard_metadata=None):
    """
    Create additional insightful visualizations.
    """
    print("Creating additional visualizations...")
    
    # 1. Model consistency in similarity scores
    plt.figure(figsize=(12, 8))
    
    # Box plot of similarity scores by model
    plt.subplot(2, 2, 1)
    model_scores = [match_df[match_df['model'] == model]['similarity_score'].values 
                   for model in MODEL_COLORS.keys() if model in match_df['model'].values]
    model_labels = [model for model in MODEL_COLORS.keys() if model in match_df['model'].values]
    
    plt.boxplot(model_scores, labels=model_labels)
    plt.title('Similarity Score Distribution by Model', fontweight='bold')
    plt.ylabel('Similarity Score')
    plt.xticks(rotation=45)
    
    # 2. Feature-wise similarity scores
    plt.subplot(2, 2, 2)
    feature_scores = match_df.groupby('feature')['similarity_score'].mean().sort_values(ascending=False)
    plt.bar(range(len(feature_scores)), feature_scores.values, 
            color=[MODEL_COLORS.get(model, '#808080') for model in feature_scores.index])
    plt.title('Average Similarity Score by Feature', fontweight='bold')
    plt.ylabel('Average Similarity Score')
    plt.xticks(range(len(feature_scores)), feature_scores.index, rotation=45, ha='right')
    
    # 3. Standard criteria coverage
    plt.subplot(2, 2, 3)
    coverage_analysis.plot(kind='bar')
    plt.title('Coverage of Standard Ranking Criteria', fontweight='bold')
    plt.ylabel('Number of LLM Criteria Matches')
    plt.xticks(rotation=45, ha='right')
    
    # 4. Model diversity in criteria generation
    plt.subplot(2, 2, 4)
    model_criteria_count = match_df['model'].value_counts()
    plt.pie(model_criteria_count.values, labels=model_criteria_count.index, 
            autopct='%1.1f%%', colors=[MODEL_COLORS.get(model, '#808080') for model in model_criteria_count.index])
    plt.title('Distribution of Criteria by Model', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'additional_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save independent plots with improved formatting
    if standard_metadata is not None:
        plot_standard_criteria_coverage_bar(coverage_analysis, standard_metadata, output_dir)
    plot_model_criteria_distribution_pie(match_df, output_dir)

    # Create a comprehensive report
    create_numerical_report(match_df, model_analysis, feature_analysis, coverage_analysis, output_dir)

def create_numerical_report(match_df, model_analysis, feature_analysis, coverage_analysis, output_dir):
    """
    Create a comprehensive numerical report.
    """
    print("Creating numerical report...")
    
    report = []
    report.append("=" * 80)
    report.append("RANKING CRITERIA CONSISTENCY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS:")
    report.append(f"Total LLM criteria analyzed: {len(match_df)}")
    report.append(f"Number of models: {match_df['model'].nunique()}")
    report.append(f"Number of features: {match_df['feature'].nunique()}")
    report.append(f"Average similarity score: {match_df['similarity_score'].mean():.3f}")
    report.append(f"Standard deviation: {match_df['similarity_score'].std():.3f}")
    report.append("")
    
    # Model-wise analysis
    report.append("MODEL-WISE ANALYSIS:")
    report.append("-" * 40)
    for model in MODEL_COLORS.keys():
        if model in match_df['model'].values:
            model_data = match_df[match_df['model'] == model]
            report.append(f"{model.upper()}:")
            report.append(f"  Number of criteria: {len(model_data)}")
            report.append(f"  Average similarity: {model_data['similarity_score'].mean():.3f}")
            report.append(f"  Standard deviation: {model_data['similarity_score'].std():.3f}")
            report.append(f"  Min similarity: {model_data['similarity_score'].min():.3f}")
            report.append(f"  Max similarity: {model_data['similarity_score'].max():.3f}")
            report.append("")
    
    # Feature-wise analysis
    report.append("FEATURE-WISE ANALYSIS:")
    report.append("-" * 40)
    for feature in match_df['feature'].unique():
        feature_data = match_df[match_df['feature'] == feature]
        report.append(f"{feature}:")
        report.append(f"  Number of criteria: {len(feature_data)}")
        report.append(f"  Average similarity: {feature_data['similarity_score'].mean():.3f}")
        report.append(f"  Models represented: {', '.join(feature_data['model'].unique())}")
        report.append("")
    
    # Standard criteria coverage
    report.append("STANDARD CRITERIA COVERAGE:")
    report.append("-" * 40)
    for criteria_id, count in coverage_analysis.items():
        report.append(f"{criteria_id}: {count} matches")
    report.append("")
    
    # Top and bottom matches
    report.append("TOP 10 HIGHEST SIMILARITY MATCHES:")
    report.append("-" * 40)
    top_matches = match_df.nlargest(10, 'similarity_score')
    for _, row in top_matches.iterrows():
        report.append(f"Score: {row['similarity_score']:.3f} | Model: {row['model']} | Feature: {row['feature']}")
        report.append(f"  LLM: {row['llm_criteria'][:100]}...")
        report.append(f"  Standard: {row['best_match'][:100]}...")
        report.append("")
    
    report.append("BOTTOM 10 LOWEST SIMILARITY MATCHES:")
    report.append("-" * 40)
    bottom_matches = match_df.nsmallest(10, 'similarity_score')
    for _, row in bottom_matches.iterrows():
        report.append(f"Score: {row['similarity_score']:.3f} | Model: {row['model']} | Feature: {row['feature']}")
        report.append(f"  LLM: {row['llm_criteria'][:100]}...")
        report.append(f"  Standard: {row['best_match'][:100]}...")
        report.append("")
    
    # Save report
    with open(os.path.join(output_dir, 'consistency_analysis_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Saved report to {os.path.join(output_dir, 'consistency_analysis_report.txt')}")

def main(input_file, output_dir, rc_file, use_cache=True):
    """
    Main function to run the complete analysis.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Loading data...")
    df, rc_df = load_ranking_criteria_data(input_file, rc_file)
    
    print("Preparing criteria texts...")
    llm_criteria, llm_metadata, standard_criteria, standard_metadata = prepare_criteria_texts(df, rc_df)
    
    print("Creating embeddings and visualization...")
    llm_coords, standard_coords, embeddings = create_embeddings_and_visualization(
        llm_criteria, llm_metadata, standard_criteria, standard_metadata, output_dir, use_cache)
    
    print("Calculating similarity analysis...")
    match_df, model_analysis, feature_analysis, coverage_analysis = calculate_similarity_analysis(
        llm_criteria, llm_metadata, standard_criteria, standard_metadata, embeddings, output_dir)
    
    # NEW: Calculate internal and external consistency
    print("Calculating internal consistency...")
    internal_df = calculate_internal_consistency(df, embeddings, llm_metadata, output_dir)
    
    print("Calculating external consistency...")
    external_df, model_pairs = calculate_external_consistency(df, embeddings, llm_metadata, output_dir)
    
    print("Creating consistency tables...")
    internal_summary, external_matrix = create_consistency_tables(internal_df, external_df, output_dir)
    
    print("Creating consistency visualizations...")
    create_consistency_visualizations(internal_df, external_df, output_dir)
    
    print("Creating additional visualizations...")
    create_additional_visualizations(match_df, model_analysis, feature_analysis, coverage_analysis, output_dir, standard_metadata=standard_metadata)
    
    print("Analysis complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze ranking criteria consistency with visualization.')
    parser.add_argument('--input-file', default='data/output/evaluation/app_ranking_criteria.csv',
                        help='Path to the input CSV file with ranking criteria.')
    parser.add_argument('--output-dir', default='data/output/evaluation/ranking_criteria_visualization',
                        help='Directory to save output files.')
    parser.add_argument('--rc-file', default='data/output/features/rq1/rc.csv',
                        help='Path to the standard ranking criteria CSV file.')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching of embeddings.')
    
    args = parser.parse_args()
    main(input_file=args.input_file, 
         output_dir=args.output_dir,
         rc_file=args.rc_file,
         use_cache=not args.no_cache)