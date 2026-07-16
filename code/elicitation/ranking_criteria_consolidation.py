import argparse
import pandas as pd
import os
import numpy as np
from code.hf_cache import load_sentence_transformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score
import json
import warnings
import re
from collections import Counter

from tqdm import tqdm
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Consolidate ranking criteria using systematic clustering approach.")
    parser.add_argument('--input-file', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output-folder', type=str, required=True, help='Folder to save output files')
    parser.add_argument('--k-range', type=str, default='2,50', help='Range for cluster validation (min,max)')
    parser.add_argument('--n-bootstrap', type=int, default=10, help='Number of bootstrap samples for gap statistic')
    return parser.parse_args()

def prepare_criteria_texts(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Coerce name/description to strings and drop rows that cannot be embedded."""
    prepared = df.copy()
    for col in ("name", "description"):
        if col not in prepared.columns:
            prepared[col] = ""
        prepared[col] = prepared[col].fillna("").astype(str).str.strip()

    before = len(prepared)
    prepared = prepared[(prepared["name"] != "") | (prepared["description"] != "")].reset_index(drop=True)
    dropped = before - len(prepared)
    if dropped:
        print(f"Dropped {dropped} rows with empty name and description before embedding")

    texts = [
        f"{row.name}. {row.description}".strip(". ")
        for row in prepared.itertuples(index=False)
    ]
    return prepared, texts


def systematic_clustering_approach(df, output_folder="", k_range=(2, 50), n_bootstrap=10):
    """
    Systematic clustering approach using multiple validation methods
    
    This approach uses:
    1. Hierarchical clustering with multiple linkage methods
    2. Silhouette analysis for optimal cluster number
    3. Gap statistic for validation
    4. Representative selection based on centrality measures
    
    Args:
        df: DataFrame with 'name' and 'description' columns
        output_folder: Folder to save analysis results
        k_range: Tuple of (min_k, max_k) for cluster validation
        n_bootstrap: Number of bootstrap samples for gap statistic
    
    Returns:
        Filtered DataFrame with representative criteria
    """
    print(f"\n=== Systematic Clustering Approach ===")

    df, texts = prepare_criteria_texts(df)
    if len(df) < 2:
        raise ValueError("Need at least 2 criteria with non-empty name or description to cluster.")

    # Initialize the model
    print("Loading sentence embedding model...")
    model = load_sentence_transformer("all-MiniLM-L6-v2")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = normalize(embeddings)
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Step 1: Determine optimal number of clusters using multiple methods
    optimal_k = determine_optimal_clusters(embeddings, similarity_matrix, output_folder, k_range, n_bootstrap)
    
    # Step 2: Perform hierarchical clustering with optimal parameters
    cluster_labels = perform_hierarchical_clustering(embeddings, optimal_k, output_folder)
    
    # Step 3: Select representatives using centrality measures
    df_filtered = select_cluster_representatives(df, embeddings, cluster_labels, output_folder)
    
    return df_filtered

def determine_optimal_clusters(embeddings, similarity_matrix, output_folder, k_range, n_bootstrap):
    """
    Determine optimal number of clusters using multiple validation methods.
    """
    print("\n--- Determining Optimal Number of Clusters ---")
    
    # Method 1: Silhouette Analysis
    print("Performing silhouette analysis...")
    silhouette_scores = silhouette_analysis(embeddings, k_range)
    
    # Method 2: Gap Statistic
    print("Computing gap statistic...")
    gap_scores = gap_statistic_analysis(embeddings, k_range, n_bootstrap)
    
    # Method 3: Elbow Method (inertia-based)
    print("Computing elbow method...")
    elbow_scores = elbow_method_analysis(embeddings, k_range)
    
    # Method 4: Calinski-Harabasz Index
    print("Computing Calinski-Harabasz index...")
    ch_scores = calinski_harabasz_analysis(embeddings, k_range)
    
    # Aggregate results using voting mechanism
    optimal_k = aggregate_cluster_validation(silhouette_scores, gap_scores, 
                                           elbow_scores, ch_scores, output_folder)
    
    print(f"Optimal number of clusters: {optimal_k}")
    return optimal_k

def silhouette_analysis(embeddings, k_range=(2, 50)):
    """
    Perform silhouette analysis to find optimal number of clusters.
    """
    silhouette_scores = []
    k_max = min(k_range[1] + 1, len(embeddings) // 2)
    k_values = range(k_range[0], k_max)

    for k in tqdm(
        k_values,
        desc="Silhouette analysis",
        unit="k",
        total=max(0, k_max - k_range[0]),
    ):
        if k >= len(embeddings):
            break

        clustering = AgglomerativeClustering(n_clusters=k, linkage="ward")
        cluster_labels = clustering.fit_predict(embeddings)

        if len(np.unique(cluster_labels)) > 1:
            score = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append((k, score))
        else:
            silhouette_scores.append((k, 0))

    return silhouette_scores

def gap_statistic_analysis(embeddings, k_range=(2, 50), n_bootstrap=10):
    """
    Compute gap statistic for optimal cluster number.
    """
    def compute_inertia(data, labels):
        """Compute inertia (within-cluster sum of squares)."""
        unique_labels = np.unique(labels)
        inertia = 0
        for label in unique_labels:
            cluster_points = data[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            inertia += np.sum((cluster_points - centroid) ** 2)
        return inertia
    
    k_values = range(k_range[0], min(k_range[1] + 1, len(embeddings) // 2))
    gap_scores = []
    
    for k in k_values:
        if k >= len(embeddings):
            break
            
        # Compute inertia for actual data
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        cluster_labels = clustering.fit_predict(embeddings)
        actual_inertia = compute_inertia(embeddings, cluster_labels)
        
        # Compute expected inertia using bootstrap
        expected_inertias = []
        for _ in range(n_bootstrap):
            # Generate reference data by sampling from uniform distribution
            min_vals = np.min(embeddings, axis=0)
            max_vals = np.max(embeddings, axis=0)
            reference_data = np.random.uniform(min_vals, max_vals, embeddings.shape)
            
            clustering_ref = AgglomerativeClustering(n_clusters=k, linkage='ward')
            ref_labels = clustering_ref.fit_predict(reference_data)
            ref_inertia = compute_inertia(reference_data, ref_labels)
            expected_inertias.append(ref_inertia)
        
        expected_inertia = np.mean(expected_inertias)
        gap = np.log(expected_inertia) - np.log(actual_inertia)
        gap_scores.append((k, gap))
    
    return gap_scores

def elbow_method_analysis(embeddings, k_range=(2, 50)):
    """
    Compute elbow method scores (inertia-based).
    """
    def compute_inertia(data, labels):
        unique_labels = np.unique(labels)
        inertia = 0
        for label in unique_labels:
            cluster_points = data[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            inertia += np.sum((cluster_points - centroid) ** 2)
        return inertia
    
    k_values = range(k_range[0], min(k_range[1] + 1, len(embeddings) // 2))
    elbow_scores = []
    
    for k in k_values:
        if k >= len(embeddings):
            break
            
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        cluster_labels = clustering.fit_predict(embeddings)
        inertia = compute_inertia(embeddings, cluster_labels)
        elbow_scores.append((k, inertia))
    
    return elbow_scores

def calinski_harabasz_analysis(embeddings, k_range=(2, 50)):
    """
    Compute Calinski-Harabasz index for optimal cluster number.
    """
    k_values = range(k_range[0], min(k_range[1] + 1, len(embeddings) // 2))
    ch_scores = []
    
    for k in k_values:
        if k >= len(embeddings):
            break
            
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        cluster_labels = clustering.fit_predict(embeddings)
        
        if len(np.unique(cluster_labels)) > 1:
            score = calinski_harabasz_score(embeddings, cluster_labels)
            ch_scores.append((k, score))
        else:
            ch_scores.append((k, 0))
    
    return ch_scores

def aggregate_cluster_validation(silhouette_scores, gap_scores, elbow_scores, ch_scores, output_folder):
    """
    Aggregate results from multiple validation methods using voting mechanism.
    """
    print("\n--- Aggregating Validation Results ---")
    
    # Extract k values and scores
    k_values = [score[0] for score in silhouette_scores]
    
    # Normalize scores to [0, 1] range for comparison
    silhouette_norm = normalize_scores([score[1] for score in silhouette_scores])
    gap_norm = normalize_scores([score[1] for score in gap_scores])
    elbow_norm = normalize_scores([score[1] for score in elbow_scores], reverse=True)  # Lower is better
    ch_norm = normalize_scores([score[1] for score in ch_scores])
    
    # Compute weighted average (equal weights for now)
    weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights
    aggregated_scores = []
    
    for i, k in enumerate(k_values):
        score = (silhouette_norm[i] * weights[0] + 
                gap_norm[i] * weights[1] + 
                elbow_norm[i] * weights[2] + 
                ch_norm[i] * weights[3])
        aggregated_scores.append((k, score))
    
    # Find optimal k
    optimal_k = max(aggregated_scores, key=lambda x: x[1])[0]
    
    # Save validation results - convert NumPy types to native Python types
    validation_results = {
        'k_values': [int(k) for k in k_values],
        'silhouette_scores': [float(score[1]) for score in silhouette_scores],
        'gap_scores': [float(score[1]) for score in gap_scores],
        'elbow_scores': [float(score[1]) for score in elbow_scores],
        'ch_scores': [float(score[1]) for score in ch_scores],
        'aggregated_scores': [float(score[1]) for score in aggregated_scores],
        'optimal_k': int(optimal_k)
    }
    
    validation_file = os.path.join(output_folder, "cluster_validation_results.json")
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"Validation results saved to: {validation_file}")
    return optimal_k

def normalize_scores(scores, reverse=False):
    """Normalize scores to [0, 1] range."""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    normalized = [(score - min_score) / (max_score - min_score) for score in scores]
    
    if reverse:
        normalized = [1.0 - score for score in normalized]
    
    return normalized

def perform_hierarchical_clustering(embeddings, n_clusters, output_folder):
    """
    Perform hierarchical clustering with optimal parameters.
    """
    print(f"\n--- Performing Hierarchical Clustering (k={n_clusters}) ---")
    
    # Use Ward linkage (minimizes within-cluster variance)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Analyze cluster quality
    cluster_analysis = analyze_cluster_quality(embeddings, cluster_labels)
    
    # Save cluster analysis
    analysis_file = os.path.join(output_folder, "cluster_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(cluster_analysis, f, indent=2)
    
    print(f"Cluster analysis saved to: {analysis_file}")
    return cluster_labels

def analyze_cluster_quality(embeddings, cluster_labels):
    """
    Analyze the quality of clustering results.
    """
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)
    
    # Compute quality metrics
    silhouette_avg = silhouette_score(embeddings, cluster_labels) if n_clusters > 1 else 0
    ch_score = calinski_harabasz_score(embeddings, cluster_labels) if n_clusters > 1 else 0
    
    # Analyze cluster sizes
    cluster_sizes = [int(np.sum(cluster_labels == label)) for label in unique_labels]
    
    analysis = {
        'n_clusters': int(n_clusters),
        'silhouette_score': float(silhouette_avg),
        'calinski_harabasz_score': float(ch_score),
        'cluster_sizes': cluster_sizes,
        'min_cluster_size': int(min(cluster_sizes)),
        'max_cluster_size': int(max(cluster_sizes)),
        'avg_cluster_size': float(np.mean(cluster_sizes)),
        'std_cluster_size': float(np.std(cluster_sizes))
    }
    
    print(f"Clustering Quality Metrics:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Silhouette score: {silhouette_avg:.3f}")
    print(f"  Calinski-Harabasz score: {ch_score:.3f}")
    print(f"  Cluster sizes: {min(cluster_sizes)}-{max(cluster_sizes)} (avg: {np.mean(cluster_sizes):.1f})")
    
    return analysis

def normalize_source(source):
    """
    Normalize source text with minimum normalization:
    - Convert to lowercase
    - Handle plural forms (basic)
    - Remove extra whitespace
    - Exclude sources that are just numbers
    - Exclude sources matching 'turnXnewsY' pattern where X and Y are numbers
    """
    if pd.isna(source) or source == '':
        return ''
    
    # Convert to string and lowercase
    source = str(source).strip().lower()
    
    # Exclude sources that are just numbers (e.g., "2", "123", etc.)
    if source.isdigit():
        return ''
    
    # Exclude sources matching 'turnXnewsY' pattern where X and Y are numbers
    # This regex matches 'turn' followed by any number of digits, then 'news', then any number of digits
    if re.match(r'^turn\d+news\d+$', source):
        return ''
    
    # Basic plural normalization (remove trailing 's' for common cases)
    # This is a simple approach - you might want to use a more sophisticated library
    if source.endswith('s') and len(source) > 3:
        # Don't remove 's' from short words or words ending in 'ss'
        if not source.endswith('ss') and len(source) > 3:
            source = source[:-1]
    
    return source

def process_cluster_sources(cluster_df):
    """
    Process sources for a cluster:
    1. Normalize all sources
    2. Count frequency of each source
    3. Filter sources that appear in more than one entity
    4. Sort by frequency (most to least)
    5. Return comma-separated string
    """
    all_sources = []
    
    # Collect all sources from cluster members
    for _, row in cluster_df.iterrows():
        if 'sources' in row and pd.notna(row['sources']):
            # Split by comma and process each source
            sources = str(row['sources']).split(',')
            for source in sources:
                normalized = normalize_source(source)
                if normalized:  # Only add non-empty sources
                    all_sources.append(normalized)
    
    if not all_sources:
        return ""
    
    # Count frequency of each source
    source_counts = Counter(all_sources)
    
    # Filter sources that appear in more than one entity
    # We need to check how many different entities mention each source
    entity_source_map = {}
    for idx, row in cluster_df.iterrows():
        if 'sources' in row and pd.notna(row['sources']):
            sources = str(row['sources']).split(',')
            for source in sources:
                normalized = normalize_source(source)
                if normalized:
                    if normalized not in entity_source_map:
                        entity_source_map[normalized] = set()
                    entity_source_map[normalized].add(idx)
    
    # Filter sources that appear in more than one entity
    filtered_sources = []
    for source, count in source_counts.items():
        if source in entity_source_map and len(entity_source_map[source]) > 1:
            filtered_sources.append((source, count))
    
    # Sort by frequency (most to least)
    filtered_sources.sort(key=lambda x: x[1], reverse=True)
    
    # Return comma-separated string
    return ', '.join([source for source, _ in filtered_sources])

def process_cluster_type(cluster_df):
    """
    Process type for a cluster:
    Return the most common type value across all cluster members
    """
    if 'type' not in cluster_df.columns:
        return ""
    
    types = []
    for _, row in cluster_df.iterrows():
        if pd.notna(row['type']):
            types.append(str(row['type']).strip())
    
    if not types:
        return ""
    
    # Find most common type
    type_counts = Counter(types)
    most_common_type = type_counts.most_common(1)[0][0]
    
    return most_common_type

def select_cluster_representatives(df, embeddings, cluster_labels, output_folder):
    """
    Select representative criteria for each cluster using centrality measures.
    """
    print(f"\n--- Selecting Cluster Representatives ---")
    
    representatives = []
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        # Get indices of items in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_df = df.iloc[cluster_indices]
        
        # Method 1: Centroid-based selection
        centroid = np.mean(cluster_embeddings, axis=0)
        centroid_distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        centroid_representative_idx = cluster_indices[np.argmin(centroid_distances)]
        
        # Method 2: Connectivity-based selection (average similarity to cluster members)
        cluster_similarity_matrix = cosine_similarity(cluster_embeddings)
        connectivity_scores = np.mean(cluster_similarity_matrix, axis=1)
        connectivity_representative_idx = cluster_indices[np.argmax(connectivity_scores)]
        
        # Method 3: Silhouette-based selection (highest silhouette contribution)
        silhouette_scores = silhouette_samples(embeddings, cluster_labels)
        cluster_silhouettes = silhouette_scores[cluster_indices]
        silhouette_representative_idx = cluster_indices[np.argmax(cluster_silhouettes)]
        
        # Aggregate selection using voting
        votes = [centroid_representative_idx, connectivity_representative_idx, silhouette_representative_idx]
        representative_idx = max(set(votes), key=votes.count)  # Most common vote
        
        # Get the representative item
        representative = df.iloc[representative_idx].copy()
        
        # Process cluster-aggregated fields
        representative['type'] = process_cluster_type(cluster_df)
        representative['sources'] = process_cluster_sources(cluster_df)
        
        # Add cluster metadata
        representative['cluster_id'] = cluster_id
        representative['cluster_size'] = len(cluster_indices)
        representative['selection_method'] = 'aggregated_voting'
        representative['centroid_distance'] = centroid_distances[np.where(cluster_indices == representative_idx)[0][0]]
        representative['connectivity_score'] = connectivity_scores[np.where(cluster_indices == representative_idx)[0][0]]
        representative['silhouette_score'] = silhouette_scores[representative_idx]
        
        representatives.append(representative)
        
        print(f"Cluster {cluster_id} ({len(cluster_indices)} items): {representative['name']}")
        print(f"  Type: {representative['type']}")
        print(f"  Sources: {representative['sources']}")
        print(f"  Selection scores - Centroid: {representative['centroid_distance']:.3f}, "
              f"Connectivity: {representative['connectivity_score']:.3f}, "
              f"Silhouette: {representative['silhouette_score']:.3f}")
    
    # Create DataFrame with representatives
    representatives_df = pd.DataFrame(representatives)
    
    return representatives_df

def run_consolidation(
    df: pd.DataFrame,
    output_folder: str,
    *,
    k_range: tuple[int, int] = (2, 50),
    n_bootstrap: int = 10,
) -> pd.DataFrame:
    """Paper Step 5: embedding, clustering, representative selection."""
    os.makedirs(output_folder, exist_ok=True)
    print(f"Loaded {len(df)} criteria for consolidation")
    df_consolidated = systematic_clustering_approach(
        df,
        output_folder=output_folder,
        k_range=k_range,
        n_bootstrap=n_bootstrap,
    )
    output_path = os.path.join(output_folder, "rc_consolidated.csv")
    df_consolidated.to_csv(output_path, index=False)
    print(f"\nConsolidated criteria saved to: {output_path}")
    print(f"Final count: {len(df_consolidated)} representative criteria")
    return df_consolidated


def main():
    args = parse_args()
    k_min, k_max = map(int, args.k_range.split(","))
    df = pd.read_csv(args.input_file)
    run_consolidation(
        df,
        args.output_folder,
        k_range=(k_min, k_max),
        n_bootstrap=args.n_bootstrap,
    )

if __name__ == "__main__":
    main() 