import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import seaborn as sns
import random
import argparse
import sys
from scipy import stats

def merge_duplicates(df, similarity_threshold=0.80):
    """
    Merge duplicated ranking criteria based on semantic similarity.
    
    Args:
        df: DataFrame containing 'name' and 'description' columns
        similarity_threshold: Threshold for considering criteria as duplicates (0-1)
    
    Returns:
        DataFrame with merged criteria
    """
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Combine name and description for each item
    texts = (df['name'] + '. ' + df['description']).tolist()
    
    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = normalize(embeddings)  # Normalize for cosine similarity
    
    # Compute pairwise similarities
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find all pairs above threshold
    similar_pairs = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] >= similarity_threshold:
                similar_pairs.append((i, j, similarity_matrix[i, j]))
    
    # Sort pairs by similarity (highest first)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Create groups using a simple approach
    groups = []
    processed = set()
    
    # First pass: Create groups from pairs
    for i, j, _ in similar_pairs:
        if i in processed and j in processed:
            continue
            
        # Find or create a group for these items
        current_group = None
        for group in groups:
            if i in group or j in group:
                current_group = group
                break
                
        if current_group is None:
            current_group = set()
            groups.append(current_group)
            
        current_group.add(i)
        current_group.add(j)
        processed.add(i)
        processed.add(j)
    
    # Second pass: Add remaining items that are similar to any group member
    for i in range(len(df)):
        if i in processed:
            continue
            
        best_group = None
        best_similarity = 0
        
        for group in groups:
            # Check similarity to any member of the group
            max_similarity = max(similarity_matrix[i, j] for j in group)
            if max_similarity > best_similarity and max_similarity >= similarity_threshold:
                best_similarity = max_similarity
                best_group = group
                
        if best_group is not None:
            best_group.add(i)
            processed.add(i)
    
    # Create final DataFrame with representatives
    merged_rows = []
    
    # Process each group
    for group in groups:
        # Select representative (item with longest description)
        best_idx = max(group, key=lambda x: len(df.iloc[x]['description']))
        merged_rows.append(df.iloc[best_idx])
    
    # Add items that weren't in any group
    for i in range(len(df)):
        if i not in processed:
            merged_rows.append(df.iloc[i])
    
    # Create and return new DataFrame
    return pd.DataFrame(merged_rows)

def check_remaining_similarities(df, threshold=0.80):
    """Check for any remaining similar criteria in the final dataset."""
    texts = (df['name'] + '. ' + df['description']).tolist()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = normalize(embeddings)
    
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find pairs above threshold
    similar_pairs = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] >= threshold:
                similar_pairs.append((i, j, similarity_matrix[i, j]))
    
    if similar_pairs:
        print("\nWARNING: Found similar criteria in final dataset:")
        for i, j, sim in sorted(similar_pairs, key=lambda x: x[2], reverse=True):
            print(f"\nSimilarity: {sim:.3f}")
            print(f"1. {df.iloc[i]['name']}")
            print(f"2. {df.iloc[j]['name']}")
    else:
        print("\nNo similar criteria found above threshold in final dataset.")

def interactive_threshold_refinement(df, initial_threshold=0.50, min_samples=10, confidence_level=0.95, output_file='threshold_refinement_responses.csv'):
    """
    Interactively refine the similarity threshold by asking user feedback on randomly selected pairs.
    
    Args:
        df: DataFrame containing 'name' and 'description' columns
        initial_threshold: Starting similarity threshold (0-1)
        min_samples: Minimum number of samples to collect before checking stability
        confidence_level: Confidence level for threshold stability (0-1)
        output_file: Path to save the CSV file with user responses
    
    Returns:
        Refined similarity threshold
    """
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Combine name and description for each item
    texts = (df['name'] + '. ' + df['description']).tolist()
    
    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = normalize(embeddings)
    
    # Compute pairwise similarities
    similarity_matrix = cosine_similarity(embeddings)
    
    # Store user feedback
    feedback_data = []  # List of (similarity, is_duplicate) tuples
    
    # Create a list to store detailed feedback for CSV
    detailed_feedback = []
    
    # Get all possible pairs and their similarities
    all_pairs = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            all_pairs.append((i, j, similarity_matrix[i, j]))
    
    # Sort pairs by similarity (highest first)
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Start with pairs around the initial threshold
    threshold = initial_threshold
    window_size = 0.1  # Look at pairs within ±0.1 of current threshold
    
    # Create or load existing responses
    try:
        existing_responses = pd.read_csv(output_file)
        print(f"\nLoaded {len(existing_responses)} existing responses from {output_file}")
        # Convert existing responses to the format we need
        for _, row in existing_responses.iterrows():
            feedback_data.append((row['similarity'], row['is_duplicate']))
            detailed_feedback.append({
                'similarity': row['similarity'],
                'is_duplicate': row['is_duplicate'],
                'criterion1_name': row['criterion1_name'],
                'criterion1_description': row['criterion1_description'],
                'criterion2_name': row['criterion2_name'],
                'criterion2_description': row['criterion2_description']
            })
    except FileNotFoundError:
        print(f"\nNo existing responses found. Starting fresh.")
    
    while True:
        # Select pairs around current threshold
        relevant_pairs = [p for p in all_pairs if abs(p[2] - threshold) <= window_size]
        
        if not relevant_pairs:
            print(f"No more pairs found around threshold {threshold:.2f}")
            break
            
        # Randomly select a pair
        pair = random.choice(relevant_pairs)
        i, j, similarity = pair
        
        # Check if this pair has already been rated
        pair_already_rated = any(
            (f['criterion1_name'] == df.iloc[i]['name'] and f['criterion2_name'] == df.iloc[j]['name']) or
            (f['criterion1_name'] == df.iloc[j]['name'] and f['criterion2_name'] == df.iloc[i]['name'])
            for f in detailed_feedback
        )
        
        if pair_already_rated:
            continue
        
        # Display the pair to the user
        print("\n" + "="*80)
        print(f"Similarity: {similarity:.3f}")
        print(f"\nCriterion 1:")
        print(f"Name: {df.iloc[i]['name']}")
        print(f"Description: {df.iloc[i]['description']}")
        print(f"\nCriterion 2:")
        print(f"Name: {df.iloc[j]['name']}")
        print(f"Description: {df.iloc[j]['description']}")
        print("\nAre these criteria duplicates? (Y/N/Q to quit)")
        
        # Get user input
        response = input().strip().upper()
        if response == 'Q':
            break
            
        is_duplicate = response == 'Y'
        feedback_data.append((similarity, is_duplicate))
        
        # Store detailed feedback
        detailed_feedback.append({
            'similarity': similarity,
            'is_duplicate': is_duplicate,
            'criterion1_name': df.iloc[i]['name'],
            'criterion1_description': df.iloc[i]['description'],
            'criterion2_name': df.iloc[j]['name'],
            'criterion2_description': df.iloc[j]['description']
        })
        
        # Save to CSV after each response
        pd.DataFrame(detailed_feedback).to_csv(output_file, index=False)
        print(f"\nResponse saved to {output_file}")
        
        # Check if we have enough samples to assess stability
        if len(feedback_data) >= min_samples:
            # Calculate the optimal threshold based on user feedback
            similarities = np.array([s for s, _ in feedback_data])
            is_duplicate = np.array([d for _, d in feedback_data])
            
            # Find the threshold that maximizes the F1 score
            best_threshold = 0
            best_f1 = 0
            
            for t in np.arange(0.5, 1.0, 0.01):
                predictions = similarities >= t
                tp = np.sum((predictions == 1) & (is_duplicate == 1))
                fp = np.sum((predictions == 1) & (is_duplicate == 0))
                fn = np.sum((predictions == 0) & (is_duplicate == 1))
                
                if tp + fp == 0 or tp + fn == 0:
                    continue
                    
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * (precision * recall) / (precision + recall)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t
            
            threshold = best_threshold
            
            # Calculate confidence interval
            z = stats.norm.ppf((1 + confidence_level) / 2)
            std_err = np.sqrt(np.sum((similarities - threshold)**2) / (len(similarities) - 2))
            margin = z * std_err / np.sqrt(len(similarities))
            
            print(f"\nCurrent threshold: {threshold:.3f} ± {margin:.3f}")
            print(f"Number of samples: {len(feedback_data)}")
            
            # Check if threshold is stable
            if margin < 0.05:  # Threshold is stable if margin is less than 0.05
                print("\nThreshold has stabilized!")
                break
    
    return threshold

def active_learning_threshold_optimization(df, initial_threshold=0.72, window_size=0.05, 
                                         min_samples=20, confidence_level=0.95, 
                                         max_samples=100, feedback_file='active_learning_feedback.csv'):
    """
    Active learning approach to determine optimal similarity threshold.
    
    Args:
        df: DataFrame containing 'name' and 'description' columns
        initial_threshold: Starting similarity threshold
        window_size: Range around threshold to sample from (±window_size)
        min_samples: Minimum number of samples before checking statistical validity
        confidence_level: Confidence level for statistical tests
        max_samples: Maximum number of samples to collect
        feedback_file: Path to save user feedback
    
    Returns:
        Optimal similarity threshold
    """
    print(f"\n=== Active Learning Threshold Optimization ===")
    print(f"Initial threshold: {initial_threshold}")
    print(f"Window size: ±{window_size}")
    print(f"Min samples: {min_samples}")
    print(f"Max samples: {max_samples}")
    print(f"Confidence level: {confidence_level}")
    
    # Initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Combine name and description for each item
    texts = (df['name'] + '. ' + df['description']).tolist()
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = normalize(embeddings)
    
    # Compute pairwise similarities
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Get all possible pairs and their similarities
    all_pairs = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            all_pairs.append((i, j, similarity_matrix[i, j]))
    
    # Sort pairs by similarity (highest first)
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Load existing feedback if available
    feedback_data = []
    detailed_feedback = []
    
    try:
        existing_responses = pd.read_csv(feedback_file)
        print(f"\nLoaded {len(existing_responses)} existing responses from {feedback_file}")
        for _, row in existing_responses.iterrows():
            feedback_data.append((row['similarity'], row['is_duplicate']))
            detailed_feedback.append({
                'similarity': row['similarity'],
                'is_duplicate': row['is_duplicate'],
                'criterion1_name': row['criterion1_name'],
                'criterion1_description': row['criterion1_description'],
                'criterion2_name': row['criterion2_name'],
                'criterion2_description': row['criterion2_description']
            })
    except FileNotFoundError:
        print(f"\nNo existing responses found. Starting fresh.")
    
    # Active learning loop
    current_threshold = initial_threshold
    samples_collected = len(feedback_data)
    
    while samples_collected < max_samples:
        # Select pairs within the window around current threshold
        relevant_pairs = [p for p in all_pairs 
                         if abs(p[2] - current_threshold) <= window_size]
        
        if not relevant_pairs:
            print(f"\nNo more pairs found around threshold {current_threshold:.3f}")
            break
        
        # Randomly select a pair that hasn't been rated
        random.shuffle(relevant_pairs)
        selected_pair = None
        
        for pair in relevant_pairs:
            i, j, similarity = pair
            # Check if this pair has already been rated
            pair_already_rated = any(
                (f['criterion1_name'] == df.iloc[i]['name'] and f['criterion2_name'] == df.iloc[j]['name']) or
                (f['criterion1_name'] == df.iloc[j]['name'] and f['criterion2_name'] == df.iloc[i]['name'])
                for f in detailed_feedback
            )
            
            if not pair_already_rated:
                selected_pair = pair
                break
        
        if selected_pair is None:
            print(f"\nNo unrated pairs found around threshold {current_threshold:.3f}")
            break
        
        i, j, similarity = selected_pair
        
        # Display the pair to the user
        print("\n" + "="*80)
        print(f"Sample {samples_collected + 1}/{max_samples}")
        print(f"Similarity: {similarity:.3f} (target: {current_threshold:.3f} ± {window_size})")
        print(f"\nCriterion 1:")
        print(f"Name: {df.iloc[i]['name']}")
        print(f"Description: {df.iloc[i]['description']}")
        print(f"\nCriterion 2:")
        print(f"Name: {df.iloc[j]['name']}")
        print(f"Description: {df.iloc[j]['description']}")
        print("\nAre these criteria duplicates? (Y/N/Q to quit)")
        
        # Get user input
        response = input().strip().upper()
        if response == 'Q':
            print("\nQuitting active learning. Using current best threshold.")
            break
            
        is_duplicate = response == 'Y'
        feedback_data.append((similarity, is_duplicate))
        
        # Store detailed feedback
        detailed_feedback.append({
            'similarity': similarity,
            'is_duplicate': is_duplicate,
            'criterion1_name': df.iloc[i]['name'],
            'criterion1_description': df.iloc[i]['description'],
            'criterion2_name': df.iloc[j]['name'],
            'criterion2_description': df.iloc[j]['description']
        })
        
        # Save to CSV after each response
        pd.DataFrame(detailed_feedback).to_csv(feedback_file, index=False)
        print(f"\nResponse saved to {feedback_file}")
        
        samples_collected += 1
        
        # Check statistical validity if we have enough samples
        if samples_collected >= min_samples:
            optimal_threshold = find_optimal_threshold(feedback_data)
            
            # Calculate confidence interval for the optimal threshold
            confidence_interval = calculate_threshold_confidence_interval(
                feedback_data, optimal_threshold, confidence_level
            )
            
            print(f"\nCurrent optimal threshold: {optimal_threshold:.3f}")
            print(f"Confidence interval: {confidence_interval[0]:.3f} - {confidence_interval[1]:.3f}")
            print(f"Margin of error: ±{(confidence_interval[1] - confidence_interval[0])/2:.3f}")
            
            # Check if threshold is statistically stable
            margin_of_error = (confidence_interval[1] - confidence_interval[0]) / 2
            if margin_of_error < 0.03:  # Threshold is stable if margin is less than 0.03
                print("\nThreshold has reached statistical stability!")
                current_threshold = optimal_threshold
                break
            
            # Update current threshold for next iteration
            current_threshold = optimal_threshold
    
    # Final optimal threshold calculation
    if feedback_data:
        final_threshold = find_optimal_threshold(feedback_data)
        print(f"\n=== Final Results ===")
        print(f"Total samples collected: {len(feedback_data)}")
        print(f"Optimal threshold: {final_threshold:.3f}")
        
        # Calculate final statistics
        similarities = np.array([s for s, _ in feedback_data])
        is_duplicate = np.array([d for _, d in feedback_data])
        
        # Calculate accuracy at optimal threshold
        predictions = similarities >= final_threshold
        accuracy = np.mean(predictions == is_duplicate)
        
        print(f"Accuracy at optimal threshold: {accuracy:.3f}")
        
        return final_threshold
    else:
        print("\nNo feedback collected. Using initial threshold.")
        return initial_threshold

def find_optimal_threshold(feedback_data):
    """
    Find the optimal threshold that maximizes accuracy.
    
    Args:
        feedback_data: List of (similarity, is_duplicate) tuples
    
    Returns:
        Optimal threshold value
    """
    similarities = np.array([s for s, _ in feedback_data])
    is_duplicate = np.array([d for _, d in feedback_data])
    
    best_threshold = 0.5
    best_accuracy = 0
    
    # Test different thresholds
    for threshold in np.arange(0.5, 1.0, 0.01):
        predictions = similarities >= threshold
        accuracy = np.mean(predictions == is_duplicate)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold

def calculate_threshold_confidence_interval(feedback_data, threshold, confidence_level=0.95):
    """
    Calculate confidence interval for the optimal threshold using bootstrap.
    
    Args:
        feedback_data: List of (similarity, is_duplicate) tuples
        threshold: Current optimal threshold
        confidence_level: Confidence level (0-1)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    similarities = np.array([s for s, _ in feedback_data])
    is_duplicate = np.array([d for _, d in feedback_data])
    
    # Bootstrap approach
    n_bootstrap = 1000
    bootstrap_thresholds = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(len(feedback_data), len(feedback_data), replace=True)
        bootstrap_data = [(similarities[i], is_duplicate[i]) for i in indices]
        bootstrap_threshold = find_optimal_threshold(bootstrap_data)
        bootstrap_thresholds.append(bootstrap_threshold)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_thresholds, lower_percentile)
    upper_bound = np.percentile(bootstrap_thresholds, upper_percentile)
    
    return (lower_bound, upper_bound)

def select_cluster_representatives(df, embeddings, labels, output_file='cluster_representatives.csv'):
    """
    Select a representative ranking criteria for each cluster based on centroid proximity.
    
    Args:
        df: DataFrame containing 'name' and 'description' columns
        embeddings: Normalized embeddings for each criteria
        labels: Cluster labels for each item
        output_file: Path to save the representatives CSV file
    
    Returns:
        DataFrame with cluster representatives
    """
    print(f"\nSelecting cluster representatives...")
    
    representatives = []
    unique_clusters = np.unique(labels)
    
    for cluster_id in unique_clusters:
        # Get indices of items in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        
        # Compute centroid of the cluster
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Find the item closest to the centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = np.argmin(distances)
        
        # Get the original index in the full dataset
        original_idx = cluster_indices[closest_idx]
        
        # Get the representative item
        representative = df.iloc[original_idx].copy()
        representative['cluster_id'] = cluster_id
        representative['cluster_size'] = len(cluster_indices)
        representative['centroid_distance'] = distances[closest_idx]
        
        representatives.append(representative)
        
        print(f"Cluster {cluster_id} ({len(cluster_indices)} items): {representative['name']}")
    
    # Create DataFrame with representatives
    representatives_df = pd.DataFrame(representatives)
    
    # Sort by cluster_id
    representatives_df = representatives_df.sort_values('cluster_id')
    
    # Save to CSV
    representatives_df.to_csv(output_file, index=False)
    print(f"\nSaved cluster representatives to '{output_file}'")
    
    return representatives_df

def main(input_file, similarity_threshold=0.72, output_file='criteria_clusters.csv', 
         use_active_learning=True, feedback_file='active_learning_feedback.csv',
         skip_deduplication=False):
    """
    Main function to process criteria clustering.
    
    Args:
        input_file: Path to the input CSV file
        similarity_threshold: Threshold for merging duplicates
        output_file: Path to save the output CSV file
        use_active_learning: Whether to use active learning for threshold optimization
        feedback_file: Path to save active learning feedback
        skip_deduplication: Whether to skip duplicate removal and go directly to clustering
    """
    # --- Step 1: Read CSV ---
    try:
        # Try reading with comma delimiter first
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} criteria from {input_file}")
    except Exception as e:
        # If that fails, try with semicolon delimiter
        try:
            df = pd.read_csv(input_file, sep=';')
            print(f"Loaded {len(df)} criteria from {input_file} (using semicolon delimiter)")
        except Exception as e2:
            print(f"Error reading file '{input_file}': {e2}")
            sys.exit(1)

    # --- Step 2: Active Learning Threshold Optimization (only if not skipping deduplication) ---
    if not skip_deduplication:
        if use_active_learning:
            optimal_threshold = active_learning_threshold_optimization(
                df, 
                initial_threshold=similarity_threshold,
                feedback_file=feedback_file
            )
            print(f"\nUsing optimal threshold: {optimal_threshold:.3f}")
        else:
            optimal_threshold = similarity_threshold
            print(f"\nUsing provided threshold: {optimal_threshold:.3f}")

        # --- Step 3: Merge semantic duplicates ---
        print(f"\nMerging duplicates with threshold: {optimal_threshold:.3f}")
        df = merge_duplicates(df, similarity_threshold=optimal_threshold)
        print(f"After deduplication: {len(df)} criteria remaining")
    else:
        print(f"\nSkipping deduplication step. Proceeding directly to clustering with {len(df)} criteria.")

    # --- Step 4: Prepare Text Data ---
    texts = (df['name'] + '. ' + df['description']).tolist()
    print(f"Preparing embeddings for {len(texts)} criteria")

    # --- Step 5: Embed with Free LLM ---
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, small, local model
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = normalize(embeddings)

    # --- Step 6: Hierarchical Clustering ---
    Z = linkage(embeddings, method='ward')

    # Calculate similarity threshold based on 3rd quartile
    similarities = Z[:, 2]  # Get all similarity values from the linkage matrix
    threshold = np.percentile(similarities, 78)  # 3rd quartile
    labels = fcluster(Z, threshold, criterion='distance')

    # Get the actual number of clusters
    num_clusters = len(np.unique(labels))
    print(f"\nNumber of clusters based on 3rd quartile threshold: {num_clusters}")

    # Create a DataFrame with cluster assignments
    cluster_df = pd.DataFrame({
        'name': df['name'],
        'description': df['description'],
        'cluster_id': labels
    })

    # Sort by cluster_id and name for better readability
    cluster_df = cluster_df.sort_values(['cluster_id', 'name'])

    # Save to CSV
    cluster_df.to_csv(output_file, index=False)
    print(f"\nSaved cluster assignments to '{output_file}'")

    # Print clusters in console
    print("\nCluster Assignments:")
    print("===================")
    for cluster_id in sorted(cluster_df['cluster_id'].unique()):
        cluster_items = cluster_df[cluster_df['cluster_id'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_items)} items):")
        for _, row in cluster_items.iterrows():
            print(f"  - {row['name']}")

    # --- Step 7: Select Cluster Representatives (only when skipping deduplication) ---
    if skip_deduplication:
        # Generate output filename for representatives
        base_name = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
        representatives_file = f"{base_name}_representatives.csv"
        
        representatives_df = select_cluster_representatives(
            df, embeddings, labels, representatives_file
        )
        
        print(f"\nCluster Representatives Summary:")
        print("================================")
        for _, row in representatives_df.iterrows():
            print(f"Cluster {row['cluster_id']}: {row['name']} (size: {row['cluster_size']}, distance: {row['centroid_distance']:.4f})")

    # Create visualization - pass the threshold for the cut-off line
    plot_clusters(Z, labels, df['name'].tolist(), num_clusters, threshold)

def plot_clusters(Z, labels, names, num_clusters, cluster_threshold):
    """
    Create a modern visualization of the clusters using seaborn and matplotlib.
    
    Args:
        Z: Linkage matrix from hierarchical clustering
        labels: Cluster labels for each item
        names: Names of the items
        num_clusters: Number of clusters
        cluster_threshold: Distance threshold used for clustering (for cut-off line)
    """
    # Set modern style
    plt.style.use('seaborn')
    
    # Create figure with a modern color scheme and adjusted width for label space
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 12))
    
    # Create a consistent color palette for both plots
    colors = sns.color_palette("husl", n_colors=num_clusters)
    
    # Plot 1: Dendrogram
    # Create a custom color map for the dendrogram that matches the bar colors
    color_map = {i: colors[i-1] for i in range(1, num_clusters + 1)}
    
    # Calculate the color threshold to match our number of clusters
    max_d = max(Z[:, 2])
    color_threshold = max_d * 0.7  # Adjust this value to get desired number of clusters
    
    # Create a custom color function for the dendrogram
    def color_func(x):
        if x >= color_threshold:
            return 'gray'
        # Get the cluster ID for this leaf
        leaf_idx = int(x)
        if leaf_idx < len(labels):
            cluster_id = labels[leaf_idx]
            return color_map[cluster_id]
        return 'gray'
    
    # First create the dendrogram without colors
    dendro = dendrogram(Z, labels=names, orientation='right', ax=ax1,
                        leaf_font_size=10, color_threshold=color_threshold)
    
    # Now color the leaves and links
    for i, d in enumerate(dendro['leaves']):
        cluster_id = labels[d]
        ax1.get_yticklabels()[i].set_color(color_map[cluster_id])
    
    # Color the links
    for i, d in enumerate(dendro['color_list']):
        if d == 'gray':
            continue
        # Get the cluster ID for this leaf
        leaf_idx = dendro['leaves'][i]
        cluster_id = labels[leaf_idx]
        dendro['color_list'][i] = color_map[cluster_id]
    
    # Add vertical cut-off line to show where clusters are split
    # Get the y-axis limits to draw the line across the full height
    y_min, y_max = ax1.get_ylim()
    
    # Draw the cut-off line
    ax1.axvline(x=cluster_threshold, color='red', linestyle='--', linewidth=2, 
                alpha=0.8, label=f'Cut-off (threshold: {cluster_threshold:.2f})')
    
    # Add legend for the cut-off line
    ax1.legend(loc='upper right', fontsize=10)
    
    ax1.set_title('Hierarchical Clustering Dendrogram', pad=20, fontsize=12)
    
    # Adjust margins to prevent label cropping
    plt.subplots_adjust(left=0.3, right=0.95)
    
    # Plot 2: Cluster Distribution
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    
    bars = ax2.bar(range(num_clusters), cluster_counts.values, color=colors)
    ax2.set_title('Distribution of Criteria Across Clusters', pad=20, fontsize=12)
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Criteria')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Customize appearance
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure y-axis labels are fully visible
    ax1.tick_params(axis='y', labelsize=10)
    for label in ax1.get_yticklabels():
        label.set_horizontalalignment('right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster ranking criteria using semantic similarity')
    parser.add_argument('--input-file', '-i', 
                       default='global_grouped_criteria.csv',
                       help='Path to the input CSV file (default: global_grouped_criteria.csv)')
    parser.add_argument('--similarity-threshold', '-t',
                       type=float, default=0.72,
                       help='Similarity threshold for merging duplicates (default: 0.72)')
    parser.add_argument('--output-file', '-o',
                       default='criteria_clusters.csv',
                       help='Path to save the output CSV file (default: criteria_clusters.csv)')
    parser.add_argument('--no-active-learning', action='store_true',
                       help='Skip active learning and use provided threshold directly')
    parser.add_argument('--no-deduplication', action='store_true',
                       help='Skip duplicate removal and go directly to hierarchical clustering')
    parser.add_argument('--feedback-file', '-f',
                       default='active_learning_feedback.csv',
                       help='Path to save active learning feedback (default: active_learning_feedback.csv)')
    
    args = parser.parse_args()
    
    main(args.input_file, args.similarity_threshold, args.output_file, 
         use_active_learning=not args.no_active_learning, 
         feedback_file=args.feedback_file,
         skip_deduplication=args.no_deduplication)
