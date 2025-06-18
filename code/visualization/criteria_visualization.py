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

def merge_exact_name_matches(df):
    """
    First merge criteria that have exactly the same name.
    Keep the description that is most complete (longest).
    """
    print("\nMerging exact name matches...")
    # Group by name
    name_groups = df.groupby('name')
    
    merged_rows = []
    for name, group in name_groups:
        if len(group) > 1:
            print(f"\nFound {len(group)} entries for '{name}':")
            for _, row in group.iterrows():
                print(f"  - {row['description']}")
            
            # Keep the row with the longest description
            best_row = group.iloc[group['description'].str.len().argmax()]
            merged_rows.append(best_row)
            print(f"  Keeping: {best_row['description']}")
        else:
            merged_rows.append(group.iloc[0])
    
    df = pd.DataFrame(merged_rows)
    print(f"\nAfter exact name merging: {len(df)} unique criteria")
    return df

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
            from scipy import stats
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

# --- Step 1: Read CSV ---
df = pd.read_csv('global_grouped_criteria.csv')

# --- Step 2: First merge exact name matches ---
df = merge_exact_name_matches(df)

# --- Step 3: Then merge semantic duplicates ---
#refined_threshold = interactive_threshold_refinement(df, initial_threshold=0.80, output_file='threshold_refinement_responses.csv')
df = merge_duplicates(df, similarity_threshold=0.72)

# --- Step 4: Check for remaining similarities ---
#check_remaining_similarities(df, threshold=0.85)

# --- Step 5: Prepare Text Data ---
texts = (df['name'] + '. ' + df['description']).tolist()
print(len(texts))

# --- Step 6: Embed with Free LLM ---
model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, small, local model
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = normalize(embeddings)

# --- Step 7: Hierarchical Clustering ---
Z = linkage(embeddings, method='ward')

# Calculate similarity threshold based on 3rd quartile
similarities = Z[:, 2]  # Get all similarity values from the linkage matrix
threshold = np.percentile(similarities, 85)  # 3rd quartile
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
cluster_df.to_csv('criteria_clusters.csv', index=False)
print(f"\nSaved cluster assignments to 'criteria_clusters.csv'")

# Print clusters in console
print("\nCluster Assignments:")
print("===================")
for cluster_id in sorted(cluster_df['cluster_id'].unique()):
    cluster_items = cluster_df[cluster_df['cluster_id'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_items)} items):")
    for _, row in cluster_items.iterrows():
        print(f"  - {row['name']}")

# Replace the old plot_circular_dendrogram call with the new visualization
def plot_clusters(Z, labels, names, num_clusters):
    """
    Create a modern visualization of the clusters using seaborn and matplotlib.
    """
    # Set modern style
    plt.style.use('seaborn')
    
    # Create figure with a modern color scheme and adjusted width for label space
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 15))
    
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

# Replace the old plot_circular_dendrogram call with the new visualization
plot_clusters(Z, labels, df['name'].tolist(), num_clusters)
