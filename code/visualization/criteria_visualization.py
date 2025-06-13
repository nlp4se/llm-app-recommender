import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

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

def merge_duplicates(df, similarity_threshold=0.85):
    """
    Merge duplicated ranking criteria based on semantic similarity.
    
    Args:
        df: DataFrame containing 'name' and 'description' columns
        similarity_threshold: Threshold for considering criteria as duplicates (0-1)
    
    Returns:
        DataFrame with merged criteria
    """
    # Prepare text data
    texts = (df['name'] + '. ' + df['description']).tolist()
    
    # Get embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = normalize(embeddings)
    
    # Calculate pairwise similarities
    similarity_matrix = cosine_similarity(embeddings)
    
    # Plot similarity distribution
    plt.figure(figsize=(10, 6))
    similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    plt.hist(similarities, bins=50)
    plt.axvline(x=similarity_threshold, color='r', linestyle='--', label=f'Threshold ({similarity_threshold})')
    plt.title('Distribution of Pairwise Similarities')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    
    # Find groups of similar criteria
    groups = defaultdict(list)
    processed = set()
    
    for i in range(len(similarity_matrix)):
        if i in processed:
            continue
            
        current_group = [i]
        processed.add(i)
        
        for j in range(len(similarity_matrix)):
            if j not in processed and similarity_matrix[i, j] >= similarity_threshold:
                current_group.append(j)
                processed.add(j)
        
        if len(current_group) > 1:  # Only store groups with duplicates
            groups[i] = current_group
    
    # Print and merge groups
    if groups:
        print(f"\nFound {len(groups)} groups of similar criteria:")
        merged_rows = []
        
        for leader_idx, group in groups.items():
            print(f"\nGroup {leader_idx + 1}:")
            for idx in group:
                print(f"  - {df.iloc[idx]['name']}")
            
            # Keep the row with the most complete description
            best_idx = max(group, key=lambda x: len(df.iloc[x]['description']))
            merged_rows.append(df.iloc[best_idx])
        
        # Add all non-grouped rows
        non_grouped = [i for i in range(len(df)) if i not in processed]
        for idx in non_grouped:
            merged_rows.append(df.iloc[idx])
        
        # Create new DataFrame
        df = pd.DataFrame(merged_rows)
        print(f"\nFinal number of unique criteria: {len(df)}")
    
    return df

def check_remaining_similarities(df, threshold=0.85):
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

# --- Step 1: Read CSV ---
df = pd.read_csv('data/output/category/rq1/gemini/ranking-criteria-all.csv')

# --- Step 2: First merge exact name matches ---
df = merge_exact_name_matches(df)

# --- Step 3: Then merge semantic duplicates ---
df = merge_duplicates(df, similarity_threshold=0.85)

# --- Step 4: Check for remaining similarities ---
check_remaining_similarities(df, threshold=0.85)

# --- Step 5: Prepare Text Data ---
texts = (df['name'] + '. ' + df['description']).tolist()

# --- Step 6: Embed with Free LLM ---
model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, small, local model
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = normalize(embeddings)

# --- Step 7: Hierarchical Clustering ---
Z = linkage(embeddings, method='ward')

# Choose number of clusters (e.g., 6 for color diversity)
num_clusters = 6
labels = fcluster(Z, num_clusters, criterion='maxclust')

# --- Step 8: Circular Dendrogram Plot ---
def plot_circular_dendrogram(Z, labels, names, num_clusters):
    # Create dendrogram data
    dendro = dendrogram(Z, labels=names, orientation='right', no_plot=True)
    icoord = np.array(dendro['icoord'])
    dcoord = np.array(dendro['dcoord'])
    leaves = dendro['leaves']
    leaf_labels = np.array(names)[leaves]
    cluster_labels = np.array(labels)[leaves]

    # Angles for circular layout
    theta = np.linspace(0, 2 * np.pi, len(leaves), endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'polar': True})
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))

    for i, (angle, label, cluster) in enumerate(zip(theta, leaf_labels, cluster_labels)):
        ax.text(angle, 1.1, label, color=colors[cluster % num_clusters], ha='center', va='center', fontsize=8, rotation=np.degrees(angle), rotation_mode='anchor')

    # Optionally, plot lines for clusters (not as detailed as the original, but gives a circular effect)
    for i, (angle, cluster) in enumerate(zip(theta, cluster_labels)):
        ax.plot([angle, angle], [0, 1], color=colors[cluster % num_clusters], lw=2, alpha=0.5)

    ax.set_axis_off()
    plt.title('Circular Dendrogram of Criteria', fontsize=16)
    plt.tight_layout()
    plt.show()

plot_circular_dendrogram(Z, labels, df['name'].tolist(), num_clusters)
