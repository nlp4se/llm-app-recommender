import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import os
import argparse

# Global array of k values
K_VALUES = [1, 3, 5, 10, 20]

def rbo(list1, list2, p=0.9):
    """
    Calculates Rank-Biased Overlap (RBO) between two lists.
    p is the persistence parameter, giving weight to top-ranked items.
    Returns a value between 0 and 1, where 1 means perfect agreement.
    """
    if not isinstance(list1, list): list1 = list(list1)
    if not isinstance(list2, list): list2 = list(list2)
    
    if not list1 or not list2:
        return 0

    # If lists are identical, return 1
    if list1 == list2:
        return 1.0

    s_len, t_len = len(list1), len(list2)
    max_depth = max(s_len, t_len)
    
    # Calculate agreement at each depth
    agreement = 0.0
    normalization = 0.0
    
    for d in range(1, max_depth + 1):
        set1 = set(list1[:d])
        set2 = set(list2[:d])
        agreement_d = len(set1.intersection(set2)) / d
        agreement += agreement_d * (p ** (d-1))
        normalization += p ** (d-1)

    # Normalize by the sum of weights
    return agreement / normalization if normalization > 0 else 0

def get_ranked_lists(df, model, feature, k):
    """
    Extracts ranked lists of apps for a given model and feature, truncated to k items.
    """
    model_df = df[(df['model'] == model) & (df['feature'] == feature)]
    # Get ranking columns from '1' to k
    rank_cols = [str(i) for i in range(1, k + 1)]
    # Ensure columns exist in dataframe before trying to access them
    rank_cols_exist = [col for col in rank_cols if col in model_df.columns]
    lists = model_df[rank_cols_exist].values.tolist()
    return [[app for app in l if pd.notna(app)] for l in lists]

def plot_heatmap(df, value_col, title, output_path, consistency_type='external', model_order=None):
    """
    Plots a heatmap of consistency scores with values constrained to [0,1].
    """
    if df.empty:
        print(f"Skipping plot for '{title}' as there is no data.")
        return
    
    if consistency_type == 'external':
        # Invert rows and columns: use 'feature' as index and ['model1', 'model2'] as columns
        pivot_df = df.pivot_table(index='feature', columns=['model1', 'model2'], values=value_col)
    else:  # internal
        # For internal consistency, use 'feature' as index and 'model' as columns
        pivot_df = df.pivot_table(index='feature', columns='model', values=value_col)
        
        # Reorder columns to match the original model order if provided
        if model_order is not None:
            # Filter to only include models that exist in the pivot table
            existing_models = [model for model in model_order if model in pivot_df.columns]
            if existing_models:
                pivot_df = pivot_df[existing_models]
    
    # Sort features alphabetically
    pivot_df_sorted = pivot_df.sort_index()
    
    # Add empty row for visual separation
    empty_row = pd.DataFrame([[np.nan] * len(pivot_df_sorted.columns)], 
                           index=[''], 
                           columns=pivot_df_sorted.columns)
    
    # Add average row at the bottom
    overall_avg = pivot_df_sorted.mean(axis=0)
    avg_row = pd.DataFrame([overall_avg], index=['AVERAGE'])
    
    # Concatenate: features + empty row + average
    pivot_df_with_avg = pd.concat([pivot_df_sorted, empty_row, avg_row])
    
    plt.figure(figsize=(max(12, len(pivot_df_with_avg.columns) * 1.5), 8))
    
    # Ensure the heatmap is scaled from 0 to 1
    sns.heatmap(pivot_df_with_avg, annot=True, cmap='YlGnBu', fmt=".2f", 
                vmin=0, vmax=1, cbar_kws={'label': f'{value_col.title()} Score'})
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def analyze_external_consistency(df, features, models, k):
    """
    Analyze consistency between different models (external consistency) for a specific k value.
    """
    jaccard_results = []
    rbo_results = []

    for feature in features:
        model_pairs = list(combinations(models, 2))
        if not model_pairs:
            continue

        for model1, model2 in model_pairs:
            lists1 = get_ranked_lists(df, model1, feature, k)
            lists2 = get_ranked_lists(df, model2, feature, k)

            # Part 1: Jaccard Similarity for set consistency
            # For consistency with RBO, we should compare individual runs, not aggregated sets
            jaccard_scores = []
            if lists1 and lists2:
                for l1 in lists1:
                    for l2 in lists2:
                        set1 = set(l1)
                        set2 = set(l2)
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        jaccard_sim = intersection / union if union != 0 else 0
                        jaccard_scores.append(jaccard_sim)
            
            avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
            jaccard_results.append({'feature': feature, 'model1': model1, 'model2': model2, 'jaccard': avg_jaccard})

            # Part 2: Rank-Biased Overlap (RBO) for rank consistency
            rbo_scores = []
            if lists1 and lists2:
                for l1 in lists1:
                    for l2 in lists2:
                        rbo_scores.append(rbo(l1, l2, p=0.90))
            
            avg_rbo = np.mean(rbo_scores) if rbo_scores else 0
            rbo_results.append({'feature': feature, 'model1': model1, 'model2': model2, 'rbo': avg_rbo})

    return pd.DataFrame(jaccard_results), pd.DataFrame(rbo_results)

def analyze_internal_consistency(df, features, models, k):
    """
    Analyze consistency within each model across different runs (internal consistency) for a specific k value.
    """
    jaccard_results = []
    rbo_results = []

    for model in models:
        for feature in features:
            # Get all ranked lists for this model and feature (different runs)
            model_feature_df = df[(df['model'] == model) & (df['feature'] == feature)]
            
            if len(model_feature_df) < 2:
                continue  # Need at least 2 runs to measure consistency
            
            # Get all ranked lists for this model-feature combination, truncated to k
            rank_cols = [str(i) for i in range(1, k + 1)]
            rank_cols_exist = [col for col in rank_cols if col in model_feature_df.columns]
            all_lists = model_feature_df[rank_cols_exist].values.tolist()
            all_lists = [[app for app in l if pd.notna(app)] for l in all_lists]
            
            if len(all_lists) < 2:
                continue
            
            # Calculate Jaccard similarity between all pairs of runs
            jaccard_scores = []
            rbo_scores = []
            
            for i, j in combinations(range(len(all_lists)), 2):
                list1, list2 = all_lists[i], all_lists[j]
                
                # Jaccard similarity
                set1 = set(list1)
                set2 = set(list2)
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                jaccard_sim = intersection / union if union != 0 else 0
                jaccard_scores.append(jaccard_sim)
                
                # RBO - Fix the missing parameter
                rbo_scores.append(rbo(list1, list2, p=0.90))
            
            # Average the scores across all pairs
            avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
            avg_rbo = np.mean(rbo_scores) if rbo_scores else 0
            
            jaccard_results.append({'feature': feature, 'model': model, 'jaccard': avg_jaccard})
            rbo_results.append({'feature': feature, 'model': model, 'rbo': avg_rbo})

    return pd.DataFrame(jaccard_results), pd.DataFrame(rbo_results)

def analyze_consistency_for_k(df, features, models, k, output_dir, consistency_type):
    """
    Analyze consistency for a specific k value and save results to a subfolder.
    """
    # Create subfolder for this k value
    k_output_dir = os.path.join(output_dir, f'k{k}')
    if not os.path.exists(k_output_dir):
        os.makedirs(k_output_dir)
    
    print(f"\n=== Analyzing consistency for k={k} ===")
    
    # Calculate consistency metrics based on type
    if consistency_type == 'external':
        jaccard_df, rbo_df = analyze_external_consistency(df, features, models, k)
        print(f"--- External Consistency (k={k}): Jaccard Similarity Results (Set Consistency) ---")
        print(jaccard_df)
        
        print(f"\n--- External Consistency (k={k}): Rank-Biased Overlap (RBO) Results (Rank Consistency) ---")
        print(rbo_df)
        
        # Save results to CSV
        jaccard_df.to_csv(os.path.join(k_output_dir, 'external_jaccard_similarity.csv'), index=False)
        rbo_df.to_csv(os.path.join(k_output_dir, 'external_rbo_consistency.csv'), index=False)
        
        # Visualization
        plot_heatmap(jaccard_df, 'jaccard', f'External Consistency (k={k}): Jaccard Similarity between Models across Features', 
                    os.path.join(k_output_dir, 'external_jaccard_similarity_heatmap.png'), 'external')
        plot_heatmap(rbo_df, 'rbo', f'External Consistency (k={k}): Rank-Biased Overlap (RBO) between Models across Features', 
                    os.path.join(k_output_dir, 'external_rbo_consistency_heatmap.png'), 'external')
    
    elif consistency_type == 'internal':
        jaccard_df, rbo_df = analyze_internal_consistency(df, features, models, k)
        print(f"--- Internal Consistency (k={k}): Jaccard Similarity Results (Set Consistency) ---")
        print(jaccard_df)
        
        print(f"\n--- Internal Consistency (k={k}): Rank-Biased Overlap (RBO) Results (Rank Consistency) ---")
        print(rbo_df)
        
        # Save results to CSV
        jaccard_df.to_csv(os.path.join(k_output_dir, 'internal_jaccard_similarity.csv'), index=False)
        rbo_df.to_csv(os.path.join(k_output_dir, 'internal_rbo_consistency.csv'), index=False)
        
        # Visualization
        plot_heatmap(jaccard_df, 'jaccard', f'Internal Consistency (k={k}): Jaccard Similarity within Models across Features', 
                    os.path.join(k_output_dir, 'internal_jaccard_similarity_heatmap.png'), 'internal', models)
        plot_heatmap(rbo_df, 'rbo', f'Internal Consistency (k={k}): Rank-Biased Overlap (RBO) within Models across Features', 
                    os.path.join(k_output_dir, 'internal_rbo_consistency_heatmap.png'), 'internal', models)

def main(input_csv_path='data/output/evaluation/app_rankings.csv', 
         output_dir='data/output/evaluation',
         consistency_type='external',
         k_values=None):
    """
    Main function to run the analysis for multiple k values.
    
    Args:
        input_csv_path (str): Path to the input CSV file containing app rankings
        output_dir (str): Directory to save output files
        consistency_type (str): 'external' for between-model consistency, 'internal' for within-model consistency
        k_values (list): List of k values to analyze. If None, uses the global K_VALUES
    """
    if k_values is None:
        k_values = K_VALUES
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load data
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: The file {input_csv_path} was not found.")
        return

    # 2. Get unique features and models (preserve original order)
    features = df['feature'].unique()
    # Preserve the order of models as they appear in the CSV file
    models = df['model'].unique().tolist()

    # 3. Analyze consistency for each k value
    for k in k_values:
        analyze_consistency_for_k(df, features, models, k, output_dir, consistency_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze consistency of app rankings for different k values')
    parser.add_argument('--input-file', 
                       help='Path to the input CSV file containing app rankings')
    parser.add_argument('--output-dir', 
                       help='Directory to save output files')
    parser.add_argument('--consistency-type', 
                       choices=['external', 'internal'], 
                       default='external',
                       help='Type of consistency analysis: external (between models) or internal (within models)')
    parser.add_argument('--k-values', 
                       nargs='+', 
                       type=int,
                       help='List of k values to analyze (e.g., 1 2 3 5 10 15 20). If not provided, uses default values.')
    
    args = parser.parse_args()
    
    # Use provided k values or default to global K_VALUES
    k_values = args.k_values if args.k_values else None
    
    main(input_csv_path=args.input_file, 
         output_dir=args.output_dir, 
         consistency_type=args.consistency_type,
         k_values=k_values)
