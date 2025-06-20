import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import os
import argparse

def rbo(list1, list2, p=0.9):
    """
    Calculates Rank-Biased Overlap (RBO) between two lists.
    p is the persistence parameter, giving weight to top-ranked items.
    """
    if not isinstance(list1, list): list1 = list(list1)
    if not isinstance(list2, list): list2 = list(list2)
    
    if not list1 or not list2:
        return 0

    s_len, t_len = len(list1), len(list2)
    max_depth = max(s_len, t_len)
    
    # Calculate agreement at each depth
    agreement = 0.0
    
    for d in range(1, max_depth + 1):
        set1 = set(list1[:d])
        set2 = set(list2[:d])
        agreement_d = len(set1.intersection(set2)) / d
        agreement += agreement_d * (p ** (d-1))

    return (1 - p) * agreement

def get_ranked_lists(df, model, feature):
    """
    Extracts ranked lists of apps for a given model and feature.
    """
    model_df = df[(df['model'] == model) & (df['feature'] == feature)]
    # Assuming ranking columns are from '1' to '20'
    rank_cols = [str(i) for i in range(1, 21)]
    # Ensure columns exist in dataframe before trying to access them
    rank_cols_exist = [col for col in rank_cols if col in model_df.columns]
    lists = model_df[rank_cols_exist].values.tolist()
    return [[app for app in l if pd.notna(app)] for l in lists]

def plot_heatmap(df, value_col, title, output_path, consistency_type='external', model_order=None):
    """
    Plots a heatmap of consistency scores.
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
    sns.heatmap(pivot_df_with_avg, annot=True, cmap='YlGnBu', fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def analyze_external_consistency(df, features, models):
    """
    Analyze consistency between different models (external consistency).
    """
    jaccard_results = []
    rbo_results = []

    for feature in features:
        model_pairs = list(combinations(models, 2))
        if not model_pairs:
            continue

        for model1, model2 in model_pairs:
            lists1 = get_ranked_lists(df, model1, feature)
            lists2 = get_ranked_lists(df, model2, feature)

            # Part 1: Jaccard Similarity for set consistency
            set1 = set(app for l in lists1 for app in l)
            set2 = set(app for l in lists2 for app in l)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            jaccard_sim = intersection / union if union != 0 else 0
            jaccard_results.append({'feature': feature, 'model1': model1, 'model2': model2, 'jaccard': jaccard_sim})

            # Part 2: Rank-Biased Overlap (RBO) for rank consistency
            rbo_scores = []
            if lists1 and lists2:
                for l1 in lists1:
                    for l2 in lists2:
                        rbo_scores.append(rbo(l1, l2, p=0.95)) # Using p=0.95 gives more weight to top ranks
            
            avg_rbo = np.mean(rbo_scores) if rbo_scores else 0
            rbo_results.append({'feature': feature, 'model1': model1, 'model2': model2, 'rbo': avg_rbo})

    return pd.DataFrame(jaccard_results), pd.DataFrame(rbo_results)

def analyze_internal_consistency(df, features, models):
    """
    Analyze consistency within each model across different runs (internal consistency).
    """
    jaccard_results = []
    rbo_results = []

    for model in models:
        for feature in features:
            # Get all ranked lists for this model and feature (different runs)
            model_feature_df = df[(df['model'] == model) & (df['feature'] == feature)]
            
            if len(model_feature_df) < 2:
                continue  # Need at least 2 runs to measure consistency
            
            # Get all ranked lists for this model-feature combination
            rank_cols = [str(i) for i in range(1, 21)]
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
                
                # RBO
                rbo_scores.append(rbo(list1, list2, p=0.95))
            
            # Average the scores across all pairs
            avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
            avg_rbo = np.mean(rbo_scores) if rbo_scores else 0
            
            jaccard_results.append({'feature': feature, 'model': model, 'jaccard': avg_jaccard})
            rbo_results.append({'feature': feature, 'model': model, 'rbo': avg_rbo})

    return pd.DataFrame(jaccard_results), pd.DataFrame(rbo_results)

def main(input_csv_path='data/output/evaluation/app_rankings.csv', 
         output_dir='data/output/evaluation',
         consistency_type='external'):
    """
    Main function to run the analysis.
    
    Args:
        input_csv_path (str): Path to the input CSV file containing app rankings
        output_dir (str): Directory to save output files
        consistency_type (str): 'external' for between-model consistency, 'internal' for within-model consistency
    """
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

    # 3. Calculate consistency metrics based on type
    if consistency_type == 'external':
        jaccard_df, rbo_df = analyze_external_consistency(df, features, models)
        print("--- External Consistency: Jaccard Similarity Results (Set Consistency) ---")
        print(jaccard_df)
        
        print("\n--- External Consistency: Rank-Biased Overlap (RBO) Results (Rank Consistency) ---")
        print(rbo_df)
        
        # 4. Visualization
        plot_heatmap(jaccard_df, 'jaccard', 'External Consistency: Jaccard Similarity between Models across Features', 
                    os.path.join(output_dir, 'external_jaccard_similarity_heatmap.png'), 'external')
        plot_heatmap(rbo_df, 'rbo', 'External Consistency: Rank-Biased Overlap (RBO) between Models across Features', 
                    os.path.join(output_dir, 'external_rbo_consistency_heatmap.png'), 'external')
    
    elif consistency_type == 'internal':
        jaccard_df, rbo_df = analyze_internal_consistency(df, features, models)
        print("--- Internal Consistency: Jaccard Similarity Results (Set Consistency) ---")
        print(jaccard_df)
        
        print("\n--- Internal Consistency: Rank-Biased Overlap (RBO) Results (Rank Consistency) ---")
        print(rbo_df)
        
        # 4. Visualization
        plot_heatmap(jaccard_df, 'jaccard', 'Internal Consistency: Jaccard Similarity within Models across Features', 
                    os.path.join(output_dir, 'internal_jaccard_similarity_heatmap.png'), 'internal', models)
        plot_heatmap(rbo_df, 'rbo', 'Internal Consistency: Rank-Biased Overlap (RBO) within Models across Features', 
                    os.path.join(output_dir, 'internal_rbo_consistency_heatmap.png'), 'internal', models)
    
    else:
        print(f"Error: Invalid consistency_type '{consistency_type}'. Must be 'external' or 'internal'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze consistency of app rankings')
    parser.add_argument('--input-file', 
                       help='Path to the input CSV file containing app rankings')
    parser.add_argument('--output-dir', 
                       help='Directory to save output files')
    parser.add_argument('--consistency-type', 
                       choices=['external', 'internal'], 
                       default='external',
                       help='Type of consistency analysis: external (between models) or internal (within models)')
    
    args = parser.parse_args()
    main(input_csv_path=args.input_file, 
         output_dir=args.output_dir, 
         consistency_type=args.consistency_type)
