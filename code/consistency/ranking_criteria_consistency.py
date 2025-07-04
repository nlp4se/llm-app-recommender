import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import os
import argparse
import re
import hashlib
import json
from sentence_transformers import SentenceTransformer, util

def get_set_hash(set1, set2):
    """
    Creates a hash for a pair of sets to use as cache key.
    The hash is order-independent and deterministic.
    """
    # Convert sets to sorted lists for consistent hashing
    list1 = sorted(list(set1))
    list2 = sorted(list(set2))
    
    # Create a string representation and hash it
    combined = f"{str(list1)}|||{str(list2)}"
    return hashlib.md5(combined.encode()).hexdigest()

def load_similarity_cache(cache_file):
    """
    Load similarity cache from file.
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load cache from {cache_file}")
            return {}
    return {}

def save_similarity_cache(cache, cache_file):
    """
    Save similarity cache to file.
    """
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"Saved similarity cache to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_file}: {e}")

def soft_jaccard_similarity(set1, set2, model, cache=None, cache_file=None):
    """
    Calculates a 'soft' Jaccard similarity between two sets of strings based on semantic similarity.
    Uses caching to avoid recomputing the same similarities.
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    # Check cache first
    if cache is not None:
        set_hash = get_set_hash(set1, set2)
        if set_hash in cache:
            return cache[set_hash]

    # Compute similarity if not in cache
    list1 = list(set1)
    list2 = list(set2)

    embeddings1 = model.encode(list1, convert_to_tensor=True)
    embeddings2 = model.encode(list2, convert_to_tensor=True)

    cos_sim = util.cos_sim(embeddings1, embeddings2)

    intersection_score = 0.0
    used_indices1 = set()
    used_indices2 = set()
    
    all_pairs = []
    for i in range(len(list1)):
        for j in range(len(list2)):
            all_pairs.append({'score': cos_sim[i][j].item(), 'i': i, 'j': j})
            
    all_pairs = sorted(all_pairs, key=lambda x: x['score'], reverse=True)
    
    for pair in all_pairs:
        if pair['i'] not in used_indices1 and pair['j'] not in used_indices2:
            intersection_score += pair['score']
            used_indices1.add(pair['i'])
            used_indices2.add(pair['j'])

    denominator = len(set1) + len(set2) - intersection_score
    if denominator == 0:
        result = 1.0
    else:
        result = intersection_score / denominator

    # Store result in cache
    if cache is not None:
        set_hash = get_set_hash(set1, set2)
        cache[set_hash] = result
        
        # Save cache periodically (every 100 new entries)
        if len(cache) % 100 == 0 and cache_file is not None:
            save_similarity_cache(cache, cache_file)

    return result

def parse_sources(sources_series):
    """
    Parses a series of comma-separated sources, filters out unwanted patterns,
    and returns a set of unique sources.
    """
    sources = set()
    # Pattern to ignore strings like 'turnXsearchY' or 'turnXnewsY'
    ignore_pattern = re.compile(r'turn\d+(search|news)\d+')
    
    for s in sources_series.dropna():
        # Each 's' can be a comma-separated string of sources
        items = [item.strip() for item in s.split(',')]
        for item in items:
            if item and not ignore_pattern.match(item):
                sources.add(item)
    return sources

def get_criteria_and_sources_for_hard_sim(df_subset):
    """
    Extracts sets for HARD similarity (name+desc are tuples).
    """
    if df_subset.empty:
        return set(), set()
    
    criteria_name_desc = set(zip(df_subset['n'].dropna(), df_subset['d'].dropna()))
    sources = parse_sources(df_subset['s'])
    
    return criteria_name_desc, sources

def get_criteria_and_sources_for_soft_sim(df_subset):
    """
    Extracts sets for SOFT similarity (name+desc are concatenated strings).
    """
    if df_subset.empty:
        return set(), set()
    
    df_nd = df_subset[['n', 'd']].dropna()
    criteria_name_desc = set(df_nd['n'] + ': ' + df_nd['d'])
    sources = parse_sources(df_subset['s'])
    
    return criteria_name_desc, sources

def rename_feature_for_display(feature_name):
    """
    Rename features for better display in plots.
    """
    if feature_name == 'Broadcast messages to multiple contacts':
        return 'Broadcast messages'
    return feature_name

def plot_side_by_side_heatmaps(internal_df, external_df, value_col, title, output_path, similarity_type, models):
    """
    Creates side-by-side heatmaps: internal consistency on the left, external on the right.
    """
    if internal_df.empty and external_df.empty:
        print(f"Skipping plot for '{title}' as there is no data.")
        return
    
    # Set font sizes for better readability
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [0.8, 1.5]})
    
    # Process internal consistency data (left plot)
    if not internal_df.empty:
        pivot_internal = internal_df.pivot_table(index='feature', columns='model', values=value_col)
        if models is not None:
            existing_models = [model for model in models if model in pivot_internal.columns]
            if existing_models:
                pivot_internal = pivot_internal[existing_models]
        
        # Rename features for display
        pivot_internal.index = pivot_internal.index.map(rename_feature_for_display)
        pivot_internal_sorted = pivot_internal.sort_index()
        
        # Add empty row and average row
        empty_row = pd.DataFrame([[np.nan] * len(pivot_internal_sorted.columns)], 
                               index=[''], 
                               columns=pivot_internal_sorted.columns)
        overall_avg = pivot_internal_sorted.mean(axis=0)
        avg_row = pd.DataFrame([overall_avg], index=['AVERAGE'])
        pivot_internal_with_avg = pd.concat([pivot_internal_sorted, empty_row, avg_row])
        
        # Plot internal consistency (left)
        sns.heatmap(pivot_internal_with_avg, annot=True, cmap='YlGnBu', fmt=".2f", 
                    vmin=0, vmax=1, cbar=False, ax=ax1, annot_kws={'size': 14})
        ax1.set_title('Internal Consistency', fontsize=16, pad=20)
        ax1.set_xlabel('models', fontsize=16)
        #ax1.set_ylabel('Features', fontsize=16)
        
        # Rotate x-axis labels for better readability
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=14)
        ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=14)
    
    # Process external consistency data (right plot)
    if not external_df.empty:
        pivot_external = external_df.pivot_table(index='feature', columns=['model1', 'model2'], values=value_col)
        
        # Rename features for display
        pivot_external.index = pivot_external.index.map(rename_feature_for_display)
        pivot_external_sorted = pivot_external.sort_index()
        
        # Add empty row and average row
        empty_row = pd.DataFrame([[np.nan] * len(pivot_external_sorted.columns)], 
                               index=[''], 
                               columns=pivot_external_sorted.columns)
        overall_avg = pivot_external_sorted.mean(axis=0)
        avg_row = pd.DataFrame([overall_avg], index=['AVERAGE'])
        pivot_external_with_avg = pd.concat([pivot_external_sorted, empty_row, avg_row])
        
        # Plot external consistency (right) - without y-axis labels
        sns.heatmap(pivot_external_with_avg, annot=True, cmap='YlGnBu', fmt=".2f", 
                    vmin=0, vmax=1, cbar=True, ax=ax2, annot_kws={'size': 14},
                    cbar_kws={'label': f'{value_col.title()} Score'})
        ax2.set_title('External Consistency', fontsize=16, pad=20)
        ax2.set_xlabel('model-model', fontsize=14)
        ax2.set_ylabel('')  # No y-axis label for right plot
        
        # Rotate x-axis labels vertically for better fit
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, ha='center', fontsize=15)
        ax2.set_yticklabels([])  # Hide y-axis labels on right plot
    
    # Adjust layout to ensure both plots have the same height
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved side-by-side plot to {output_path}")

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
    
    # Rename features for display
    pivot_df.index = pivot_df.index.map(rename_feature_for_display)
    
    # Sort features alphabetically instead of by average metric value
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
    
    # Set larger figure size and font sizes
    plt.figure(figsize=(max(7, len(pivot_df_with_avg.columns) * 0.8), 8))
    
    # Set font sizes for better readability
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Ensure the heatmap is scaled from 0 to 1 with larger annotation font
    sns.heatmap(pivot_df_with_avg, annot=True, cmap='YlGnBu', fmt=".2f", 
                vmin=0, vmax=1, cbar_kws={'label': f'{value_col.title()} Score'},
                annot_kws={'size': 12})  # Larger annotation font
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")
    plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=0, ha='center', fontsize=12)
    plt.gca().set_yticklabels(plt.gca().get_yticklabels(), fontsize=12)
    plt.gca().xaxis.label.set_size(14)

def analyze_external_consistency_hard(df, features, models):
    criteria_name_desc_results, sources_results = [], []
    for feature in features:
        for model1, model2 in combinations(models, 2):
            df1 = df[(df['model'] == model1) & (df['feature'] == feature)]
            df2 = df[(df['model'] == model2) & (df['feature'] == feature)]
            nd1, src1 = get_criteria_and_sources_for_hard_sim(df1)
            nd2, src2 = get_criteria_and_sources_for_hard_sim(df2)
            
            # Standard Jaccard
            def jaccard(s1, s2):
                return len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0
                
            criteria_name_desc_results.append({'feature': feature, 'model1': model1, 'model2': model2, 'jaccard': jaccard(nd1, nd2)})
            sources_results.append({'feature': feature, 'model1': model1, 'model2': model2, 'jaccard': jaccard(src1, src2)})
    return pd.DataFrame(criteria_name_desc_results), pd.DataFrame(sources_results)

def analyze_internal_consistency_hard(df, features, models):
    criteria_name_desc_results, sources_results = [], []
    for model in models:
        for feature in features:
            model_feature_df = df[(df['model'] == model) & (df['feature'] == feature)]
            runs = model_feature_df['run'].unique()
            if len(runs) < 2: continue
            
            run_data = {run: get_criteria_and_sources_for_hard_sim(model_feature_df[model_feature_df['run'] == run]) for run in runs}
            
            def jaccard(s1, s2):
                return len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0

            nd_scores, src_scores = [], []
            for (r1, (nd1, src1)), (r2, (nd2, src2)) in combinations(run_data.items(), 2):
                nd_scores.append(jaccard(nd1, nd2))
                src_scores.append(jaccard(src1, src2))
                
            criteria_name_desc_results.append({'feature': feature, 'model': model, 'jaccard': np.mean(nd_scores) if nd_scores else 0})
            sources_results.append({'feature': feature, 'model': model, 'jaccard': np.mean(src_scores) if src_scores else 0})
    return pd.DataFrame(criteria_name_desc_results), pd.DataFrame(sources_results)

def analyze_external_consistency_soft(df, features, models, embedding_model, cache=None, cache_file=None):
    criteria_name_desc_results, sources_results = [], []
    for feature in features:
        for model1, model2 in combinations(models, 2):
            df1 = df[(df['model'] == model1) & (df['feature'] == feature)]
            df2 = df[(df['model'] == model2) & (df['feature'] == feature)]
            nd1, src1 = get_criteria_and_sources_for_soft_sim(df1)
            nd2, src2 = get_criteria_and_sources_for_soft_sim(df2)
            
            criteria_name_desc_results.append({
                'feature': feature, 
                'model1': model1, 
                'model2': model2, 
                'jaccard': soft_jaccard_similarity(nd1, nd2, embedding_model, cache, cache_file)
            })
            src_union = src1.union(src2)
            sources_results.append({
                'feature': feature, 
                'model1': model1, 
                'model2': model2, 
                'jaccard': len(src1.intersection(src2)) / len(src_union) if src_union else 0
            })
    return pd.DataFrame(criteria_name_desc_results), pd.DataFrame(sources_results)

def analyze_internal_consistency_soft(df, features, models, embedding_model, cache=None, cache_file=None):
    criteria_name_desc_results, sources_results = [], []
    for model in models:
        for feature in features:
            model_feature_df = df[(df['model'] == model) & (df['feature'] == feature)]
            runs = model_feature_df['run'].unique()
            if len(runs) < 2: continue

            run_data = {run: get_criteria_and_sources_for_soft_sim(model_feature_df[model_feature_df['run'] == run]) for run in runs}
            
            nd_scores, src_scores = [], []
            for (r1, (nd1, src1)), (r2, (nd2, src2)) in combinations(run_data.items(), 2):
                nd_scores.append(soft_jaccard_similarity(nd1, nd2, embedding_model, cache, cache_file))
                src_union = src1.union(src2)
                src_scores.append(len(src1.intersection(src2)) / len(src_union) if src_union else 0)

            criteria_name_desc_results.append({'feature': feature, 'model': model, 'jaccard': np.mean(nd_scores) if nd_scores else 0})
            sources_results.append({'feature': feature, 'model': model, 'jaccard': np.mean(src_scores) if src_scores else 0})
    return pd.DataFrame(criteria_name_desc_results), pd.DataFrame(sources_results)

def main(input_csv_path, output_dir, consistency_type, similarity_type):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    embedding_model = None
    cache = None
    cache_file = None
    
    if similarity_type == 'soft':
        print("Loading sentence embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")
        
        # Set up caching
        cache_file = os.path.join(output_dir, 'similarity_cache.json')
        cache = load_similarity_cache(cache_file)
        print(f"Loaded {len(cache)} cached similarity scores from {cache_file}")

    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: The file {input_csv_path} was not found.")
        return

    features = df['feature'].unique()
    models = df['model'].unique().tolist()
    
    sim_label = "Soft Jaccard" if similarity_type == 'soft' else "Jaccard"
    sim_file_label = similarity_type # This will be 'soft' or 'hard'

    if consistency_type == 'external':
        if similarity_type == 'soft':
            nd_df, src_df = analyze_external_consistency_soft(df, features, models, embedding_model, cache, cache_file)
        else:
            nd_df, src_df = analyze_external_consistency_hard(df, features, models)
        
        plot_heatmap(nd_df, 'jaccard', f'External Consistency ({sim_label} - Criteria Name & Desc)', 
                     os.path.join(output_dir, f'external_criteria_name_desc_{sim_file_label}_heatmap.png'), 'external')
        plot_heatmap(src_df, 'jaccard', 'External Consistency (Jaccard - Sources)', 
                     os.path.join(output_dir, 'external_sources_jaccard_heatmap.png'), 'external')
    
    elif consistency_type == 'internal':
        if similarity_type == 'soft':
            nd_df, src_df = analyze_internal_consistency_soft(df, features, models, embedding_model, cache, cache_file)
        else:
            nd_df, src_df = analyze_internal_consistency_hard(df, features, models)

        plot_heatmap(nd_df, 'jaccard', f'Internal Consistency ({sim_label} - Criteria Name & Desc)', 
                     os.path.join(output_dir, f'internal_criteria_name_desc_{sim_file_label}_heatmap.png'), 'internal', models)
        plot_heatmap(src_df, 'jaccard', 'Internal Consistency (Jaccard - Sources)', 
                     os.path.join(output_dir, 'internal_sources_jaccard_heatmap.png'), 'internal', models)
    
    elif consistency_type is None:
        # Generate both internal and external consistency plots side by side
        print("Generating both internal and external consistency plots...")
        
        # Analyze internal consistency
        if similarity_type == 'soft':
            internal_nd_df, internal_src_df = analyze_internal_consistency_soft(df, features, models, embedding_model, cache, cache_file)
        else:
            internal_nd_df, internal_src_df = analyze_internal_consistency_hard(df, features, models)
        
        # Analyze external consistency
        if similarity_type == 'soft':
            external_nd_df, external_src_df = analyze_external_consistency_soft(df, features, models, embedding_model, cache, cache_file)
        else:
            external_nd_df, external_src_df = analyze_external_consistency_hard(df, features, models)
        
        # Create side-by-side plots
        plot_side_by_side_heatmaps(internal_nd_df, external_nd_df, 'jaccard', 
                                   f'Consistency Analysis ({sim_label} - Criteria Name & Desc)', 
                                   os.path.join(output_dir, f'consistency_criteria_name_desc_{sim_file_label}_side_by_side.png'), 
                                   similarity_type, models)
        
        plot_side_by_side_heatmaps(internal_src_df, external_src_df, 'jaccard', 
                                   'Consistency Analysis (Jaccard - Sources)', 
                                   os.path.join(output_dir, 'consistency_sources_jaccard_side_by_side.png'), 
                                   similarity_type, models)
    
    else:
        print(f"Error: Invalid consistency_type '{consistency_type}'. Must be 'external', 'internal', or None for both.")
    
    # Save final cache state
    if cache is not None and cache_file is not None:
        save_similarity_cache(cache, cache_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze consistency of ranking criteria.')
    parser.add_argument('--input-file', default='data/output/evaluation/app_ranking_criteria.csv',
                        help='Path to the input CSV file with ranking criteria.')
    parser.add_argument('--output-dir', default='data/output/evaluation/ranking_criteria_consistency',
                        help='Directory to save output files.')
    parser.add_argument('--consistency-type', choices=['external', 'internal'], default=None,
                        help='Type of consistency analysis: external, internal, or omit for both side by side.')
    parser.add_argument('--similarity-type', choices=['hard', 'soft'], default='hard',
                        help='Type of similarity for criteria: hard (exact match) or soft (semantic).')
    
    args = parser.parse_args()
    main(input_csv_path=args.input_file, 
         output_dir=args.output_dir, 
         consistency_type=args.consistency_type,
         similarity_type=args.similarity_type)
