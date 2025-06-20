import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import os
import argparse
import re
from sentence_transformers import SentenceTransformer, util

def soft_jaccard_similarity(set1, set2, model):
    """
    Calculates a 'soft' Jaccard similarity between two sets of strings based on semantic similarity.
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

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
        return 1.0

    return intersection_score / denominator

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
        return set(), set(), set()
    
    criteria_name_only = set(df_subset['n'].dropna())
    criteria_name_desc = set(zip(df_subset['n'].dropna(), df_subset['d'].dropna()))
    sources = parse_sources(df_subset['s'])
    
    return criteria_name_only, criteria_name_desc, sources

def get_criteria_and_sources_for_soft_sim(df_subset):
    """
    Extracts sets for SOFT similarity (name+desc are concatenated strings).
    """
    if df_subset.empty:
        return set(), set(), set()
    
    criteria_name_only = set(df_subset['n'].dropna())
    df_nd = df_subset[['n', 'd']].dropna()
    criteria_name_desc = set(df_nd['n'] + ': ' + df_nd['d'])
    sources = parse_sources(df_subset['s'])
    
    return criteria_name_only, criteria_name_desc, sources

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
    
    plt.figure(figsize=(max(12, len(pivot_df_with_avg.columns) * 1.5), 8))
    sns.heatmap(pivot_df_with_avg, annot=True, cmap='YlGnBu', fmt=".2f", vmin=0, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def analyze_external_consistency_hard(df, features, models):
    criteria_name_results, criteria_name_desc_results, sources_results = [], [], []
    for feature in features:
        for model1, model2 in combinations(models, 2):
            df1 = df[(df['model'] == model1) & (df['feature'] == feature)]
            df2 = df[(df['model'] == model2) & (df['feature'] == feature)]
            name1, nd1, src1 = get_criteria_and_sources_for_hard_sim(df1)
            name2, nd2, src2 = get_criteria_and_sources_for_hard_sim(df2)
            
            # Standard Jaccard
            def jaccard(s1, s2):
                return len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0
                
            criteria_name_results.append({'feature': feature, 'model1': model1, 'model2': model2, 'jaccard': jaccard(name1, name2)})
            criteria_name_desc_results.append({'feature': feature, 'model1': model1, 'model2': model2, 'jaccard': jaccard(nd1, nd2)})
            sources_results.append({'feature': feature, 'model1': model1, 'model2': model2, 'jaccard': jaccard(src1, src2)})
    return pd.DataFrame(criteria_name_results), pd.DataFrame(criteria_name_desc_results), pd.DataFrame(sources_results)

def analyze_internal_consistency_hard(df, features, models):
    criteria_name_results, criteria_name_desc_results, sources_results = [], [], []
    for model in models:
        for feature in features:
            model_feature_df = df[(df['model'] == model) & (df['feature'] == feature)]
            runs = model_feature_df['run'].unique()
            if len(runs) < 2: continue
            
            run_data = {run: get_criteria_and_sources_for_hard_sim(model_feature_df[model_feature_df['run'] == run]) for run in runs}
            
            def jaccard(s1, s2):
                return len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0

            name_scores, nd_scores, src_scores = [], [], []
            for (r1, (name1, nd1, src1)), (r2, (name2, nd2, src2)) in combinations(run_data.items(), 2):
                name_scores.append(jaccard(name1, name2))
                nd_scores.append(jaccard(nd1, nd2))
                src_scores.append(jaccard(src1, src2))
                
            criteria_name_results.append({'feature': feature, 'model': model, 'jaccard': np.mean(name_scores) if name_scores else 0})
            criteria_name_desc_results.append({'feature': feature, 'model': model, 'jaccard': np.mean(nd_scores) if nd_scores else 0})
            sources_results.append({'feature': feature, 'model': model, 'jaccard': np.mean(src_scores) if src_scores else 0})
    return pd.DataFrame(criteria_name_results), pd.DataFrame(criteria_name_desc_results), pd.DataFrame(sources_results)

def analyze_external_consistency_soft(df, features, models, embedding_model):
    criteria_name_results, criteria_name_desc_results, sources_results = [], [], []
    for feature in features:
        for model1, model2 in combinations(models, 2):
            df1 = df[(df['model'] == model1) & (df['feature'] == feature)]
            df2 = df[(df['model'] == model2) & (df['feature'] == feature)]
            name1, nd1, src1 = get_criteria_and_sources_for_soft_sim(df1)
            name2, nd2, src2 = get_criteria_and_sources_for_soft_sim(df2)
            
            criteria_name_results.append({'feature': feature, 'model1': model1, 'model2': model2, 'jaccard': soft_jaccard_similarity(name1, name2, embedding_model)})
            criteria_name_desc_results.append({'feature': feature, 'model1': model1, 'model2': model2, 'jaccard': soft_jaccard_similarity(nd1, nd2, embedding_model)})
            src_union = src1.union(src2)
            sources_results.append({'feature': feature, 'model1': model1, 'model2': model2, 'jaccard': len(src1.intersection(src2)) / len(src_union) if src_union else 0})
    return pd.DataFrame(criteria_name_results), pd.DataFrame(criteria_name_desc_results), pd.DataFrame(sources_results)

def analyze_internal_consistency_soft(df, features, models, embedding_model):
    criteria_name_results, criteria_name_desc_results, sources_results = [], [], []
    for model in models:
        for feature in features:
            model_feature_df = df[(df['model'] == model) & (df['feature'] == feature)]
            runs = model_feature_df['run'].unique()
            if len(runs) < 2: continue

            run_data = {run: get_criteria_and_sources_for_soft_sim(model_feature_df[model_feature_df['run'] == run]) for run in runs}
            
            name_scores, nd_scores, src_scores = [], [], []
            for (r1, (name1, nd1, src1)), (r2, (name2, nd2, src2)) in combinations(run_data.items(), 2):
                name_scores.append(soft_jaccard_similarity(name1, name2, embedding_model))
                nd_scores.append(soft_jaccard_similarity(nd1, nd2, embedding_model))
                src_union = src1.union(src2)
                src_scores.append(len(src1.intersection(src2)) / len(src_union) if src_union else 0)

            criteria_name_results.append({'feature': feature, 'model': model, 'jaccard': np.mean(name_scores) if name_scores else 0})
            criteria_name_desc_results.append({'feature': feature, 'model': model, 'jaccard': np.mean(nd_scores) if nd_scores else 0})
            sources_results.append({'feature': feature, 'model': model, 'jaccard': np.mean(src_scores) if src_scores else 0})
    return pd.DataFrame(criteria_name_results), pd.DataFrame(criteria_name_desc_results), pd.DataFrame(sources_results)

def main(input_csv_path, output_dir, consistency_type, similarity_type):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    embedding_model = None
    if similarity_type == 'soft':
        print("Loading sentence embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")

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
            name_df, nd_df, src_df = analyze_external_consistency_soft(df, features, models, embedding_model)
        else:
            name_df, nd_df, src_df = analyze_external_consistency_hard(df, features, models)
        
        plot_heatmap(name_df, 'jaccard', f'External Consistency ({sim_label} - Criteria Name Only)', 
                     os.path.join(output_dir, f'external_criteria_name_{sim_file_label}_heatmap.png'), 'external')
        plot_heatmap(nd_df, 'jaccard', f'External Consistency ({sim_label} - Criteria Name & Desc)', 
                     os.path.join(output_dir, f'external_criteria_name_desc_{sim_file_label}_heatmap.png'), 'external')
        plot_heatmap(src_df, 'jaccard', 'External Consistency (Jaccard - Sources)', 
                     os.path.join(output_dir, 'external_sources_jaccard_heatmap.png'), 'external')
    
    elif consistency_type == 'internal':
        if similarity_type == 'soft':
            name_df, nd_df, src_df = analyze_internal_consistency_soft(df, features, models, embedding_model)
        else:
            name_df, nd_df, src_df = analyze_internal_consistency_hard(df, features, models)

        plot_heatmap(name_df, 'jaccard', f'Internal Consistency ({sim_label} - Criteria Name Only)', 
                     os.path.join(output_dir, f'internal_criteria_name_{sim_file_label}_heatmap.png'), 'internal', models)
        plot_heatmap(nd_df, 'jaccard', f'Internal Consistency ({sim_label} - Criteria Name & Desc)', 
                     os.path.join(output_dir, f'internal_criteria_name_desc_{sim_file_label}_heatmap.png'), 'internal', models)
        plot_heatmap(src_df, 'jaccard', 'Internal Consistency (Jaccard - Sources)', 
                     os.path.join(output_dir, 'internal_sources_jaccard_heatmap.png'), 'internal', models)
    
    else:
        print(f"Error: Invalid consistency_type '{consistency_type}'. Must be 'external' or 'internal'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze consistency of ranking criteria.')
    parser.add_argument('--input-file', default='data/output/evaluation/app_ranking_criteria.csv',
                        help='Path to the input CSV file with ranking criteria.')
    parser.add_argument('--output-dir', default='data/output/evaluation/ranking_criteria_consistency',
                        help='Directory to save output files.')
    parser.add_argument('--consistency-type', choices=['external', 'internal'], default='external',
                        help='Type of consistency analysis: external or internal.')
    parser.add_argument('--similarity-type', choices=['hard', 'soft'], default='hard',
                        help='Type of similarity for criteria: hard (exact match) or soft (semantic).')
    
    args = parser.parse_args()
    main(input_csv_path=args.input_file, 
         output_dir=args.output_dir, 
         consistency_type=args.consistency_type,
         similarity_type=args.similarity_type)
