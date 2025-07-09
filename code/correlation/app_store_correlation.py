import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Global array of k values
K_VALUES = [3, 10, 20]

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

def get_llm_ranked_lists(df, model, feature, k):
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

def get_app_store_ranking(df, feature, source, k):
    """
    Extracts the app store ranking for a given feature and source, truncated to k items.
    """
    store_df = df[(df['feature'] == feature) & (df['source'] == source)]
    if store_df.empty:
        return []
    
    # Get ranking columns from '1' to k
    rank_cols = [str(i) for i in range(1, k + 1)]
    rank_cols_exist = [col for col in rank_cols if col in store_df.columns]
    
    if not rank_cols_exist:
        return []
    
    # Get the first (and only) row
    ranking = store_df[rank_cols_exist].iloc[0].tolist()
    return [app for app in ranking if pd.notna(app)]

def rename_feature_for_display(feature_name):
    """
    Rename features for better display in plots.
    """
    if feature_name == 'Broadcast messages to multiple contacts':
        return 'Broadcast messages'
    return feature_name

def calculate_rbo_metrics(llm_lists, app_store_list, k):
    """
    Calculate RBO metrics between LLM rankings and app store ranking.
    """
    if not llm_lists or not app_store_list:
        return {'jaccard': 0, 'rbo': 0}
    
    # Calculate RBO and Jaccard for each LLM run
    rbo_scores = []
    jaccard_scores = []
    
    for llm_list in llm_lists:
        # RBO
        rbo_score = rbo(llm_list, app_store_list, p=0.9)
        rbo_scores.append(rbo_score)
        
        # Jaccard similarity
        set1 = set(llm_list)
        set2 = set(app_store_list)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard_sim = intersection / union if union != 0 else 0
        jaccard_scores.append(jaccard_sim)
    
    return {
        'jaccard': np.mean(jaccard_scores) if jaccard_scores else 0,
        'rbo': np.mean(rbo_scores) if rbo_scores else 0
    }

def analyze_rbo(llm_df, app_store_df, features, models, k, source):
    """
    Analyze RBO between LLM recommendations and app store recommendations.
    """
    results = []
    
    for model in models:
        for feature in features:
            # Get LLM rankings
            llm_lists = get_llm_ranked_lists(llm_df, model, feature, k)
            
            # Get app store ranking
            app_store_list = get_app_store_ranking(app_store_df, feature, source, k)
            
            if not llm_lists or not app_store_list:
                continue
            
            # Calculate metrics
            metrics = calculate_rbo_metrics(llm_lists, app_store_list, k)
            
            results.append({
                'model': model,
                'feature': feature,
                'jaccard': metrics['jaccard'],
                'rbo': metrics['rbo']
            })
    
    return pd.DataFrame(results)

def plot_heatmap(df, metric, title, output_path, model_order=None):
    """
    Plots a heatmap of RBO scores.
    """
    if df.empty:
        print(f"Skipping plot for '{title}' as there is no data.")
        return
    
    # Create pivot table
    pivot_df = df.pivot_table(index='feature', columns='model', values=metric)
    
    # Reorder columns to match the original model order if provided
    if model_order is not None:
        existing_models = [model for model in model_order if model in pivot_df.columns]
        if existing_models:
            pivot_df = pivot_df[existing_models]
    
    # Rename features for display
    pivot_df.index = pivot_df.index.map(rename_feature_for_display)
    
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
    
    # Set larger figure size and font sizes
    plt.figure(figsize=(max(12, len(pivot_df_with_avg.columns) * 1.5), 8))
    
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
    
    # Use YlGnBu colormap for RBO and Jaccard (0 to 1 range)
    cmap = 'YlGnBu'
    vmin, vmax = 0, 1
    
    # Create heatmap
    sns.heatmap(pivot_df_with_avg, annot=True, cmap=cmap, fmt=".3f", 
                vmin=vmin, vmax=vmax, cbar_kws={'label': f'{metric.title()} Score'},
                annot_kws={'size': 10})
    
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")

def create_combined_heatmap_figure(all_data, metric, source, output_dir, model_order=None):
    """
    Create a combined figure with all k-value heatmaps arranged horizontally.
    """
    if not all_data:
        print(f"No data available for {metric} combined heatmap")
        return
    
    k_values = sorted(all_data.keys())
    num_k = len(k_values)
    
    # Set up the figure with subplots
    fig_height = 8
    widths = [1] * (num_k - 1) + [1.3]  # Make last subplot wider
    fig, axes = plt.subplots(
        1, num_k, 
        figsize=(4.5 * num_k + 2, fig_height),
        gridspec_kw={'width_ratios': widths}
    )
    if num_k == 1:
        axes = [axes]
    
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
    
    # Use YlGnBu colormap for RBO and Jaccard (0 to 1 range)
    cmap = 'YlGnBu'
    vmin, vmax = 0, 1
    
    # Process each k value
    for i, k in enumerate(k_values):
        df = all_data[k]
        if df.empty:
            continue
            
        ax = axes[i]
        
        # Create pivot table
        pivot_df = df.pivot_table(index='feature', columns='model', values=metric)
        
        # Reorder columns to match the original model order if provided
        if model_order is not None:
            existing_models = [model for model in model_order if model in pivot_df.columns]
            if existing_models:
                pivot_df = pivot_df[existing_models]
        
        # Rename features for display
        pivot_df.index = pivot_df.index.map(rename_feature_for_display)
        
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
        
        # Create heatmap
        if i == 0:  # First subplot - show y-axis labels
            sns.heatmap(pivot_df_with_avg, annot=True, cmap=cmap, fmt=".3f", 
                       vmin=vmin, vmax=vmax, cbar=False, ax=ax,
                       annot_kws={'size': 10})
        elif i == num_k - 1:  # Last subplot - show colorbar
            heatmap = sns.heatmap(pivot_df_with_avg, annot=True, cmap=cmap, fmt=".3f", 
                       vmin=vmin, vmax=vmax, cbar=True, ax=ax,
                       cbar_kws={'label': f'{metric.title()} Score'},
                       annot_kws={'size': 10})
        else:  # Middle subplots - no y-axis labels, no colorbar
            sns.heatmap(pivot_df_with_avg, annot=True, cmap=cmap, fmt=".3f", 
                       vmin=vmin, vmax=vmax, cbar=False, ax=ax,
                       annot_kws={'size': 10})
        
        # Set title with k value
        ax.set_title(f'k = {k}', fontsize=12, pad=15)
        
        # Hide y-axis labels for all except first subplot
        if i > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        ax.xaxis.label.set_size(12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'combined_{source}_{metric}_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined {metric} heatmap to {output_path}")

def analyze_rbo_for_k(llm_df, app_store_df, features, models, k, source, output_dir):
    """
    Analyze RBO for a specific k value and save results.
    """
    print(f"\n=== Analyzing RBO for k={k}, source={source} ===")
    
    # Calculate RBO metrics
    results_df = analyze_rbo(llm_df, app_store_df, features, models, k, source)
    
    if results_df.empty:
        print(f"No results for k={k}, source={source}")
        return results_df
    
    print(f"--- RBO Results (k={k}, source={source}) ---")
    print(results_df)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, f'k{k}_{source}_rbo_results.csv'), index=False)
    
    # Create individual heatmaps for each metric
    metrics = ['jaccard', 'rbo']
    for metric in metrics:
        title = f'{metric.title()} Score with {source.title()} (k={k})'
        output_path = os.path.join(output_dir, f'k{k}_{source}_{metric}_heatmap.png')
        plot_heatmap(results_df, metric, title, output_path, models)
    
    return results_df

def create_summary_report(all_data, source, output_dir):
    """
    Create a summary report with overall statistics.
    """
    report_lines = []
    report_lines.append(f"=== RBO ANALYSIS SUMMARY REPORT ===")
    report_lines.append(f"Source: {source}")
    report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    metrics = ['jaccard', 'rbo']
    
    for metric in metrics:
        report_lines.append(f"--- {metric.upper()} SCORES ---")
        
        for k in sorted(all_data.keys()):
            df = all_data[k]
            if not df.empty and metric in df.columns:
                avg_score = df[metric].mean()
                std_score = df[metric].std()
                min_score = df[metric].min()
                max_score = df[metric].max()
                
                report_lines.append(f"k={k}: Mean={avg_score:.3f}, Std={std_score:.3f}, Min={min_score:.3f}, Max={max_score:.3f}")
        
        report_lines.append("")
    
    # Model performance comparison
    report_lines.append("--- MODEL PERFORMANCE COMPARISON ---")
    for k in sorted(all_data.keys()):
        df = all_data[k]
        if not df.empty:
            report_lines.append(f"k={k}:")
            model_avg = df.groupby('model')[metrics].mean()
            for model in model_avg.index:
                report_lines.append(f"  {model}:")
                for metric in metrics:
                    if metric in model_avg.columns:
                        report_lines.append(f"    {metric}: {model_avg.loc[model, metric]:.3f}")
            report_lines.append("")
    
    # Save report
    report_path = os.path.join(output_dir, f'{source}_rbo_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Saved summary report to {report_path}")

def main(llm_recommendation_path, app_store_recommendation_path, output_folder, k_values=None):
    """
    Main function to run the RBO analysis.
    
    Args:
        llm_recommendation_path (str): Path to the LLM recommendations CSV file
        app_store_recommendation_path (str): Path to the app store recommendations CSV file
        output_folder (str): Directory to save output files
        k_values (list): List of k values to analyze. If None, uses the global K_VALUES
    """
    if k_values is None:
        k_values = K_VALUES
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load data
    try:
        llm_df = pd.read_csv(llm_recommendation_path)
        app_store_df = pd.read_csv(app_store_recommendation_path, delimiter=';')
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return

    # Get unique features and models
    features = llm_df['feature'].unique()
    models = llm_df['model'].unique().tolist()
    sources = app_store_df['source'].unique()

    print(f"Loaded data:")
    print(f"  LLM models: {models}")
    print(f"  Features: {len(features)}")
    print(f"  App store sources: {sources}")
    print(f"  K values: {k_values}")

    # Analyze RBO for each source
    for source in sources:
        print(f"\n{'='*60}")
        print(f"ANALYZING RBO WITH {source.upper()}")
        print(f"{'='*60}")
        
        # Create source-specific output directory
        source_output_dir = os.path.join(output_folder, source)
        if not os.path.exists(source_output_dir):
            os.makedirs(source_output_dir)
        
        # Analyze RBO for each k value and collect results
        all_data = {}
        
        for k in k_values:
            results_df = analyze_rbo_for_k(llm_df, app_store_df, features, models, k, source, source_output_dir)
            all_data[k] = results_df
        
        # Create combined heatmap figures
        metrics = ['jaccard', 'rbo']
        for metric in metrics:
            create_combined_heatmap_figure(all_data, metric, source, source_output_dir, models)
        
        # Create summary report
        create_summary_report(all_data, source, source_output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze RBO between LLM recommendations and app store recommendations')
    parser.add_argument('--llm-recommendation', 
                       required=True,
                       help='Path to the LLM recommendations CSV file (like app_rankings.csv)')
    parser.add_argument('--app-store-recommendation', 
                       required=True,
                       help='Path to the app store recommendations CSV file')
    parser.add_argument('--output-folder', 
                       required=True,
                       help='Directory to save output files')
    parser.add_argument('--k-values', 
                       nargs='+', 
                       type=int,
                       help='List of k values to analyze (e.g., 3 5 10 20). If not provided, uses default values.')
    
    args = parser.parse_args()
    
    # Use provided k values or default to global K_VALUES
    k_values = args.k_values if args.k_values else None
    
    main(args.llm_recommendation, args.app_store_recommendation, args.output_folder, k_values)
