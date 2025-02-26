import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from adjustText import adjust_text
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity

def load_data(uc1_file: str, uc2_file: str):
    """Load experiment results for UC1 and UC2"""
    uc1 = pd.read_csv(uc1_file, index_col=0)
    uc2 = pd.read_csv(uc2_file, index_col=0)
    
    # Get all unique apps from both use cases
    all_apps = set(uc1.index) | set(uc2.index)
    
    # Reindex both dataframes to include all apps, filling missing with 0
    uc1 = uc1.reindex(all_apps, fill_value=0)
    uc2 = uc2.reindex(all_apps, fill_value=0)
    
    return uc1, uc2

def plot_rank_stability(uc1, uc2, output_path):
    """Create visualization for rank stability comparison between UC1 and UC2"""
    plt.figure(figsize=(12, 8))
    
    # Calculate rank stability metrics for all apps
    all_apps = sorted(list(set(uc1.index) | set(uc2.index)))
    stability_data = pd.DataFrame(index=all_apps)
    
    # Calculate std dev only of actual rankings (when app was recommended)
    stability_data['UC1_std'] = uc1.apply(lambda x: x[x > 0].std(), axis=1)
    stability_data['UC2_std'] = uc2.apply(lambda x: x[x > 0].std(), axis=1)
    stability_data['UC1_recommendations'] = uc1.apply(lambda x: (x > 0).sum(), axis=1)
    stability_data['UC2_recommendations'] = uc2.apply(lambda x: (x > 0).sum(), axis=1)
    
    # Fill NaN with 0 for apps that were never recommended
    stability_data = stability_data.fillna(0)
    
    # Create bar plot
    x = np.arange(len(stability_data))
    width = 0.35
    
    plt.bar(x - width/2, stability_data['UC1_std'], width, 
            label=f'UC1', alpha=0.6)
    plt.bar(x + width/2, stability_data['UC2_std'], width, 
            label=f'UC2', alpha=0.6)
    
    plt.xlabel('Apps', fontsize=14)
    plt.ylabel('Rank Standard Deviation\n(when recommended)', fontsize=14)
    plt.title('Rank Stability Comparison between UC1 and UC2', fontsize=16)
    
    plt.xticks(x, stability_data.index, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Enhanced legend showing recommendation counts
    legend_labels = []
    for uc in ['UC1', 'UC2']:
        total_runs = len(uc1.columns)  # same for both UCs
        never_recommended = sum(stability_data[f'{uc}_recommendations'] == 0)
        legend_labels.append(
            f'{uc} ({never_recommended} apps never recommended, '
            f'{len(stability_data) - never_recommended} apps recommended at least once)'
        )
    
    plt.legend(legend_labels, fontsize=12, loc='upper right')
    
    # Add recommendation count annotations
    for i, app in enumerate(stability_data.index):
        uc1_count = stability_data.loc[app, 'UC1_recommendations']
        uc2_count = stability_data.loc[app, 'UC2_recommendations']
        
        # Always show counts, even if 0
        plt.text(i - width/2, stability_data.loc[app, 'UC1_std'], 
                f'{int(uc1_count)}', ha='center', va='bottom')
        plt.text(i + width/2, stability_data.loc[app, 'UC2_std'], 
                f'{int(uc2_count)}', ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path / 'rank_stability.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stability_data

def plot_rank_comparison(uc1, uc2, output_path):
    """Create scatter plot comparing average ranks between UC1 and UC2"""
    plt.figure(figsize=(8, 6))
    
    # Calculate average ranks (excluding zeros)
    avg_ranks = pd.DataFrame(index=sorted(set(uc1.index) | set(uc2.index)))
    avg_ranks['UC1'] = uc1.apply(lambda x: x[x > 0].mean() if (x > 0).any() else 0, axis=1)
    avg_ranks['UC2'] = uc2.apply(lambda x: x[x > 0].mean() if (x > 0).any() else 0, axis=1)
    
    # Filter out apps that never appear in either use case
    avg_ranks = avg_ranks[(avg_ranks['UC1'] > 0) | (avg_ranks['UC2'] > 0)]
    
    # Create scatter plot
    plt.scatter(avg_ranks['UC1'], avg_ranks['UC2'], alpha=0.6, s=100)
    
    # Add diagonal line for perfect agreement
    max_rank = max(avg_ranks['UC1'].max(), avg_ranks['UC2'].max())
    plt.plot([0, max_rank], [0, max_rank], 'r--', alpha=0.5, label='Perfect Agreement')
    
    # Add labels for each point
    for app in avg_ranks.index:
        plt.annotate(app, 
                    (avg_ranks.loc[app, 'UC1'], avg_ranks.loc[app, 'UC2']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10)
    
    plt.xlabel('Average Rank in UC1', fontsize=14)
    plt.ylabel('Average Rank in UC2', fontsize=14)
    plt.title('Average Rank Comparison between UC1 and UC2', fontsize=16)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_path / 'rank_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data to CSV
    avg_ranks.to_csv(output_path / 'rank_comparison.csv')
    
    return avg_ranks

def compute_consistency_metrics(uc1, uc2):
    """Compute various consistency metrics for each app between UC1 and UC2"""
    all_apps = sorted(set(uc1.index) | set(uc2.index))
    metrics = pd.DataFrame(index=all_apps)
    
    for app in all_apps:
        uc1_ranks = uc1.loc[app]
        uc2_ranks = uc2.loc[app]
        
        # Get non-zero values
        uc1_nonzero = uc1_ranks[uc1_ranks > 0]
        uc2_nonzero = uc2_ranks[uc2_ranks > 0]
        
        metrics.loc[app, 'presence_rate_uc1'] = (uc1_ranks > 0).mean()
        metrics.loc[app, 'presence_rate_uc2'] = (uc2_ranks > 0).mean()
        
        # Average ranks when recommended
        metrics.loc[app, 'avg_rank_uc1'] = uc1_nonzero.mean() if len(uc1_nonzero) > 0 else 0
        metrics.loc[app, 'avg_rank_uc2'] = uc2_nonzero.mean() if len(uc2_nonzero) > 0 else 0
        
        # Rank stability within each UC
        metrics.loc[app, 'rank_std_uc1'] = uc1_nonzero.std() if len(uc1_nonzero) > 0 else 0
        metrics.loc[app, 'rank_std_uc2'] = uc2_nonzero.std() if len(uc2_nonzero) > 0 else 0
        
        # Cross-UC metrics
        metrics.loc[app, 'abs_rank_difference'] = abs(
            (uc1_nonzero.mean() if len(uc1_nonzero) > 0 else 0) -
            (uc2_nonzero.mean() if len(uc2_nonzero) > 0 else 0)
        )
        
        # Recommendation consistency
        metrics.loc[app, 'appears_in_both'] = int((len(uc1_nonzero) > 0) and (len(uc2_nonzero) > 0))
        metrics.loc[app, 'appears_in_one_only'] = int((len(uc1_nonzero) > 0) != (len(uc2_nonzero) > 0))
        
        # Relative rank stability
        total_recommendations = len(uc1_nonzero) + len(uc2_nonzero)
        if total_recommendations > 0:
            metrics.loc[app, 'rank_consistency_score'] = 1 - (
                metrics.loc[app, 'abs_rank_difference'] / total_recommendations
            )
        else:
            metrics.loc[app, 'rank_consistency_score'] = 0

    # Add column descriptions
    metrics.columns.name = 'Metric'
    metrics.index.name = 'App'
    
    # Add metric descriptions
    descriptions = {
        'presence_rate_uc1': 'Fraction of runs where app appears in UC1 (0-1)',
        'presence_rate_uc2': 'Fraction of runs where app appears in UC2 (0-1)',
        'avg_rank_uc1': 'Average rank when recommended in UC1 (0 if never recommended)',
        'avg_rank_uc2': 'Average rank when recommended in UC2 (0 if never recommended)',
        'rank_std_uc1': 'Standard deviation of ranks in UC1 (0 if never recommended)',
        'rank_std_uc2': 'Standard deviation of ranks in UC2 (0 if never recommended)',
        'abs_rank_difference': 'Absolute difference between average ranks in UC1 and UC2',
        'appears_in_both': 'Binary indicator if app appears in both use cases (1) or not (0)',
        'appears_in_one_only': 'Binary indicator if app appears in exactly one use case (1) or not (0)',
        'rank_consistency_score': 'Normalized rank consistency (1 = perfect consistency, 0 = high inconsistency)'
    }
    
    return metrics, pd.Series(descriptions)

def compute_similarity_metrics(uc1, uc2):
    """Compute similarity metrics both at app level and globally"""
    all_apps = sorted(set(uc1.index) | set(uc2.index))
    metrics = pd.DataFrame(index=all_apps, 
                         columns=['jaccard_similarity', 'overlap_coefficient', 
                                'cosine_similarity', 'spearman_correlation', 
                                'kendall_correlation'])
    
    # App-level metrics
    for app in all_apps:
        uc1_ranks = uc1.loc[app]
        uc2_ranks = uc2.loc[app]
        
        # Convert ranks to binary presence (0/1) for each run
        uc1_presence = (uc1_ranks > 0).astype(int)
        uc2_presence = (uc2_ranks > 0).astype(int)
        
        # Jaccard similarity (presence overlap)
        metrics.loc[app, 'jaccard_similarity'] = jaccard_score(uc1_presence, uc2_presence, zero_division=0)
        
        # Overlap coefficient (handles different sizes)
        overlap = sum((uc1_presence == 1) & (uc2_presence == 1))
        metrics.loc[app, 'overlap_coefficient'] = overlap / min(sum(uc1_presence), sum(uc2_presence)) if min(sum(uc1_presence), sum(uc2_presence)) > 0 else 0
        
        # Cosine similarity between rank vectors (handles identical values well)
        metrics.loc[app, 'cosine_similarity'] = cosine_similarity(
            uc1_ranks.values.reshape(1, -1), 
            uc2_ranks.values.reshape(1, -1)
        )[0][0]
        
        # Set correlations to NaN for individual apps
        metrics.loc[app, 'spearman_correlation'] = np.nan
        metrics.loc[app, 'kendall_correlation'] = np.nan
    
    # Global metrics (as a new row)
    # Calculate average correlations across runs
    run_correlations = []
    for run in uc1.columns:
        uc1_run = uc1[run]
        uc2_run = uc2[run]
        common_apps = (uc1_run > 0) & (uc2_run > 0)
        if sum(common_apps) > 1:
            spearman = spearmanr(uc1_run[common_apps], uc2_run[common_apps])[0]
            kendall = kendalltau(uc1_run[common_apps], uc2_run[common_apps])[0]
            run_correlations.append((spearman, kendall))
    
    # Calculate global metrics
    total_recommendations_uc1 = (uc1 > 0).sum().sum()
    total_recommendations_uc2 = (uc2 > 0).sum().sum()
    common_recommendations = ((uc1 > 0) & (uc2 > 0)).sum().sum()
    
    # Calculate global cosine similarity (average across all runs)
    global_cosine = np.mean([
        cosine_similarity(
            uc1[col].values.reshape(1, -1),
            uc2[col].values.reshape(1, -1)
        )[0][0]
        for col in uc1.columns
    ])
    
    # Add global metrics as a new row with all metrics
    metrics.loc['AGGREGATED'] = {
        'jaccard_similarity': common_recommendations / (total_recommendations_uc1 + total_recommendations_uc2 - common_recommendations),
        'overlap_coefficient': common_recommendations / min(total_recommendations_uc1, total_recommendations_uc2),
        'cosine_similarity': global_cosine,
        'spearman_correlation': np.mean([corr[0] for corr in run_correlations]) if run_correlations else np.nan,
        'kendall_correlation': np.mean([corr[1] for corr in run_correlations]) if run_correlations else np.nan
    }
    
    # Create description of metrics
    descriptions = {
        'jaccard_similarity': 'Similarity of app presence patterns across runs (0-1)',
        'overlap_coefficient': 'Overlap in recommendations considering different recommendation frequencies (0-1)',
        'cosine_similarity': 'Cosine similarity between rank vectors (-1 to 1)',
        'spearman_correlation': 'Average Spearman correlation across all runs (aggregated only)',
        'kendall_correlation': 'Average Kendall Tau correlation across all runs (aggregated only)'
    }
    
    return metrics, pd.Series(descriptions)

def analyze_consistency(uc1_file: str, uc2_file: str, output_path: str):
    """Analyze consistency between UC1 and UC2 experiments"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    uc1, uc2 = load_data(uc1_file, uc2_file)
    
    # Generate all metrics and plots
    stability_data = plot_rank_stability(uc1, uc2, output_path)
    avg_ranks = plot_rank_comparison(uc1, uc2, output_path)
    metrics, descriptions = compute_similarity_metrics(uc1, uc2)
    
    # Save all data in both CSV and XLSX formats
    # 1. Stability data
    stability_data.to_csv(output_path / 'rank_stability.csv')
    stability_data.to_excel(output_path / 'rank_stability.xlsx')
    
    # 2. Average ranks
    avg_ranks.to_csv(output_path / 'rank_comparison.csv')
    avg_ranks.to_excel(output_path / 'rank_comparison.xlsx')
    
    # 3. Similarity metrics
    metrics.to_csv(output_path / 'similarity_metrics.csv')
    metrics.to_excel(output_path / 'similarity_metrics.xlsx')
    
    # 4. Metric descriptions
    descriptions.to_csv(output_path / 'metric_descriptions.csv')
    descriptions.to_frame('Description').to_excel(output_path / 'metric_descriptions.xlsx')
    
    print("\nSimilarity Metrics (including aggregated):")
    print(metrics)
    print("\nMetric Descriptions:")
    print(descriptions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze consistency between two use case experiments')
    parser.add_argument('--uc1_file', type=str, required=True,
                        help='Path to the CSV file containing UC1 experiment results')
    parser.add_argument('--uc2_file', type=str, required=True,
                        help='Path to the CSV file containing UC2 experiment results')
    parser.add_argument('--output_path', type=str, default='output/metrics',
                        help='Path to directory where metrics and visualizations will be saved')
    
    args = parser.parse_args()
    
    analyze_consistency(
        uc1_file=args.uc1_file,
        uc2_file=args.uc2_file,
        output_path=args.output_path
    )
