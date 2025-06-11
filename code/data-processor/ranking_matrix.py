import argparse
import json
import os
from pathlib import Path
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def process_rankings(input_folder: str, experiment_name: str, output_folder: str, category_name: str, max_rank: int = 20):
    logger = setup_logging()
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files for the experiment
    json_pattern = f"{experiment_name}_*.json"
    json_files = list(Path(input_folder).glob(json_pattern))
    
    logger.info(f"Found {len(json_files)} files matching pattern {json_pattern}")
    
    # Dictionary to store rankings for each app
    app_rankings = {}
    
    # Process each JSON file
    total_runs = len(json_files)
    for run_index, json_file in enumerate(json_files):
        logger.info(f"Processing file: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
            current_apps = set()  # Track apps in current run
            
            for app in data['apps']:
                app_name = app['name']
                current_apps.add(app_name)
                rank = int(app['rank'])
                
                # Only consider ranks up to max_rank
                if rank <= max_rank:
                    if app_name not in app_rankings:
                        app_rankings[app_name] = [None] * run_index
                    app_rankings[app_name].append(rank)
                else:
                    if app_name not in app_rankings:
                        app_rankings[app_name] = [None] * run_index
                    app_rankings[app_name].append(None)
            
            # Add None for apps that didn't appear in this run
            for app_name in app_rankings:
                if app_name not in current_apps:
                    app_rankings[app_name].append(None)
                    
    # Ensure all apps have the same number of rankings
    for app_name in app_rankings:
        current_length = len(app_rankings[app_name])
        if current_length < total_runs:
            # Pad with None values if necessary
            app_rankings[app_name].extend([None] * (total_runs - current_length))
    
    # Create DataFrame for heatmap
    apps = list(app_rankings.keys())
    
    # Calculate average rank for each app (lower is better)
    app_scores = {}
    for app in apps:
        valid_rankings = [r for r in app_rankings[app] if r is not None]
        if valid_rankings:
            # Calculate weighted average where lower ranks have higher weight
            # We use 1/rank to give more weight to top positions
            weighted_sum = sum(1/r for r in valid_rankings)
            app_scores[app] = weighted_sum / len(valid_rankings)
        else:
            app_scores[app] = 0  # Apps with no rankings go to the bottom
    
    # Filter out apps that never appear in top max_rank positions
    filtered_apps = []
    for app in apps:
        valid_rankings = [r for r in app_rankings[app] if r is not None and r <= max_rank]
        if valid_rankings:  # Only include apps that have at least one valid ranking
            filtered_apps.append(app)
    
    # Sort filtered apps based on their scores
    filtered_apps = sorted(filtered_apps, key=lambda x: app_scores[x], reverse=True)
    
    # Initialize matrix for heatmap using only filtered apps
    heatmap_data = []
    for app in filtered_apps:
        row = [0] * max_rank  # Initialize counts for each rank (1-max_rank)
        for rank in app_rankings[app]:
            if rank is not None and rank <= max_rank:
                row[rank-1] += 1  # -1 because ranks are 1-based
        heatmap_data.append(row)
    
    # Create DataFrame with filtered apps
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=filtered_apps,
        columns=[f"Rank {i+1}" for i in range(max_rank)]
    )
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='YlGn', annot_kws={'size': 12})
    plt.title(f'App Rankings Distribution - {category_name}')
    plt.ylabel('Apps')
    plt.xlabel('Ranking Position')
    
    # Save heatmap
    heatmap_path = os.path.join(output_folder, f"{experiment_name}_rankings_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved heatmap to {heatmap_path}")
    
    # Create CSV with raw rankings (also using filtered apps)
    csv_data = []
    for app in filtered_apps:
        rankings = app_rankings[app]
        # Ensure rankings is the same length as the number of runs
        if len(rankings) < len(json_files):
            rankings = rankings + [None] * (len(json_files) - len(rankings))
        elif len(rankings) > len(json_files):
            rankings = rankings[:len(json_files)]
        csv_data.append([app] + rankings)
    
    # Create DataFrame and save to CSV
    csv_df = pd.DataFrame(
        csv_data,
        columns=['App'] + [f'Run_{i+1}' for i in range(len(json_files))]
    )
    
    # Convert numeric columns to Int64 type and replace None with -1
    numeric_columns = [f'Run_{i+1}' for i in range(len(json_files))]
    csv_df[numeric_columns] = csv_df[numeric_columns].fillna(0).astype('Int64')

    csv_path = os.path.join(output_folder, f"{experiment_name}_rankings.csv")
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"Saved rankings CSV to {csv_path}")
    
    # Create rankings evolution heatmap
    plt.figure(figsize=(12, 8))
    
    # Create a custom colormap:
    # - 0 values will be very light (almost white)
    # - 1-10 rankings will go from dark to light green (reversed)
    # - Clear distinction between 0 and other values
    ranking_colors = sns.color_palette("YlGn_r", 10)  # Get 10 colors from dark to light, reversed
    custom_cmap = ['#f7f7f7'] + ranking_colors.as_hex()  # Add very light gray for 0
    
    # Convert the data to float type for the heatmap
    heatmap_data = csv_df.iloc[:, 1:].astype(float)
    
    # Create heatmap from the rankings data
    sns.heatmap(heatmap_data, 
                yticklabels=csv_df['App'],
                cmap=custom_cmap,
                vmin=0,
                vmax=10,
                annot=True,
                fmt='g',
                annot_kws={'size': 12})
    
    # Modify annotation colors based on value
    for text in plt.gca().texts:
        value = float(text.get_text())
        if value == 0:
            text.set_color('#999999')  # Light gray text for 0 values
    
    plt.title(f'App Rankings Evolution - {experiment_name}', size=12)
    plt.xlabel('Run Number', size=12)
    plt.ylabel('Apps', size=12)
    
    # Save evolution heatmap
    evolution_heatmap_path = os.path.join(output_folder, f"{experiment_name}_rankings_evolution.png")
    plt.savefig(evolution_heatmap_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved rankings evolution heatmap to {evolution_heatmap_path}")
    
    # Calculate ranking metrics
    metrics = calculate_ranking_metrics(app_rankings, max_rank)
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame([{
        'category_name': category_name,
        'avg_run_correlation': metrics['avg_run_correlation'],
        'rss_position_correlation': metrics['rss_position_correlation'],
        **{f'rss_k{k}': metrics[f'rss_k{k}'] for k in range(1, max_rank + 1)},
        **{f'apps_k{k}': metrics[f'apps_k{k}'] for k in range(1, max_rank + 1)},
        'total_apps': len(filtered_apps),
        'total_runs': len(json_files)
    }])
    
    # Save metrics to CSV in parent folder
    metrics_path = os.path.join(os.path.dirname(input_folder), "ranking_metrics.csv")
    
    # If file exists, append the new row
    if os.path.exists(metrics_path):
        existing_df = pd.read_csv(metrics_path)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    
    # Save the combined DataFrame
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved/updated ranking metrics to {metrics_path}")
    
    # Log the metrics
    logger.info("Ranking Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Print metadata
    logger.info(f"Total number of distinct apps: {len(filtered_apps)}")
    logger.info(f"Number of experiment runs: {len(json_files)}")
    
    # Print apps that maintained consistent rankings
    for app in filtered_apps:
        rankings = app_rankings[app]
        if len(set(rankings)) == 1 and all(rank is None for rank in rankings):
            logger.info(f"{app} maintained consistent rank of None across all runs")

def calculate_ranking_metrics(app_rankings: Dict[str, List[int]], max_rank: int) -> Dict[str, float]:
    """Calculate RSS scores for each position and correlation metrics."""
    metrics = {}
    
    # Find the maximum length of rankings
    max_length = max(len(rankings) for rankings in app_rankings.values())
    
    # Pad all rankings to the same length with None values
    padded_rankings = []
    for rankings in app_rankings.values():
        padded = rankings + [None] * (max_length - len(rankings))
        padded_rankings.append(padded)
    
    # Convert to numpy array, replacing None with np.nan
    rankings_array = np.array(padded_rankings, dtype=float)  # This will automatically convert None to np.nan
    
    # Calculate RSS for each position k and count apps per k
    rss_scores = {}
    apps_per_k = {}
    for k in range(1, max_rank + 1):
        stability_scores = []
        apps_at_k = 0
        for app_rankings in rankings_array:
            valid_ranks = [r for r in app_rankings if not np.isnan(r) and r <= k]
            if len(valid_ranks) > 1:
                diffs = np.abs(np.diff(valid_ranks))
                stability_scores.append(np.mean(diffs))
            if any(not np.isnan(r) and r <= k for r in app_rankings):
                apps_at_k += 1
        rss_scores[f'rss_k{k}'] = np.mean(stability_scores) if stability_scores else 0
        apps_per_k[f'apps_k{k}'] = apps_at_k
    
    metrics.update(rss_scores)
    metrics.update(apps_per_k)
    
    # Calculate correlations between runs
    run_correlations = []
    for i in range(max_length - 1):
        for j in range(i + 1, max_length):
            run1 = rankings_array[:, i]
            run2 = rankings_array[:, j]
            # Filter out NaN values
            valid_indices = ~(np.isnan(run1) | np.isnan(run2))
            if np.sum(valid_indices) > 1:  # Need at least 2 points for correlation
                corr = np.corrcoef(run1[valid_indices], run2[valid_indices])[0, 1]
                run_correlations.append(corr)
    
    metrics['avg_run_correlation'] = np.mean(run_correlations) if run_correlations else 0
    
    # Calculate correlation between RSS scores at different positions
    rss_values = [rss_scores[f'rss_k{k}'] for k in range(1, max_rank + 1)]
    positions = list(range(1, max_rank + 1))
    rss_position_corr = np.corrcoef(positions, rss_values)[0, 1]
    metrics['rss_position_correlation'] = rss_position_corr
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Process app rankings from JSON files')
    parser.add_argument('--input_folder', required=True, help='Input folder containing JSON files')
    parser.add_argument('--experiment_name', required=True, help='Experiment name prefix for JSON files')
    parser.add_argument('--category_name', required=True, help='Category name for the experiment')
    parser.add_argument('--output_folder', required=True, help='Output folder for results')
    parser.add_argument('--max_rank', type=int, default=20, help='Maximum rank position to consider (default: 20)')
    
    args = parser.parse_args()
    process_rankings(args.input_folder, args.experiment_name, args.output_folder, args.category_name, args.max_rank)

if __name__ == "__main__":
    main()