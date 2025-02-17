import argparse
import json
import os
from pathlib import Path
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def process_rankings(input_folder: str, experiment_name: str, output_folder: str):
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
            
            for app in data['apps_ranked']:
                app_name = app['name']
                current_apps.add(app_name)
                rank = int(app['rank'])
                
                if app_name not in app_rankings:
                    # Initialize with None values for previous runs
                    app_rankings[app_name] = [None] * run_index
                app_rankings[app_name].append(rank)
            
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
    apps = sorted(list(app_rankings.keys()))  # Sort apps alphabetically
    max_rank = 10  # As specified in requirements
    
    # Initialize matrix for heatmap
    heatmap_data = []
    for app in apps:
        row = [0] * max_rank  # Initialize counts for each rank (1-10)
        for rank in app_rankings[app]:
            if rank is not None:
                row[rank-1] += 1  # -1 because ranks are 1-based
        heatmap_data.append(row)
    
    # Create DataFrame
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=apps,
        columns=[f"Rank {i+1}" for i in range(max_rank)]
    )
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='YlGn', annot_kws={'size': 12})
    plt.title(f'App Rankings Distribution - {experiment_name}')
    plt.ylabel('Apps')
    plt.xlabel('Ranking Position')
    
    # Save heatmap
    heatmap_path = os.path.join(output_folder, f"{experiment_name}_rankings_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved heatmap to {heatmap_path}")
    
    # Create CSV with raw rankings
    csv_data = []
    for app in apps:  # Using the same sorted apps list
        # Include all rankings, including None values
        rankings = app_rankings[app]
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
    
    # Print metadata
    logger.info(f"Total number of distinct apps: {len(apps)}")
    logger.info(f"Number of experiment runs: {len(json_files)}")
    
    # Print apps that maintained consistent rankings
    for app in apps:
        rankings = app_rankings[app]
        if len(set(rankings)) == 1 and all(rank is None for rank in rankings):
            logger.info(f"{app} maintained consistent rank of None across all runs")

def main():
    parser = argparse.ArgumentParser(description='Process app rankings from JSON files')
    parser.add_argument('--input_folder', required=True, help='Input folder containing JSON files')
    parser.add_argument('--experiment_name', required=True, help='Experiment name prefix for JSON files')
    parser.add_argument('--output_folder', required=True, help='Output folder for results')
    
    args = parser.parse_args()
    process_rankings(args.input_folder, args.experiment_name, args.output_folder)

if __name__ == "__main__":
    main()