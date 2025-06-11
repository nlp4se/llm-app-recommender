import pandas as pd
import subprocess
import os
from pathlib import Path
import argparse

def run_ranking_matrix_commands(model: str):
    # Read the categories and k values
    categories_df = pd.read_csv('data/input/use-case/categories.csv')
    k_df = pd.read_csv('data/input/use-case/k.csv')
    
    # Get unique values
    categories = categories_df['category'].unique()
    k_values = k_df['k'].unique()
    
    # Base paths
    base_input_path = Path('data/output/search/uc1') / model
    base_output_path = Path('data/output/search/uc1') / model
    
    # Run commands for each combination
    for k in k_values:
        for category in categories:
            # Construct folder names
            k_folder = f'k{k}'
            category_folder = category.replace(' ', '_')
            
            # Construct full paths
            folder_name = f"{k_folder}_{category_folder}"
            input_folder = base_input_path / folder_name
            output_folder = base_output_path / folder_name
            
            # Ensure output directory exists
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Construct and run the command
            cmd = [
                'python', '-m', 'code.data-processor.ranking_matrix',
                '--input_folder', str(input_folder),
                '--experiment_name', 'user-prompt-uc1',
                '--output_folder', str(output_folder),
                '--category_name', f'"{category}"',
                '--max_rank', '20'
            ]
            
            print(f"\nRunning command for model={model}, k={k}, category={category}")
            print(f"Input folder: {input_folder}")
            print(f"Output folder: {output_folder}")
            
            try:
                subprocess.run(cmd, check=True)
                print("Command completed successfully")
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
                print("\nTo reproduce this command locally, run:")
                print(" ".join(cmd))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ranking matrix commands for different categories and k values')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., openai)')
    args = parser.parse_args()
    
    run_ranking_matrix_commands(args.model)
