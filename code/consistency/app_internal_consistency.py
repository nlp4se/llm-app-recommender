import pandas as pd
import subprocess
import os
from pathlib import Path
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def run_ranking_matrix_commands(model: str):
    # Read the features and k values
    features_df = pd.read_csv('data/input/use-case/features.csv')
    k_df = pd.read_csv('data/input/use-case/k.csv')
    
    # Get unique values
    features = features_df['feature'].unique()
    k_values = k_df['k'].unique()
    
    # Base paths
    base_input_path = Path('data/output/features/rq1') / model
    base_output_path = Path('data/output/features/rq1') / model
    
    # Store all heatmap paths for later concatenation
    heatmap_paths = []
    
    # Run commands for each combination
    for k in k_values:
        for feature in features:
            # Construct folder names
            k_folder = f'k{k}'
            feature_folder = feature.replace(' ', '_')
            
            # Construct full paths
            folder_name = f"{k_folder}_{feature_folder}"
            input_folder = base_input_path / folder_name
            output_folder = base_output_path / folder_name
            
            # Ensure output directory exists
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Construct and run the command
            cmd = [
                'python', '-m', 'code.data-processor.ranking_matrix',
                '--input_folder', str(input_folder),
                '--experiment_name', 'user-prompt-feature-rq1',
                '--output_folder', str(output_folder),
                '--feature_name', f'"{feature}"',
                '--max_rank', '20'
            ]
            
            print(f"\nRunning command for model={model}, k={k}, feature={feature}")
            print(f"Input folder: {input_folder}")
            print(f"Output folder: {output_folder}")
            
            try:
                subprocess.run(cmd, check=True)
                print("Command completed successfully")
                
                # Store the heatmap path for later concatenation
                heatmap_path = output_folder / "user-prompt-feature-rq1_rankings_heatmap.png"
                if heatmap_path.exists():
                    heatmap_paths.append((heatmap_path, f"{k}_{feature}"))
                
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
                print("\nTo reproduce this command locally, run:")
                print(" ".join(cmd))
    
    # Create concatenated heatmap image
    if heatmap_paths:
        create_concatenated_heatmap(heatmap_paths, model)

def create_concatenated_heatmap(heatmap_paths, model: str):
    """Create a horizontally concatenated image of all ranking heatmaps."""
    print(f"\nCreating concatenated heatmap for {len(heatmap_paths)} features...")
    
    # Load all images
    images = []
    labels = []
    
    for heatmap_path, label in heatmap_paths:
        try:
            img = Image.open(heatmap_path)
            images.append(img)
            labels.append(label)
            print(f"Loaded: {label}")
        except Exception as e:
            print(f"Error loading {heatmap_path}: {e}")
    
    if not images:
        print("No images to concatenate!")
        return
    
    # Get dimensions
    heights = [img.height for img in images]
    widths = [img.width for img in images]
    
    # Use the maximum height for all images (resize if needed)
    max_height = max(heights)
    total_width = sum(widths)
    
    # Create the concatenated image
    concatenated_img = Image.new('RGB', (total_width, max_height), 'white')
    
    # Paste images horizontally
    x_offset = 0
    for i, img in enumerate(images):
        # Resize image to match max height if needed
        if img.height != max_height:
            # Calculate new width maintaining aspect ratio
            aspect_ratio = img.width / img.height
            new_width = int(max_height * aspect_ratio)
            img = img.resize((new_width, max_height), Image.Resampling.LANCZOS)
        
        concatenated_img.paste(img, (x_offset, 0))
        x_offset += img.width
    
    # Save the concatenated image
    output_path = Path('data/output/features/rq1') / model / f"{model}_all_rankings_heatmaps.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concatenated_img.save(output_path, 'PNG', dpi=(300, 300))
    
    print(f"\nConcatenated heatmap saved to: {output_path}")
    print(f"Image dimensions: {concatenated_img.size}")
    print(f"Number of features included: {len(images)}")
    
    # Create a legend/annotation image
    create_legend_image(labels, model)

def create_legend_image(labels, model: str):
    """Create a legend showing which feature corresponds to which position in the concatenated image."""
    fig, ax = plt.subplots(figsize=(20, len(labels) * 0.5 + 2))
    
    # Create a simple legend
    for i, label in enumerate(labels):
        # Create a colored rectangle for each feature
        rect = patches.Rectangle((0.1, len(labels) - i - 0.8), 0.1, 0.6, 
                               facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        
        # Add text label
        ax.text(0.25, len(labels) - i - 0.5, label, fontsize=12, 
               verticalalignment='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(labels))
    ax.set_title(f'Feature Legend for {model} Concatenated Heatmap', fontsize=16, pad=20)
    ax.axis('off')
    
    # Save legend
    legend_path = Path('data/output/features/rq1') / model / f"{model}_heatmap_legend.png"
    plt.savefig(legend_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Legend saved to: {legend_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ranking matrix commands for different features and k values')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., openai)')
    args = parser.parse_args()
    
    run_ranking_matrix_commands(args.model)
