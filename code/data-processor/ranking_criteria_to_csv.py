# Save this as ranking_criteria_to_csv.py
import json
import csv
import os
import argparse
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Set, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def clean_source(source):
    """Clean a source string by removing trailing punctuation and whitespace."""
    return source.strip().rstrip('.,;:')

def process_json_file(file_path):
    """Process a single JSON file and extract ranking criteria."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    criteria_data = []
    for criterion in data.get('c', []):
        # Get the base criterion data
        base_criterion = {
            'name': criterion.get('n', ''),
            'description': criterion.get('d', ''),
            'type': criterion.get('t', '')
        }
        
        # Get sources and combine them into a single comma-separated string
        sources = criterion.get('s', [])
        if isinstance(sources, list):
            # Clean each source and join
            sources = ', '.join(clean_source(s) for s in sources if s)
        elif not sources:
            sources = ''
        
        # Add one row with all sources combined
        criteria_data.append({**base_criterion, 'sources': sources})
    
    return criteria_data

def find_most_representative_description(descriptions: List[str]) -> str:
    """Find the most representative description using TF-IDF and cosine similarity."""
    if not descriptions:
        return ""
    
    if len(descriptions) == 1:
        return descriptions[0]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        # Calculate cosine similarity between all descriptions
        similarity_matrix = cosine_similarity(tfidf_matrix)
        # Find the description with highest average similarity to others
        avg_similarities = np.mean(similarity_matrix, axis=1)
        most_representative_idx = np.argmax(avg_similarities)
        return descriptions[most_representative_idx]
    except:
        # If vectorization fails, return the longest description
        return max(descriptions, key=len)

def merge_exact_name_description_matches(criteria_list: List[Dict], min_occurrences: int = 1) -> List[Dict]:
    """Merge criteria with exactly the same name AND description, keeping most representative type."""
    # Group criteria by name AND description
    name_desc_groups = defaultdict(list)
    for criterion in criteria_list:
        # Create a key that combines name and description (both lowercase for comparison)
        key = (criterion['name'].lower(), criterion['description'].lower())
        name_desc_groups[key].append(criterion)
    
    merged_criteria = []
    for (name_lower, desc_lower), group in name_desc_groups.items():
        if len(group) < min_occurrences:
            # Skip if not enough occurrences
            continue
            
        # Find most representative type
        types = [c['type'] for c in group]
        best_type = find_most_representative_description(types)
        
        # Collect all unique sources
        all_sources = set()
        for criterion in group:
            if criterion['sources']:
                sources = [clean_source(s) for s in criterion['sources'].split(',') if s.strip()]
                all_sources.update(sources)
        
        # Create merged criterion
        merged = {
            'name': group[0]['name'],  # Use original case from first occurrence
            'description': group[0]['description'],  # Use original case from first occurrence
            'type': best_type,
            'sources': ', '.join(sorted(all_sources))
        }
        merged_criteria.append(merged)
    
    return sorted(merged_criteria, key=lambda x: x['name'].lower())

def write_criteria_to_csv(criteria: List[Dict], output_file: str):
    """Write criteria to CSV file."""
    fieldnames = ['name', 'description', 'type', 'sources']
    # Sort criteria by name (case-insensitive)
    sorted_criteria = sorted(criteria, key=lambda x: x['name'].lower())
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_criteria)
    print(f"Successfully wrote {len(criteria)} criteria to {output_file}")

def find_json_files(directory: str) -> List[str]:
    """Recursively find all JSON files in a directory and its subdirectories."""
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def get_criteria_by_occurrence_count(criteria_list, occurrence_counts, get_occurrence_count_func):
    """Helper function to get criteria that appear at least N times.
    
    Args:
        criteria_list: List of criteria to process
        occurrence_counts: List of minimum occurrence counts to check (e.g., [2, 3, 4])
        get_occurrence_count_func: Function that takes a criterion name and returns its occurrence count
    
    Returns:
        Dictionary mapping occurrence count to filtered criteria list
    """
    result = {}
    for criterion in criteria_list:
        criterion_name = criterion['name'].lower()
        count = get_occurrence_count_func(criterion_name)
        for min_count in occurrence_counts:
            if count >= min_count:
                result.setdefault(min_count, []).append(criterion)
    
    # Merge exact name AND description matches for each count
    return {count: merge_exact_name_description_matches(criteria) 
            for count, criteria in result.items()}

def read_criteria_from_csv(csv_path: str) -> List[Dict]:
    """Read criteria from a CSV file."""
    criteria = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                criteria.append(row)
    except FileNotFoundError:
        print(f"Warning: File {csv_path} not found")
    except Exception as e:
        print(f"Error reading {csv_path}: {str(e)}")
    return criteria

def create_folder_minN_visualizations(folder_criteria, occurrence_counts, output_folder):
    """Create individual histograms for each input folder showing criteria distribution by minN."""
    print("Creating individual folder minN visualizations...")
    
    # Calculate number of subplots needed
    num_folders = len(folder_criteria)
    cols = min(3, num_folders)  # Max 3 columns
    rows = (num_folders + cols - 1) // cols  # Calculate rows needed
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Handle different subplot configurations
    if num_folders == 1:
        axes = [axes]  # Single subplot
    elif rows == 1 and cols == 1:
        axes = [axes]  # Single subplot (1x1 grid)
    elif rows == 1:
        axes = axes.reshape(1, -1).flatten()  # 1D array of subplots
    else:
        axes = axes.flatten()  # Flatten 2D array to 1D
    
    # Set modern style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    
    folder_names = sorted(folder_criteria.keys())  # Sort folder names
    
    # First pass: collect all data to determine global min/max for y-axis
    all_minN_data = {}
    for folder in folder_names:
        minN_data = {}
        for count in occurrence_counts:
            minN_file = os.path.join(folder, f'grouped_all_criteria_min{count}.csv')
            try:
                criteria = read_criteria_from_csv(minN_file)
                minN_data[count] = len(criteria)
            except FileNotFoundError:
                minN_data[count] = 0
        all_minN_data[folder] = minN_data
    
    # Calculate global min and max values for y-axis
    all_values = []
    for minN_data in all_minN_data.values():
        all_values.extend(minN_data.values())
    
    if all_values:
        y_min = 0  # Always start from 0 for bar charts
        y_max = max(all_values) + 1  # Add some padding
    else:
        y_min, y_max = 0, 10  # Default range if no data
    
    # Second pass: create the plots with consistent y-axis
    for idx, folder in enumerate(folder_names):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        minN_data = all_minN_data[folder]
        
        # Create histogram for this folder
        bars = ax.bar(minN_data.keys(), minN_data.values(), 
                     color='#2E86AB', alpha=0.8, edgecolor='#1B4965', linewidth=1.5)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, minN_data.values()):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       str(value), ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Customize subplot
        # Get the last part of the folder name (after the last separator)
        folder_display_name = folder.split(os.sep)[-1] if folder != '.' else 'Root'
        # If the folder name is empty (e.g., root directory), use a default name
        if not folder_display_name:
            folder_display_name = 'Root'
        
        # Set title with larger font and more padding
        ax.set_title(f'{folder_display_name}', fontweight='bold', fontsize=14, pad=15)
        ax.set_xlabel('minN', fontweight='bold')
        ax.set_ylabel('Criteria Count', fontweight='bold')
        ax.set_xticks(list(minN_data.keys()))
        ax.grid(axis='y', alpha=0.3)
        
        # Set consistent y-axis range for all subplots
        ax.set_ylim(y_min, y_max)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide unused subplots
    for idx in range(num_folders, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    fig.suptitle('Criteria Distribution by minN Value - Individual Folders', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout with more space for titles
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.3)  # More space for titles and between subplots
    
    # Save the plot to output folder
    output_path = os.path.join(output_folder, 'folder_criteria_distribution_histograms.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_pdf = os.path.join(output_folder, 'folder_criteria_distribution_histograms.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved folder histograms as '{output_path}' and '{output_path_pdf}'")
    
    plt.show()

def create_minN_visualization_and_csv(occurrence_counts, output_folder):
    """Create a modern histogram of criteria distribution across minN values and save CSV with criteria names."""
    print("Creating global minN visualization and CSV...")
    
    # Collect data for each minN value
    minN_data = {}
    criteria_names_by_minN = {}
    
    for count in occurrence_counts:
        filename = os.path.join(output_folder, f'global_grouped_all_criteria_min{count}.csv')
        try:
            criteria = read_criteria_from_csv(filename)
            minN_data[count] = len(criteria)
            criteria_names_by_minN[count] = [c['name'] for c in criteria]
            print(f"min{count}: {len(criteria)} criteria")
        except FileNotFoundError:
            minN_data[count] = 0
            criteria_names_by_minN[count] = []
    
    # Create modern styled histogram
    plt.figure(figsize=(12, 8))
    
    # Set modern style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    
    # Create the histogram
    bars = plt.bar(minN_data.keys(), minN_data.values(), 
                   color='#2E86AB', alpha=0.8, edgecolor='#1B4965', linewidth=1.5)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, minN_data.values()):
        if value > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(value), ha='center', va='bottom', fontweight='bold')
    
    # Customize the plot
    plt.xlabel('Minimum Occurrence Count (minN)', fontweight='bold')
    plt.ylabel('Number of Criteria', fontweight='bold')
    plt.title('Global Distribution of Ranking Criteria by Minimum Occurrence Count', 
              fontweight='bold', pad=20)
    
    # Set x-axis ticks to show all minN values
    plt.xticks(list(minN_data.keys()))
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add some padding
    plt.tight_layout()
    
    # Save the plot to output folder
    output_path = os.path.join(output_folder, 'global_criteria_distribution_histogram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path_pdf = os.path.join(output_folder, 'global_criteria_distribution_histogram.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved global histogram as '{output_path}' and '{output_path_pdf}'")
    
    # Create CSV with criteria names by minN value
    create_minN_criteria_csv(criteria_names_by_minN, occurrence_counts, output_folder)
    
    plt.show()

def create_minN_criteria_csv(criteria_names_by_minN, occurrence_counts, output_folder):
    """Create a CSV file where each column is a minN value and contains the list of criteria names."""
    # Find the maximum number of criteria across all minN values
    max_criteria = max(len(names) for names in criteria_names_by_minN.values())
    
    # Create a DataFrame with minN values as columns
    data = {}
    for count in occurrence_counts:
        names = criteria_names_by_minN.get(count, [])
        # Pad with empty strings to make all columns the same length
        padded_names = names + [''] * (max_criteria - len(names))
        data[f'min{count}'] = padded_names
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    csv_filename = os.path.join(output_folder, 'criteria_by_minN_values.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Saved criteria names by minN values to '{csv_filename}'")
    
    # Also create a summary CSV with counts
    summary_data = {
        'minN_value': list(occurrence_counts),
        'criteria_count': [len(criteria_names_by_minN.get(count, [])) for count in occurrence_counts]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_filename = os.path.join(output_folder, 'criteria_summary_by_minN.csv')
    summary_df.to_csv(summary_filename, index=False)
    print(f"Saved criteria summary to '{summary_filename}'")

def merge_global_minN_files(folder_criteria, occurrence_counts, output_folder):
    """Merge all grouped_all_criteria_minN.csv files from different folders into global files."""
    print("Merging global minN files...")
    
    # For each occurrence count, collect criteria from all folders
    for count in occurrence_counts:
        all_criteria_for_count = []
        
        # Collect criteria from each folder's minN file
        for folder in sorted(folder_criteria.keys()):  # Sort folder names
            minN_file = os.path.join(folder, f'grouped_all_criteria_min{count}.csv')
            criteria = read_criteria_from_csv(minN_file)
            all_criteria_for_count.extend(criteria)
        
        if all_criteria_for_count:
            # Merge exact name AND description matches and save to output folder
            merged_criteria = merge_exact_name_description_matches(all_criteria_for_count)
            global_file = os.path.join(output_folder, f'global_grouped_all_criteria_min{count}.csv')
            write_criteria_to_csv(merged_criteria, global_file)
            print(f"Created {global_file} with {len(merged_criteria)} criteria")
    
    # First create individual folder visualizations
    create_folder_minN_visualizations(folder_criteria, occurrence_counts, output_folder)
    
    # Then create global visualization and CSV
    create_minN_visualization_and_csv(occurrence_counts, output_folder)

def process_files(input_files: List[str], output_folder: str = '.'):
    """Process multiple input files and create corresponding output files."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Dictionary to store criteria by folder
    folder_criteria = defaultdict(list)
    # Dictionary to store criteria by feature (lowest level subfolder)
    feature_criteria = defaultdict(list)
    # Dictionary to store criteria by file
    file_criteria = {}
    # Set to track all criteria across all files
    all_criteria = []
    # Dictionary to track which subfolders each criterion appears in
    criterion_subfolders = defaultdict(set)
    
    # Process each input directory
    for input_path in input_files:
        try:
            print(f"Processing directory: {input_path}")
            # Find all JSON files in the directory and its subdirectories
            json_files = find_json_files(input_path)
            
            if not json_files:
                print(f"No JSON files found in {input_path}")
                continue
                
            for json_file in json_files:
                try:
                    print(f"  Processing file: {json_file}")
                    criteria_data = process_json_file(json_file)
                    
                    # Store criteria for this file
                    file_criteria[json_file] = criteria_data
                    
                    # Add to folder criteria
                    folder = input_path
                    folder_criteria[folder].extend(criteria_data)
                    
                    # Add to feature criteria (lowest level subfolder)
                    feature_dir = os.path.dirname(json_file)
                    feature_criteria[feature_dir].extend(criteria_data)
                    
                    # Add to all criteria
                    all_criteria.extend(criteria_data)
                    
                    # Track which subfolder each criterion appears in
                    for criterion in criteria_data:
                        # Use both name and description as key for tracking
                        criterion_key = (criterion['name'].lower(), criterion['description'].lower())
                        criterion_subfolders[criterion_key].add(feature_dir)
                    
                except Exception as e:
                    print(f"Error processing {json_file}: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"Error processing directory {input_path}: {str(e)}")
            continue
    
    if not all_criteria:
        print("No criteria found in any input files")
        return
    
    # Process each folder (input directory) - sort folders for consistent output
    for folder in sorted(folder_criteria.keys()):
        criteria = folder_criteria[folder]
        
        # Create output files in the output folder with folder name prefix
        folder_name = os.path.basename(folder) if folder != '.' else 'root'
        folder_all_file = os.path.join(output_folder, f'{folder_name}_all_criteria.csv')
        
        # Save grouped criteria in the input folder itself
        folder_grouped_file = os.path.join(folder, 'grouped_criteria.csv')
        
        # Write all criteria for this folder (with deduplication by name AND description)
        unique_criteria = merge_exact_name_description_matches(criteria)
        write_criteria_to_csv(unique_criteria, folder_all_file)
        
        # Write grouped criteria for this folder (only those appearing in multiple files)
        file_names = [f for f in file_criteria.keys() if f.startswith(folder)]
        criteria_occurrences = defaultdict(int)
        for file_name in file_names:
            for criterion in file_criteria[file_name]:
                # Use both name and description as key
                criterion_key = (criterion['name'].lower(), criterion['description'].lower())
                criteria_occurrences[criterion_key] += 1
        
        # Filter criteria that appear in multiple files
        multi_file_criteria = [c for c in criteria if criteria_occurrences[(c['name'].lower(), c['description'].lower())] > 1]
        merged_criteria = merge_exact_name_description_matches(multi_file_criteria)
        write_criteria_to_csv(merged_criteria, folder_grouped_file)
        
        # For folder level - find criteria present in at least N subfolders
        folder_subfolders = {f for f in feature_criteria.keys() if f.startswith(folder)}
        subfolder_criteria = defaultdict(set)
        for criterion in criteria:
            criterion_key = (criterion['name'].lower(), criterion['description'].lower())
            subfolder_criteria[criterion_key].update(criterion_subfolders[criterion_key].intersection(folder_subfolders))
        
        def get_subfolder_count(criterion_name):
            # Find all criteria with this name (regardless of description)
            count = 0
            for key, subfolders in subfolder_criteria.items():
                if key[0] == criterion_name.lower():  # key[0] is the name
                    count += len(subfolders)
            return count
        
        # Get criteria for different occurrence counts
        occurrence_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # Can be easily modified to include more counts
        subfolder_criteria_by_count = get_criteria_by_occurrence_count(
            criteria, occurrence_counts, get_subfolder_count)
        
        # Write files for each occurrence count
        for count, filtered_criteria in subfolder_criteria_by_count.items():
            output_file = os.path.join(output_folder, f'{folder_name}_grouped_all_criteria_min{count}.csv')
            write_criteria_to_csv(filtered_criteria, output_file)
    
    # Process global criteria - only include criteria that appear in multiple folders
    global_all_file = os.path.join(output_folder, 'global_all_criteria.csv')
    global_grouped_file = os.path.join(output_folder, 'global_grouped_criteria.csv')
    
    # Write all global criteria (with deduplication by name AND description)
    unique_global_criteria = merge_exact_name_description_matches(all_criteria)
    write_criteria_to_csv(unique_global_criteria, global_all_file)
    
    # Write grouped global criteria (only those appearing in multiple folders)
    folder_occurrences = defaultdict(int)
    for criterion in all_criteria:
        criterion_key = (criterion['name'].lower(), criterion['description'].lower())
        for folder in folder_criteria.keys():
            if any(c['name'].lower() == criterion['name'].lower() and 
                   c['description'].lower() == criterion['description'].lower() 
                   for c in folder_criteria[folder]):
                folder_occurrences[criterion_key] += 1
    
    # Filter criteria that appear in multiple folders
    multi_folder_criteria = [c for c in all_criteria if folder_occurrences[(c['name'].lower(), c['description'].lower())] > 1]
    merged_global_criteria = merge_exact_name_description_matches(multi_folder_criteria)
    write_criteria_to_csv(merged_global_criteria, global_grouped_file)

    # NEW: Merge all minN files from different folders into global minN files
    occurrence_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    merge_global_minN_files(folder_criteria, occurrence_counts, output_folder)

    #print_criteria_tables("data/output/features")

def count_criteria_in_csv(csv_path):
    """Count the number of criteria (rows) in a CSV file, excluding the header."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f) - 1  # Subtract header

def print_criteria_tables(base_dir):
    """Scan output directories and print tables of criteria counts."""
    all_table = {}
    grouped_table = {}
    features = set()
    models = set()

    # Walk through the directory tree
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file in ('all_criteria.csv', 'grouped_criteria.csv'):
                # root: .../model/feature
                parts = os.path.normpath(root).split(os.sep)
                if len(parts) < 2:
                    continue  # Not enough depth
                feature = parts[-1]
                model = parts[-2]
                features.add(feature)
                models.add(model)
                csv_path = os.path.join(root, file)
                count = count_criteria_in_csv(csv_path)
                if file == 'all_criteria.csv':
                    all_table.setdefault(model, {})[feature] = count
                else:
                    grouped_table.setdefault(model, {})[feature] = count

    features = sorted(features)
    models = sorted(models)

    def print_table(table, title):
        print(f"\n{title}")
        header = ["Model"] + features
        print("\t".join(header))
        for model in models:
            row = [model] + [str(table.get(model, {}).get(feature, "")) for feature in features]
            print("\t".join(row))

    print_table(all_table, "All")
    print_table(grouped_table, "Grouped")

def count_criteria_by_occurrence():
    """Count the number of criteria in each global_grouped_all_criteria_minN.csv file."""
    counts = {}
    # Check for occurrence counts from 2 to 16
    for count in range(2, 17):
        filename = f'global_grouped_all_criteria_min{count}.csv'
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                # Count lines excluding header
                num_criteria = sum(1 for line in f) - 1
                counts[count] = num_criteria
        except FileNotFoundError:
            counts[count] = 0
    
    # Print results in a table format
    print("\nNumber of criteria by minimum occurrence count:")
    print("Occurrence Count | Number of Criteria")
    print("-" * 35)
    for count, num_criteria in sorted(counts.items()):
        print(f"{count:^15} | {num_criteria:^15}")

def main():
    parser = argparse.ArgumentParser(description='Convert ranking criteria from JSON files to CSV')
    parser.add_argument('--input-files', nargs='+', required=True, help='List of input JSON files')
    parser.add_argument('--output-folder', default='.', help='Output folder for all generated files (default: current directory)')
    
    args = parser.parse_args()
    
    # Validate input files exist
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist")
            return
    
    process_files(args.input_files, args.output_folder)

if __name__ == '__main__':
    main()