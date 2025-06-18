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

def merge_exact_name_matches(criteria_list: List[Dict], min_occurrences: int = 1) -> List[Dict]:
    """Merge criteria with exactly the same name, keeping most representative description and type."""
    # Group criteria by name
    name_groups = defaultdict(list)
    for criterion in criteria_list:
        name_groups[criterion['name'].lower()].append(criterion)
    
    merged_criteria = []
    for name, group in name_groups.items():
        if len(group) < min_occurrences:
            # Skip if not enough occurrences
            continue
            
        # Find most representative description and type
        descriptions = [c['description'] for c in group]
        types = [c['type'] for c in group]
        
        best_desc = find_most_representative_description(descriptions)
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
            'description': best_desc,
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
    
    # Merge exact name matches for each count
    return {count: merge_exact_name_matches(criteria) 
            for count, criteria in result.items()}

def process_files(input_files: List[str]):
    """Process multiple input files and create corresponding output files."""
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
                    folder = os.path.dirname(input_path)  # Use the input directory as the folder
                    folder_criteria[folder].extend(criteria_data)
                    
                    # Add to feature criteria (lowest level subfolder)
                    feature_dir = os.path.dirname(json_file)
                    feature_criteria[feature_dir].extend(criteria_data)
                    
                    # Add to all criteria
                    all_criteria.extend(criteria_data)
                    
                    # Track which subfolder each criterion appears in
                    for criterion in criteria_data:
                        criterion_subfolders[criterion['name'].lower()].add(feature_dir)
                    
                except Exception as e:
                    print(f"Error processing {json_file}: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"Error processing directory {input_path}: {str(e)}")
            continue
    
    if not all_criteria:
        print("No criteria found in any input files")
        return
    
    # Process each folder (input directory)
    for folder, criteria in folder_criteria.items():
        # Create output files in the same folder
        folder_all_file = os.path.join(folder, 'all_criteria.csv')
        folder_grouped_file = os.path.join(folder, 'grouped_criteria.csv')
        
        # Write all criteria for this folder (with deduplication)
        unique_criteria = merge_exact_name_matches(criteria)
        write_criteria_to_csv(unique_criteria, folder_all_file)
        
        # Write grouped criteria for this folder (only those appearing in multiple files)
        file_names = [f for f in file_criteria.keys() if f.startswith(folder)]
        criteria_occurrences = defaultdict(int)
        for file_name in file_names:
            for criterion in file_criteria[file_name]:
                criteria_occurrences[criterion['name'].lower()] += 1
        
        # Filter criteria that appear in multiple files
        multi_file_criteria = [c for c in criteria if criteria_occurrences[c['name'].lower()] > 1]
        merged_criteria = merge_exact_name_matches(multi_file_criteria)
        write_criteria_to_csv(merged_criteria, folder_grouped_file)
        
        # For folder level - find criteria present in at least N subfolders
        folder_subfolders = {f for f in feature_criteria.keys() if f.startswith(folder)}
        subfolder_criteria = defaultdict(set)
        for criterion in criteria:
            criterion_name = criterion['name'].lower()
            subfolder_criteria[criterion_name].update(criterion_subfolders[criterion_name].intersection(folder_subfolders))
        
        def get_subfolder_count(criterion_name):
            return len(subfolder_criteria[criterion_name])
        
        # Get criteria for different occurrence counts
        occurrence_counts = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # Can be easily modified to include more counts
        subfolder_criteria_by_count = get_criteria_by_occurrence_count(
            criteria, occurrence_counts, get_subfolder_count)
        
        # Write files for each occurrence count
        for count, filtered_criteria in subfolder_criteria_by_count.items():
            output_file = os.path.join(folder, f'grouped_all_criteria_min{count}.csv')
            write_criteria_to_csv(filtered_criteria, output_file)
    
    # Process global criteria
    global_all_file = 'global_all_criteria.csv'
    global_grouped_file = 'global_grouped_criteria.csv'
    
    # Write all global criteria (with deduplication)
    unique_global_criteria = merge_exact_name_matches(all_criteria)
    write_criteria_to_csv(unique_global_criteria, global_all_file)
    
    # Write grouped global criteria (only those appearing in multiple folders)
    folder_occurrences = defaultdict(int)
    for criterion in all_criteria:
        for folder in folder_criteria.keys():
            if any(c['name'].lower() == criterion['name'].lower() for c in folder_criteria[folder]):
                folder_occurrences[criterion['name'].lower()] += 1
    
    # Filter criteria that appear in multiple folders
    multi_folder_criteria = [c for c in all_criteria if folder_occurrences[c['name'].lower()] > 1]
    merged_global_criteria = merge_exact_name_matches(multi_folder_criteria)
    write_criteria_to_csv(merged_global_criteria, global_grouped_file)

    # For global level - find criteria present in at least N input folders
    all_folders = set(folder_criteria.keys())
    folder_criteria_count = defaultdict(int)
    for criterion in all_criteria:
        criterion_name = criterion['name'].lower()
        for folder in all_folders:
            if any(c['name'].lower() == criterion_name for c in folder_criteria[folder]):
                folder_criteria_count[criterion_name] += 1
    
    def get_folder_count(criterion_name):
        return folder_criteria_count[criterion_name]
    
    # Get criteria for different occurrence counts
    global_criteria_by_count = get_criteria_by_occurrence_count(
        all_criteria, occurrence_counts, get_folder_count)
    
    # Write files for each occurrence count
    for count, filtered_criteria in global_criteria_by_count.items():
        output_file = f'global_grouped_all_criteria_min{count}.csv'
        write_criteria_to_csv(filtered_criteria, output_file)

    print_criteria_tables("data/output/features")

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
    
    args = parser.parse_args()
    
    # Validate input files exist
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist")
            return
    
    process_files(args.input_files)

if __name__ == '__main__':
    main()