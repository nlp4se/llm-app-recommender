# Save this as ranking_criteria_to_csv.py
import json
import csv
import os
import argparse
from collections import defaultdict
from typing import List, Dict

def extract_model_from_path(input_folder: str) -> str:
    """Extract model name from file path by finding which input folder it belongs to."""
    return os.path.basename(input_folder)

def extract_feature_from_path(subfolder: str) -> str:
    """Extract feature name from subfolder path.
    Splits by '_' and takes everything after the first '_'.
    Example: 'k20_Watch_streams' -> 'Watch streams'
    """
    subfolder_name = os.path.basename(subfolder)
    if '_' in subfolder_name:
        # Split by '_' and join everything after the first '_'
        parts = subfolder_name.split('_', 1)
        if len(parts) > 1:
            return parts[1].replace('_', ' ')
    return subfolder_name

def process_json_file(file_path, input_folder):
    """Process a single JSON file and extract ranking criteria."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract metadata
    model = extract_model_from_path(input_folder)
    subfolder = os.path.dirname(file_path)
    feature = extract_feature_from_path(subfolder)
    
    criteria_data = []
    for criterion in data.get('c', []):
        # Get the base criterion data with metadata first
        base_criterion = {
            'model': model,
            'feature': feature,
            'name': criterion.get('n', ''),
            'description': criterion.get('d', ''),
            'type': criterion.get('t', '')
        }
        
        # Get sources and combine them into a single comma-separated string
        sources = criterion.get('s', [])
        if isinstance(sources, list):
            # Clean each source and join
            sources = ', '.join(s.strip().rstrip('.,;:') for s in sources if s)
        elif not sources:
            sources = ''
        
        # Add one row with all sources combined
        criteria_data.append({**base_criterion, 'sources': sources})
    
    return criteria_data

def write_criteria_to_csv(criteria: List[Dict], output_file: str):
    """Write criteria to CSV file."""
    fieldnames = ['model', 'feature', 'name', 'description', 'type', 'sources']
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(criteria)
    print(f"Successfully wrote {len(criteria)} criteria to {output_file}")

def find_json_files(directory: str) -> List[str]:
    """Recursively find all JSON files in a directory and its subdirectories."""
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def process_files(input_folders: List[str], output_folder: str = '.'):
    """Process multiple input folders and create CSV files at each folder level."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Dictionary to store criteria by folder
    folder_criteria = defaultdict(list)
    # Dictionary to store criteria by subfolder
    subfolder_criteria = defaultdict(list)
    # List to store all criteria across all folders
    all_criteria = []
    
    # Process each input folder
    for input_folder in input_folders:
        try:
            print(f"Processing folder: {input_folder}")
            
            # Find all JSON files in this folder and its subfolders
            json_files = find_json_files(input_folder)
            
            if not json_files:
                print(f"No JSON files found in {input_folder}")
                continue
                
            # Process each JSON file
            for json_file in json_files:
                try:
                    print(f"  Processing file: {json_file}")
                    criteria_data = process_json_file(json_file, input_folder)
                    
                    # Add to folder criteria (all files in this input folder)
                    folder_criteria[input_folder].extend(criteria_data)
                    
                    # Add to subfolder criteria (files in specific subfolder)
                    subfolder = os.path.dirname(json_file)
                    subfolder_criteria[subfolder].extend(criteria_data)
                    
                    # Add to all criteria
                    all_criteria.extend(criteria_data)
                    
                except Exception as e:
                    print(f"Error processing {json_file}: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"Error processing folder {input_folder}: {str(e)}")
            continue
    
    if not all_criteria:
        print("No criteria found in any input files")
        return
    
    # 1. Create root CSV in output folder (all criteria from all input folders)
    root_csv_path = os.path.join(output_folder, 'rc_extracted.csv')
    write_criteria_to_csv(all_criteria, root_csv_path)
    
    # 2. Create CSV for each input folder (criteria from that folder and its subfolders)
    for folder, criteria in folder_criteria.items():
        folder_csv_path = os.path.join(folder, 'rc_extracted.csv')
        write_criteria_to_csv(criteria, folder_csv_path)
    
    # 3. Create CSV for each subfolder (criteria from that specific subfolder)
    for subfolder, criteria in subfolder_criteria.items():
        subfolder_csv_path = os.path.join(subfolder, 'rc_extracted.csv')
        write_criteria_to_csv(criteria, subfolder_csv_path)

def main():
    parser = argparse.ArgumentParser(description='Extract ranking criteria from JSON files to CSV')
    parser.add_argument('--input-folders', nargs='+', required=True, help='List of input folders to process')
    parser.add_argument('--output-folder', default='.', help='Output folder for root CSV file (default: current directory)')
    
    args = parser.parse_args()
    
    # Validate input folders exist
    for input_folder in args.input_folders:
        if not os.path.exists(input_folder):
            print(f"Error: Input folder '{input_folder}' does not exist")
            return
    
    process_files(args.input_folders, args.output_folder)

if __name__ == '__main__':
    main()