# Save this as ranking_criteria_to_csv.py
import json
import csv
import os
import argparse
from pathlib import Path

def process_json_file(file_path):
    """Process a single JSON file and extract ranking criteria."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    criteria_data = []
    for criterion in data.get('criteria', []):
        # Get the base criterion data
        base_criterion = {
            'name': criterion.get('name', ''),
            'description': criterion.get('description', ''),
            'type': criterion.get('type', '')
        }
        
        # Get sources and combine them into a single comma-separated string
        sources = criterion.get('sources', [])
        if isinstance(sources, list):
            sources = ', '.join(sources)
        elif not sources:
            sources = ''
        
        # Add one row with all sources combined
        criteria_data.append({**base_criterion, 'sources': sources})
    
    return criteria_data

def process_directory(input_folder, output_file):
    """Process all JSON files in the input folder and write to CSV."""
    all_criteria = []
    
    # Walk through all files in the input folder
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    criteria_data = process_json_file(file_path)
                    all_criteria.extend(criteria_data)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    # Write to CSV
    if all_criteria:
        fieldnames = ['name', 'description', 'type', 'sources']
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_criteria)
        print(f"Successfully wrote {len(all_criteria)} criteria to {output_file}")
    else:
        print("No criteria found in the input files")

def main():
    parser = argparse.ArgumentParser(description='Convert ranking criteria from JSON files to CSV')
    parser.add_argument('--input-folder', required=True, help='Input folder containing JSON files')
    parser.add_argument('--output-file', required=True, help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Validate input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    process_directory(args.input_folder, args.output_file)

if __name__ == '__main__':
    main()