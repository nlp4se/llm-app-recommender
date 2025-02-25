import json
import csv
import glob
import os
import argparse
from typing import List, Dict

def parse_args():
    parser = argparse.ArgumentParser(description='Process ranking criteria from JSON files')
    parser.add_argument('--input_folder', required=True, help='Folder containing JSON files')
    parser.add_argument('--output_folder', required=True, help='Folder for output files')
    parser.add_argument('--experiment_name', required=True, help='Experiment name prefix for JSON files')
    return parser.parse_args()

def read_json_files(input_folder: str, experiment_name: str) -> List[Dict]:
    pattern = os.path.join(input_folder, f"{experiment_name}_*.json")
    json_files = glob.glob(pattern)
    
    all_data = []
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Extract iteration number from filename
            iteration = int(file_path.split('_')[-1].split('.')[0])
            all_data.append((iteration, data))
    
    return sorted(all_data, key=lambda x: x[0])

def write_criteria_to_csv(criteria: List[Dict], output_file: str):
    fieldnames = ['name', 'metrics', 'data_sources']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for criterion in criteria:
            # Convert lists to semicolon-separated strings
            row = criterion.copy()
            row['metrics'] = '; '.join(criterion['metrics'])
            row['data_sources'] = '; '.join(criterion['data_sources'])
            writer.writerow(row)

def write_detailed_criteria_to_csv(criteria: List[Dict], output_file: str):
    """Write criteria to CSV with one row per name-metric-data_source combination."""
    fieldnames = ['id', 'name', 'metric', 'data_source']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Counter for generating IDs
        counter = 1
        
        for criterion in criteria:
            # Create a row for each metric-data_source combination
            for metric in criterion['metrics']:
                for data_source in criterion['data_sources']:
                    row = {
                        'id': f'M_{counter:03d}',  # Format as M_001, M_002, etc.
                        'name': criterion['name'],
                        'metric': metric,
                        'data_source': data_source
                    }
                    writer.writerow(row)
                    counter += 1

def combine_criteria(all_data: List[tuple]) -> List[Dict]:
    """Combine ranking criteria from all iterations, merging metrics and data_sources for duplicate names."""
    combined_criteria = {}
    for _, data in all_data:
        for criterion in data['ranking_criteria']:
            key = criterion['name']
            if key in combined_criteria:
                # Merge metrics and data_sources, removing duplicates
                combined_criteria[key]['metrics'] = list(set(combined_criteria[key]['metrics'] + criterion['metrics']))
                combined_criteria[key]['data_sources'] = list(set(combined_criteria[key]['data_sources'] + criterion['data_sources']))
            else:
                # First instance of this criterion
                combined_criteria[key] = criterion.copy()
    
    return list(combined_criteria.values())

def main():
    args = parse_args()
    
    # Read all JSON files
    all_data = read_json_files(args.input_folder, args.experiment_name)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output_folder, 'ranking_criteria')
    os.makedirs(output_dir, exist_ok=True)
    
    # Write individual iteration files
    for iteration, data in all_data:
        output_file = os.path.join(output_dir, f'ranking_criteria_{iteration}.csv')
        write_criteria_to_csv(data['ranking_criteria'], output_file)
    
    # Write aggregated file combining data from all iterations
    combined_criteria = combine_criteria(all_data)
    output_file = os.path.join(output_dir, 'ranking_criteria_aggregated.csv')
    write_criteria_to_csv(combined_criteria, output_file)

    # Write combined criteria to JSON file
    json_output_file = os.path.join(output_dir, 'ranking_criteria.json')
    json_data = {
        "ranking_criteria": [
            {
                "name": criterion["name"],
                "metrics": criterion["metrics"],
                "data_sources": criterion["data_sources"]
            }
            for criterion in combined_criteria
        ]
    }
    with open(json_output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Write detailed aggregated file with individual rows
    detailed_output_file = os.path.join(output_dir, 'ranking_criteria_detailed.csv')
    write_detailed_criteria_to_csv(combined_criteria, detailed_output_file)

if __name__ == "__main__":
    main()
