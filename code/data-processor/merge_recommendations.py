import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def extract_json_files_from_folders(input_folders: List[str]) -> List[Dict[str, Any]]:
    """
    Extract all JSON files from all subfolders within each input folder.
    
    Args:
        input_folders: List of input folder paths
        
    Returns:
        List of dictionaries containing file information and data
    """
    all_files_data = []
    
    for input_folder in input_folders:
        input_path = Path(input_folder)
        
        if not input_path.exists():
            print(f"Warning: Input folder {input_folder} does not exist. Skipping.")
            continue
            
        # Walk through all subdirectories
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = Path(root) / file
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        
                        # Calculate relative paths
                        rel_root = file_path.relative_to(input_path)
                        subfolder = rel_root.parent.name if rel_root.parent != Path('.') else ''
                        
                        # Transform field names and values
                        model = input_path.name  # Get the tail of input folder
                        
                        # Transform feature: replace "_" with " " and keep all items except the first
                        if subfolder:
                            feature_parts = subfolder.split("_")
                            if len(feature_parts) > 1:
                                feature = " ".join(feature_parts[1:])  # Skip first item, join rest with spaces
                            else:
                                feature = subfolder
                        else:
                            feature = ""
                        
                        # Transform run: keep only the value after "_" in the file name
                        if "_" in file:
                            run = file.split("_")[-1].replace(".json", "")
                        else:
                            run = file.replace(".json", "")
                        
                        file_info = {
                            'model': model,
                            'feature': feature,
                            'run': run,
                            'json_data': json_data,
                            'file_path': str(file_path)
                        }
                        
                        all_files_data.append(file_info)
                        
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
    
    return all_files_data


def generate_app_rankings_csv(files_data: List[Dict[str, Any]], output_folder: str) -> None:
    """
    Generate app_rankings.csv with one row per file and columns for each app in array 'a'.
    
    Args:
        files_data: List of file data dictionaries
        output_folder: Output folder path
    """
    rows = []
    
    for file_info in files_data:
        json_data = file_info['json_data']
        
        # Check if 'a' array exists
        if 'a' not in json_data or not isinstance(json_data['a'], list):
            print(f"Warning: No 'a' array found in {file_info['file_path']}")
            continue
        
        # Create row with file information
        row = {
            'model': file_info['model'],
            'feature': file_info['feature'],
            'run': file_info['run']
        }
        
        # Add columns for each app (1 to 20)
        apps = json_data['a']
        for i, app in enumerate(apps, 1):
            row[str(i)] = app
        
        # Fill remaining columns (if less than 20 apps) with empty strings
        for i in range(len(apps) + 1, 21):
            row[str(i)] = ''
        
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    output_path = Path(output_folder) / 'app_rankings.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated app_rankings.csv with {len(rows)} rows at {output_path}")


def generate_app_ranking_criteria_csv(files_data: List[Dict[str, Any]], output_folder: str) -> None:
    """
    Generate app_ranking_criteria.csv with N rows per file (one for each criterion in array 'c').
    
    Args:
        files_data: List of file data dictionaries
        output_folder: Output folder path
    """
    rows = []
    
    for file_info in files_data:
        json_data = file_info['json_data']
        
        # Check if 'c' array exists
        if 'c' not in json_data or not isinstance(json_data['c'], list):
            print(f"Warning: No 'c' array found in {file_info['file_path']}")
            continue
        
        # Create a row for each criterion
        for criterion in json_data['c']:
            if isinstance(criterion, dict) and all(key in criterion for key in ['n', 'd', 't', 's']):
                row = {
                    'model': file_info['model'],
                    'feature': file_info['feature'],
                    'run': file_info['run'],
                    'n': criterion['n'],
                    'd': criterion['d'],
                    't': criterion['t'],
                    's': criterion['s']
                }
                rows.append(row)
            else:
                print(f"Warning: Invalid criterion format in {file_info['file_path']}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    output_path = Path(output_folder) / 'app_ranking_criteria.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated app_ranking_criteria.csv with {len(rows)} rows at {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Process JSON files and generate CSV reports')
    parser.add_argument('--input-folders', nargs='+', required=True,
                       help='List of input folders to process')
    parser.add_argument('--output-folder', required=True,
                       help='Output folder to save CSV files')
    
    args = parser.parse_args()
    
    # Create output folder if it doesn't exist
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(args.input_folders)} input folders...")
    print(f"Input folders: {args.input_folders}")
    print(f"Output folder: {args.output_folder}")
    
    # Extract all JSON files
    files_data = extract_json_files_from_folders(args.input_folders)
    print(f"Found {len(files_data)} JSON files to process")
    
    if not files_data:
        print("No JSON files found. Exiting.")
        return
    
    # Generate CSV files
    generate_app_rankings_csv(files_data, args.output_folder)
    generate_app_ranking_criteria_csv(files_data, args.output_folder)
    
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
