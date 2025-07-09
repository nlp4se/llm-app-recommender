import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def extract_json_from_file_content(file_content: str) -> Dict[str, Any]:
    """
    Extract JSON content from file content. First tries to parse the entire content as JSON.
    If that fails, looks for markdown code blocks with ```json and parses the content between ```json and ```.
    If no closing ``` is found, extracts everything from ```json to the end of the file.
    If no markdown tags are found, tries to extract JSON from the beginning until the first non-JSON content.
    
    Args:
        file_content: The full content of the file as string
        
    Returns:
        Parsed JSON data as dictionary
        
    Raises:
        ValueError: If JSON parsing fails in all attempts
    """
    # First, try to parse the entire content as JSON
    try:
        return json.loads(file_content)
    except json.JSONDecodeError:
        # If that fails, look for markdown code block pattern
        start_pattern = "```json"
        end_pattern = "```"
        
        # Find the start pattern in the content
        start_index = file_content.find(start_pattern)
        
        if start_index != -1:
            # Found markdown tags, extract content
            end_index = file_content.find(end_pattern, start_index + len(start_pattern))
            
            if end_index == -1:
                # No closing ``` found, extract everything from ```json to the end
                json_content = file_content[start_index + len(start_pattern):].strip()
            else:
                # Extract content between the patterns
                json_content = file_content[start_index + len(start_pattern):end_index].strip()
        else:
            # No markdown tags found, try to extract JSON from the beginning
            # Look for common non-JSON patterns that might indicate the end of JSON
            lines = file_content.split('\n')
            json_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                if line.startswith('Explanation:') or line.startswith('Note:') or line.startswith('Based on') or line.startswith('**Explanation:**'):
                    break
                json_lines.append(line)
            
            json_content = '\n'.join(json_lines).strip()
        
        if not json_content:
            raise ValueError("No content found to parse as JSON")
        
        # Try to parse as JSON
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON content: {e}")


def extract_json_files_from_folders(input_folders: List[str]) -> tuple[List[Dict[str, Any]], int]:
    """
    Extract all JSON files from all subfolders within each input folder.
    
    Args:
        input_folders: List of input folder paths
        
    Returns:
        Tuple of (list of dictionaries containing file information and data, number of failed files)
    """
    all_files_data = []
    failed_files_count = 0
    
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
                        # Read file content as text first
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                        
                        # Try to extract and parse JSON using the pattern
                        try:
                            json_data = extract_json_from_file_content(file_content)
                        except ValueError as e:
                            print(f"Failed to extract JSON from {file_path}: {e}")
                            failed_files_count += 1
                            continue
                        
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
                            # Extract prefix: everything before the last underscore
                            prefix = "_".join(file.split("_")[:-1])
                        else:
                            run = file.replace(".json", "")
                            prefix = ""
                        
                        file_info = {
                            'model': model,
                            'feature': feature,
                            'run': run,
                            'prefix': prefix,
                            'json_data': json_data,
                            'file_path': str(file_path)
                        }
                        
                        all_files_data.append(file_info)
                        
                    except (IOError, OSError) as e:
                        print(f"Error reading {file_path}: {e}")
                        failed_files_count += 1
                        continue
    
    return all_files_data, failed_files_count


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
            'run': file_info['run'],
            'prefix': file_info['prefix']
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
    df.to_csv(output_path, index=False, quoting=1, escapechar='\\')
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
                    'prefix': file_info['prefix'],
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
    df.to_csv(output_path, index=False, quoting=1, escapechar='\\')
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
    files_data, failed_files_count = extract_json_files_from_folders(args.input_folders)
    print(f"Found {len(files_data)} JSON files to process")
    print(f"Failed to parse {failed_files_count} files")
    
    if not files_data:
        print("No JSON files found. Exiting.")
        return
    
    # Generate CSV files
    generate_app_rankings_csv(files_data, args.output_folder)
    #generate_app_ranking_criteria_csv(files_data, args.output_folder)
    
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
