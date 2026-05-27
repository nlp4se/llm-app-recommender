#!/usr/bin/env python3
"""
Script to transform text content with numbered app lists into proper JSON format.
Extracts app names from numbered lists and converts them to JSON arrays.
"""

import re
import json
import argparse
from pathlib import Path


def extract_apps_from_text(text):
    """
    Extract app names from text that contains numbered lists with **app names**.
    Returns a list of app names.
    """
    # Pattern to match numbered items with **app names**
    # Matches: 1. **App Name** 2. **Another App** etc.
    pattern = r'\d+\.\s*\*\*([^*]+)\*\*'
    
    matches = re.findall(pattern, text)
    
    # Clean up the extracted app names
    apps = []
    for match in matches:
        app_name = match.strip()
        if app_name:
            apps.append(app_name)
    
    return apps


def transform_text_to_json(text_content):
    """
    Transform text content into JSON format.
    Returns a dictionary with the apps array.
    """
    apps = extract_apps_from_text(text_content)
    
    # Create JSON structure
    json_data = {
        "apps": apps
    }
    
    return json_data


def process_file(file_path, output_path=None):
    """
    Process a single file and transform its content to JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Transform the content
        json_data = transform_text_to_json(content)
        
        # Determine output path
        if output_path is None:
            output_path = file_path
        
        # Write the JSON data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Transformed {file_path} -> {output_path}")
        print(f"Extracted {len(json_data['apps'])} apps")
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Transform text files with numbered app lists into JSON format"
    )
    parser.add_argument(
        "input_file", 
        help="Input file to transform"
    )
    parser.add_argument(
        "--output", 
        "-o",
        help="Output file path (defaults to input file)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be transformed without writing files"
    )
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"Error: File '{args.input_file}' does not exist.")
        return
    
    if args.dry_run:
        # Read and show what would be transformed
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        json_data = transform_text_to_json(content)
        print("Would transform to:")
        print(json.dumps(json_data, indent=2, ensure_ascii=False))
        print(f"Extracted {len(json_data['apps'])} apps")
    else:
        # Actually process the file
        process_file(args.input_file, args.output)


if __name__ == "__main__":
    main() 