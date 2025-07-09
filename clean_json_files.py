#!/usr/bin/env python3
"""
Script to clean JSON files by removing bracketed numbers and parenthetical content from app names.
Iterates through all JSON files in subfolders and removes [x, y, ...] patterns and (content) patterns.
Also handles files that contain JSON mixed with other content by extracting only the JSON part.
"""

import os
import json
import re
from pathlib import Path
import argparse


def clean_app_name(app_name):
    """
    Remove bracketed numbers and parenthetical content from app name.
    Examples:
    - "YouTube Live [2, 5, 7, 13, 14, 15, 21, 24]" -> "YouTube Live"
    - "Twitch [2, 5, 7, 14, 17, 21, 24]" -> "Twitch"
    - "WeTransfer [1, 3]" -> "WeTransfer"
    - "Netflix (for movies)" -> "Netflix"
    - "App Name [1][3][4]" -> "App Name"
    - "OBS Studio" -> "OBS Studio" (no change if no brackets or parentheses)
    """
    # Remove pattern like [2, 5, 7, 13, 14, 15, 21, 24] from anywhere in the string
    cleaned = re.sub(r'\s*\[[\d,\s]+\]\s*', ' ', app_name.strip())
    
    # Remove pattern like [1][3][4] (consecutive single brackets) from anywhere in the string
    cleaned = re.sub(r'\s*\[\d+\]\s*', ' ', cleaned)
    
    # Remove pattern like (for movies) from anywhere in the string
    cleaned = re.sub(r'\s*\([^)]*\)\s*', ' ', cleaned)
    
    # Clean up any extra whitespace that might result from the removals
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def clean_text_content(content):
    """
    Clean text content by removing bracketed numbers and parenthetical content.
    This is used when JSON parsing fails and we need to treat the file as text.
    """
    # Remove pattern like [2, 5, 7, 13, 14, 15, 21, 24] from anywhere in the string
    cleaned = re.sub(r'\s*\[[\d,\s]+\]\s*', ' ', content)
    
    # Remove pattern like [1][3][4] (consecutive single brackets) from anywhere in the string
    cleaned = re.sub(r'\s*\[\d+\]\s*', ' ', cleaned)
    
    # Remove pattern like (for movies) from anywhere in the string
    cleaned = re.sub(r'\s*\([^)]*\)\s*', ' ', cleaned)
    
    # Clean up any extra whitespace that might result from the removals
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def extract_json_from_content(content):
    """
    Extract JSON from content that may contain JSON mixed with other text.
    Returns the extracted JSON string if found, None otherwise.
    """
    # Look for JSON object patterns
    # Pattern 1: { ... } - complete JSON object
    json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_object_pattern, content, re.DOTALL)
    
    for match in matches:
        try:
            # Try to parse as JSON to validate
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue
    
    # Pattern 2: [ ... ] - JSON array
    json_array_pattern = r'\[[^\[\]]*(?:\{[^{}]*\}[^\[\]]*)*\]'
    matches = re.findall(json_array_pattern, content, re.DOTALL)
    
    for match in matches:
        try:
            # Try to parse as JSON to validate
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue
    
    return None


def process_json_file(file_path):
    """
    Process a single JSON file and clean app names.
    If JSON parsing fails, try to extract JSON from mixed content.
    Returns True if file was modified, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse as JSON first
        try:
            data = json.loads(content)
            modified = False
            
            # Handle different JSON structures
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        # Clean each item in the list
                        for i, item in enumerate(value):
                            if isinstance(item, str):
                                cleaned_item = clean_app_name(item)
                                if cleaned_item != item:
                                    value[i] = cleaned_item
                                    modified = True
                    elif isinstance(value, str):
                        # Direct string value
                        cleaned_value = clean_app_name(value)
                        if cleaned_value != value:
                            data[key] = cleaned_value
                            modified = True
            
            elif isinstance(data, list):
                # Direct list of strings
                for i, item in enumerate(data):
                    if isinstance(item, str):
                        cleaned_item = clean_app_name(item)
                        if cleaned_item != item:
                            data[i] = cleaned_item
                            modified = True
            
            # Write back to file if modified
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
            
            return False
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from mixed content
            extracted_json = extract_json_from_content(content)
            
            if extracted_json:
                print(f"Warning: {file_path} contains mixed content, extracting JSON")
                
                try:
                    data = json.loads(extracted_json)
                    modified = False
                    
                    # Handle different JSON structures
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, list):
                                # Clean each item in the list
                                for i, item in enumerate(value):
                                    if isinstance(item, str):
                                        cleaned_item = clean_app_name(item)
                                        if cleaned_item != item:
                                            value[i] = cleaned_item
                                            modified = True
                            elif isinstance(value, str):
                                # Direct string value
                                cleaned_value = clean_app_name(value)
                                if cleaned_value != value:
                                    data[key] = cleaned_value
                                    modified = True
                    
                    elif isinstance(data, list):
                        # Direct list of strings
                        for i, item in enumerate(data):
                            if isinstance(item, str):
                                cleaned_item = clean_app_name(item)
                                if cleaned_item != item:
                                    data[i] = cleaned_item
                                    modified = True
                    
                    # Write back to file (always write the extracted JSON, even if not modified)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    return True
                    
                except json.JSONDecodeError:
                    print(f"Error: Could not parse extracted JSON from {file_path}")
                    return False
            else:
                # If no JSON found, treat as text file
                print(f"Warning: {file_path} is not valid JSON and no JSON found, treating as text file")
                
                # Clean the entire content as text
                cleaned_content = clean_text_content(content)
                
                if cleaned_content != content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    return True
                
                return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def find_json_files(root_dir):
    """
    Recursively find all JSON files in the given directory and subdirectories.
    """
    json_files = []
    root_path = Path(root_dir)
    
    for file_path in root_path.rglob("*.json"):
        json_files.append(file_path)
    
    return json_files


def check_file_needs_cleanup(file_path):
    """
    Check if a file needs cleanup (for dry-run mode).
    Returns True if cleanup is needed, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse as JSON first
        try:
            data = json.loads(content)
            
            has_cleanup_needed = False
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and (
                                re.search(r'\s*\[[\d,\s]+\]\s*', item) or 
                                re.search(r'\s*\[\d+\]\s*', item) or
                                re.search(r'\s*\([^)]*\)\s*', item)
                            ):
                                has_cleanup_needed = True
                                break
                    elif isinstance(value, str) and (
                        re.search(r'\s*\[[\d,\s]+\]\s*', value) or 
                        re.search(r'\s*\[\d+\]\s*', value) or
                        re.search(r'\s*\([^)]*\)\s*', value)
                    ):
                        has_cleanup_needed = True
                        break
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, str) and (
                        re.search(r'\s*\[[\d,\s]+\]\s*', item) or 
                        re.search(r'\s*\[\d+\]\s*', item) or
                        re.search(r'\s*\([^)]*\)\s*', item)
                    ):
                        has_cleanup_needed = True
                        break
            
            return has_cleanup_needed
            
        except json.JSONDecodeError:
            # Try to extract JSON from mixed content
            extracted_json = extract_json_from_content(content)
            if extracted_json:
                try:
                    data = json.loads(extracted_json)
                    has_cleanup_needed = False
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, list):
                                for item in value:
                                    if isinstance(item, str) and (
                                        re.search(r'\s*\[[\d,\s]+\]\s*', item) or 
                                        re.search(r'\s*\[\d+\]\s*', item) or
                                        re.search(r'\s*\([^)]*\)\s*', item)
                                    ):
                                        has_cleanup_needed = True
                                        break
                            elif isinstance(value, str) and (
                                re.search(r'\s*\[[\d,\s]+\]\s*', value) or 
                                re.search(r'\s*\[\d+\]\s*', value) or
                                re.search(r'\s*\([^)]*\)\s*', value)
                            ):
                                has_cleanup_needed = True
                                break
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, str) and (
                                re.search(r'\s*\[[\d,\s]+\]\s*', item) or 
                                re.search(r'\s*\[\d+\]\s*', item) or
                                re.search(r'\s*\([^)]*\)\s*', item)
                            ):
                                has_cleanup_needed = True
                                break
                    return has_cleanup_needed
                except json.JSONDecodeError:
                    pass
            
            # If JSON parsing fails, check if text content needs cleanup
            return bool(
                re.search(r'\s*\[[\d,\s]+\]\s*', content) or 
                re.search(r'\s*\[\d+\]\s*', content) or
                re.search(r'\s*\([^)]*\)\s*', content)
            )
        
    except Exception as e:
        print(f"Error reading: {file_path} - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Clean JSON files by removing bracketed numbers and parenthetical content from app names"
    )
    parser.add_argument(
        "directory", 
        help="Root directory to search for JSON files"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be changed without actually modifying files"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return
    
    print(f"Searching for JSON files in: {args.directory}")
    json_files = find_json_files(args.directory)
    print(f"Found {len(json_files)} JSON files")
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print("Files that would be modified:")
    
    modified_count = 0
    error_count = 0
    
    for file_path in json_files:
        if args.verbose:
            print(f"Processing: {file_path}")
        
        if args.dry_run:
            # For dry run, just check if file would be modified
            if check_file_needs_cleanup(file_path):
                print(f"  Would modify: {file_path}")
                modified_count += 1
        else:
            # Actually process the file
            if process_json_file(file_path):
                print(f"Modified: {file_path}")
                modified_count += 1
    
    print(f"\nSummary:")
    print(f"  Total files processed: {len(json_files)}")
    print(f"  Files {'that would be ' if args.dry_run else ''}modified: {modified_count}")
    if error_count > 0:
        print(f"  Errors encountered: {error_count}")


if __name__ == "__main__":
    main() 