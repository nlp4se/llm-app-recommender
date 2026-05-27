import os
import glob
from pathlib import Path

def rename_json_files(input_folder):
    """
    Recursively find all JSON files ending with _0.json in the input folder
    and rename them by changing _0.json to _1.json
    """
    # Convert to Path object for easier handling
    base_path = Path(input_folder)
    
    if not base_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    
    # Find all JSON files ending with _0.json recursively
    pattern = "**/*_0.json"
    json_files = list(base_path.glob(pattern))
    
    if not json_files:
        print(f"No files ending with '_0.json' found in '{input_folder}'")
        return
    
    print(f"Found {len(json_files)} files to rename:")
    
    # Track successful and failed renames
    successful_renames = 0
    failed_renames = 0
    
    for file_path in json_files:
        try:
            # Create new filename by replacing _0.json with _1.json
            new_filename = file_path.name.replace("_0.json", "_1.json")
            new_file_path = file_path.parent / new_filename
            
            # Check if target file already exists
            if new_file_path.exists():
                print(f"Warning: Target file '{new_file_path}' already exists. Skipping '{file_path}'")
                failed_renames += 1
                continue
            
            # Rename the file
            file_path.rename(new_file_path)
            print(f"Renamed: {file_path} -> {new_file_path}")
            successful_renames += 1
            
        except Exception as e:
            print(f"Error renaming '{file_path}': {e}")
            failed_renames += 1
    
    # Print summary
    print(f"\nSummary:")
    print(f"Successfully renamed: {successful_renames} files")
    print(f"Failed to rename: {failed_renames} files")
    print(f"Total files processed: {len(json_files)}")

if __name__ == "__main__":
    # Input folder path
    input_folder = "data/output/features/rq3"
    
    print(f"Starting to rename JSON files in: {input_folder}")
    print("=" * 50)
    
    rename_json_files(input_folder)
    
    print("=" * 50)
    print("Process completed!")
