import os
import glob
from pathlib import Path

def refactor_json_files():
    """
    Recursively find all JSON files in data/output/features/rq3 folder
    and rename them by changing "_2" to "_4" in their filenames.
    """
    # Define the base directory
    base_dir = Path("data/output/features/rq3")
    
    # Check if the directory exists
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist!")
        return
    
    # Find all JSON files recursively
    json_files = list(base_dir.rglob("*.json"))
    
    if not json_files:
        print("No JSON files found in the specified directory.")
        return
    
    print(f"Found {len(json_files)} JSON files to process.")
    
    # Counter for renamed files
    renamed_count = 0
    
    # Process each JSON file
    for json_file in json_files:
        filename = json_file.name
        
        # Check if the filename contains "_2"
        if "_2" in filename:
            # Create new filename by replacing "_0" with "_2"
            new_filename = filename.replace("_2", "_4")
            new_filepath = json_file.parent / new_filename
            
            try:
                # Rename the file
                json_file.rename(new_filepath)
                print(f"Renamed: {json_file} -> {new_filepath}")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {json_file}: {e}")
        else:
            print(f"Skipped (no '_2' found): {json_file}")
    
    print(f"\nRefactoring complete! Renamed {renamed_count} files.")

if __name__ == "__main__":
    refactor_json_files() 