import pandas as pd
import json
import argparse
from typing import Dict, List

def clean_and_split(value: str) -> List[str]:
    """Split string on semicolons and clean resulting items."""
    if pd.isna(value):
        return []
    return [item.strip() for item in value.split(';')]

def process_excel_to_json(input_file: str, output_file: str) -> None:
    """
    Convert Excel file with ranking criteria to JSON format.
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output JSON file
    """
    # Read Excel file
    df = pd.read_excel(input_file)
    
    # Initialize the JSON structure
    result = {
        "ranking_criteria": []
    }
    
    # Process each row
    for _, row in df.iterrows():
        criterion = {
            "name": row['name'].strip(),
            "metrics": clean_and_split(row['metrics']),
            "data_sources": clean_and_split(row['data_sources'])
        }
        result["ranking_criteria"].append(criterion)
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert ranking criteria Excel file to JSON')
    parser.add_argument('--input', '-i', required=True, help='Input Excel file path')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the file
    process_excel_to_json(args.input, args.output)
    print(f"Successfully converted {args.input} to {args.output}")

if __name__ == "__main__":
    main()
