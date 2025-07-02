import argparse
import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Filter ranking criteria using exact match and frequency-based methods.")
    parser.add_argument('--input-file', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output-folder', type=str, required=True, help='Folder to save output files')
    return parser.parse_args()

def exact_match_deduplication(df):
    """Step 1: Exact match deduplication"""
    before = len(df)
    df = df.drop_duplicates(subset=['name', 'description'])
    after = len(df)
    print(f"Step 1 - Exact match deduplication: {before} -> {after}")
    return df

def filter_single_occurrence_names(df):
    """Step 2: Remove all items with a 'name' value that appears only once"""
    before = len(df)
    
    # Count occurrences of each name
    name_counts = df['name'].value_counts()
    
    # Find names that appear only once
    single_occurrence_names = name_counts[name_counts == 1].index.tolist()
    
    # Remove items with single-occurrence names
    df_filtered = df[~df['name'].isin(single_occurrence_names)].reset_index(drop=True)
    after = len(df_filtered)
    
    print(f"Step 2 - Single occurrence name filtering: {before} -> {after} (removed {before - after} items)")
    return df_filtered

def filter_single_feature_names(df):
    """Step 3: Remove all items with a 'name' value that appear only with one 'feature' value"""
    before = len(df)
    
    # Group by name and count unique features for each name
    name_feature_counts = df.groupby('name')['feature'].nunique()
    
    # Find names that appear with only one feature
    single_feature_names = name_feature_counts[name_feature_counts <= 2].index.tolist()
    
    # Remove items with single-feature names
    df_filtered = df[~df['name'].isin(single_feature_names)].reset_index(drop=True)
    after = len(df_filtered)
    
    print(f"Step 3 - Single feature name filtering: {before} -> {after} (removed {before - after} items)")
    return df_filtered

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    df = pd.read_csv(args.input_file)
    counts = {}

    # Step 1: Exact match deduplication
    counts['original'] = len(df)
    df = exact_match_deduplication(df)
    counts['after_step1'] = len(df)
    
    # Step 2: Remove single occurrence names
    df = filter_single_occurrence_names(df)
    counts['after_step2'] = len(df)
    
    # Step 3: Remove single feature names
    df = filter_single_feature_names(df)
    counts['after_step3'] = len(df)

    # Reporting
    print("\n=== Filtering Report ===")
    print(f"Original count: {counts['original']}")
    print(f"After Step 1 (exact match deduplication): {counts['after_step1']}")
    print(f"After Step 2 (single occurrence name filtering): {counts['after_step2']}")
    print(f"After Step 3 (single feature name filtering): {counts['after_step3']}")
    print("========================")
    
    # Save output
    output_path = os.path.join(args.output_folder, "criteria_after_basic_filtering.csv")
    df.to_csv(output_path, index=False)
    print(f"Filtered criteria saved to {output_path}")

if __name__ == "__main__":
    main()
