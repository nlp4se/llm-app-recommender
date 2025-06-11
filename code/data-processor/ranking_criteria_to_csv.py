# Save this as ranking_criteria_to_csv.py
import json
import csv
import os
import argparse
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def clean_source(source):
    """Clean a source string by removing trailing punctuation and whitespace."""
    return source.strip().rstrip('.,;:')

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
            # Clean each source and join
            sources = ', '.join(clean_source(s) for s in sources if s)
        elif not sources:
            sources = ''
        
        # Add one row with all sources combined
        criteria_data.append({**base_criterion, 'sources': sources})
    
    return criteria_data

def find_most_representative_description(descriptions):
    """Find the most representative description using TF-IDF and cosine similarity."""
    if not descriptions:
        return ""
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # Calculate cosine similarity between all descriptions
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find the description with highest average similarity to others
        avg_similarities = np.mean(similarity_matrix, axis=1)
        most_representative_idx = np.argmax(avg_similarities)
        
        return descriptions[most_representative_idx]
    except:
        # If vectorization fails, return the first description
        return descriptions[0]

def process_directory(input_folder, output_file_all, output_file_grouped):
    """Process all JSON files in the input folder and write to two CSV files."""
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
    
    # Write all criteria to first CSV
    if all_criteria:
        fieldnames = ['name', 'description', 'type', 'sources']
        with open(output_file_all, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_criteria)
        print(f"Successfully wrote {len(all_criteria)} criteria to {output_file_all}")
    else:
        print("No criteria found in the input files")
    
    # Group and process criteria for second CSV
    grouped_criteria = defaultdict(lambda: {'descriptions': [], 'types': set(), 'sources': set()})
    
    for criterion in all_criteria:
        # Use lowercase name for grouping
        name = criterion['name'].lower()
        grouped_criteria[name]['descriptions'].append(criterion['description'])
        if criterion['type']:
            grouped_criteria[name]['types'].add(criterion['type'])
        if criterion['sources']:
            # Clean and split sources
            sources = [clean_source(s) for s in criterion['sources'].split(',') if s.strip()]
            grouped_criteria[name]['sources'].update(sources)
    
    # Create grouped criteria list
    grouped_list = []
    for name, data in sorted(grouped_criteria.items()):
        representative_desc = find_most_representative_description(data['descriptions'])
        grouped_list.append({
            'name': name,  # Keep the lowercase version for consistency
            'description': representative_desc,
            'type': ', '.join(sorted(data['types'])),
            'sources': ', '.join(sorted(data['sources']))
        })
    
    # Write grouped criteria to second CSV
    if grouped_list:
        with open(output_file_grouped, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(grouped_list)
        print(f"Successfully wrote {len(grouped_list)} grouped criteria to {output_file_grouped}")
    else:
        print("No criteria found to group")

def main():
    parser = argparse.ArgumentParser(description='Convert ranking criteria from JSON files to CSV')
    parser.add_argument('--input-folder', required=True, help='Input folder containing JSON files')
    parser.add_argument('--output-file-all', required=True, help='Output CSV file path for all criteria')
    parser.add_argument('--output-file-grouped', required=True, help='Output CSV file path for grouped criteria')
    
    args = parser.parse_args()
    
    # Validate input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        return
    
    # Create output directories if they don't exist
    for output_file in [args.output_file_all, args.output_file_grouped]:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    process_directory(args.input_folder, args.output_file_all, args.output_file_grouped)

if __name__ == '__main__':
    main()