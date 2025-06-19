import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re
from collections import Counter
import numpy as np

def normalize_source_name(source):
    """
    Normalize source names to handle variations in capitalization, spacing, etc.
    
    Args:
        source (str): Raw source name
        
    Returns:
        str: Normalized source name
    """
    if pd.isna(source) or source == '':
        return None
    
    # Convert to string and strip whitespace
    source = str(source).strip()
    
    # Exclude sources matching TurnXSearchY or TurnXNewsY pattern (where X and Y are any digits)
    if re.match(r'^Turn\d+(Search|News)\d+$', source, re.IGNORECASE):
        return None
    
    # Remove common URL patterns and extract domain names
    if source.startswith('http'):
        # Extract domain from URL
        import urllib.parse
        try:
            parsed = urllib.parse.urlparse(source)
            source = parsed.netloc
            # Remove www. prefix
            if source.startswith('www.'):
                source = source[4:]
        except:
            pass
    
    # Common normalizations
    normalizations = {
        'google play': 'Google Play',
        'google play store': 'Google Play',
        'apple app store': 'Apple App Store',
        'app store': 'Apple App Store',
        'user review': 'User Reviews',
        'user reviews': 'User Reviews',
        'user rating': 'User Reviews',
        'user ratings': 'User Reviews',
        'multiple sources': 'Multiple Sources',
        'multiple source': 'Multiple Sources',
        'app descriptions': 'App Descriptions',
        'app description': 'App Descriptions',
        'expert reviews': 'Expert Reviews',
        'expert review': 'Expert Reviews',
        'blogs': 'Blogs',
        'blog': 'Blogs',
        'quora': 'Quora',
        'edutopia': 'Edutopia',
        'forbes': 'Forbes',
        'sensor tower': 'Sensor Tower',
        'sensor tower data': 'Sensor Tower',
        'app store analytics': 'App Store Analytics',
        'app store analytics platforms': 'App Store Analytics',
        'money.co.uk': 'Money.co.uk',
        'digicrusader': 'Digicrusader',
        'ai ixx': 'AI IXX',
        'dx talks': 'DX Talks',
        'landmark national bank': 'Landmark National Bank',
        'rockflow': 'RockFlow',
        'wallstreetzen': 'WallStreetZen',
        'magnifi': 'Magnifi',
        'incite ai': 'Incite AI',
        'osfin': 'Osfin',
        'vena solutions': 'Vena Solutions',
        'litslink': 'Litslink',
        'litslink.com': 'Litslink',
        'ideausher': 'IdeaUsher',
        'ideausher.com': 'IdeaUsher',
        'vktr.com': 'VKTR.com',
        'bestofai': 'BestofAI',
        'learnworlds': 'LearnWorlds',
        'owlfitt': 'Owlfitt',
        'squirrel ai': 'Squirrel AI',
        'gradescope': 'Gradescope',
        'schools that lead': 'Schools That Lead',
        'ssbm geneva': 'SSBM Geneva',
        'sbm geneva': 'SSBM Geneva',
        'university of san dieo online degrees': 'University of San Diego Online Degrees',
        'kitlabsinc.com': 'Kitlabsinc.com',
        'quokkalabs': 'Quokkalabs',
        'quokkalabs.com': 'Quokkalabs',
        'weblineindia.com': 'Weblineindia.com',
        'makesyoufluent.com': 'Makesyoufluent.com'
    }
    
    # Convert to lowercase for comparison
    source_lower = source.lower()
    
    # Check if we have a direct normalization
    if source_lower in normalizations:
        return normalizations[source_lower]
    
    # Handle common patterns
    if 'vertexaisearch.cloud.google.com' in source_lower:
        return 'Google Vertex AI Search'
    
    # If it's a simple domain, capitalize properly
    if '.' in source and not source.startswith('http'):
        # It's likely a domain name
        parts = source.split('.')
        if len(parts) == 2:
            return parts[0].title() + '.' + parts[1]
    
    # Default: title case
    return source.title()

def extract_sources_from_csv(input_file):
    """
    Extract and normalize sources from the CSV file.
    
    Args:
        input_file (str): Path to the CSV file
        
    Returns:
        list: List of normalized source names
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Check if 'sources' column exists
        if 'sources' not in df.columns:
            raise ValueError("CSV file does not contain a 'sources' column")
        
        all_sources = []
        
        # Process each row in the sources column
        for sources_str in df['sources']:
            if pd.isna(sources_str) or sources_str == '':
                continue
                
            # Split by comma and clean up
            sources = [s.strip() for s in str(sources_str).split(',')]
            
            # Normalize each source
            for source in sources:
                normalized_source = normalize_source_name(source)
                if normalized_source:
                    all_sources.append(normalized_source)
        
        return all_sources
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def plot_source_histogram(sources, output_file=None, figsize=(12, 8)):
    """
    Create a histogram of source frequencies.
    
    Args:
        sources (list): List of source names
        output_file (str): Optional path to save the plot
        figsize (tuple): Figure size
    """
    if not sources:
        print("No sources found to plot.")
        return
    
    # Count frequencies
    source_counts = Counter(sources)
    
    # Filter sources that appear more than once
    filtered_sources = {source: count for source, count in source_counts.items() if count > 1}
    
    if not filtered_sources:
        print("No sources found that appear more than once.")
        return
    
    # Sort by frequency (most to least)
    sorted_sources = sorted(filtered_sources.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for plotting - reverse the order so most frequent appears at top
    source_names = [item[0] for item in sorted_sources][::-1]  # Reverse the list
    counts = [item[1] for item in sorted_sources][::-1]  # Reverse the list
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Set modern style (matching ranking_criteria_to_csv.py)
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    
    # Create horizontal bar plot with matching colors
    bars = plt.barh(range(len(source_names)), counts, 
                   color='#2E86AB', alpha=0.8, edgecolor='#1B4965', linewidth=1.5)
    
    # Customize the plot
    plt.yticks(range(len(source_names)), source_names)
    plt.xlabel('Frequency', fontweight='bold')
    plt.ylabel('Sources', fontweight='bold')
    plt.title('Sources with Multiple Mentions', fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                str(count), ha='left', va='center', fontweight='bold', fontsize=9)
    
    # Add grid for better readability
    plt.grid(axis='x', alpha=0.3)
    
    # Remove top and right spines for cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total unique sources: {len(source_counts)}")
    print(f"Sources with multiple mentions: {len(filtered_sources)}")
    print(f"Total source mentions: {sum(source_counts.values())}")
    print(f"\nAll sources with multiple mentions:")
    for i, (source, count) in enumerate(sorted_sources):
        print(f"{i+1:2d}. {source:<30} {count:>3d} mentions")

def main():
    """
    Main function to handle command line arguments and execute the visualization.
    """
    parser = argparse.ArgumentParser(description='Visualize source frequencies from ranking criteria CSV')
    parser.add_argument('--input-file', help='Path to the CSV file containing ranking criteria')
    parser.add_argument('--output-file', help='Output file path for the plot (optional)')
    
    args = parser.parse_args()
    
    # Extract sources from CSV
    print(f"Reading sources from: {args.input_file}")
    sources = extract_sources_from_csv(args.input_file)
    
    if sources:
        print(f"Found {len(set(sources))} unique sources from {len(sources)} total mentions")
        
        # Create the visualization
        plot_source_histogram(
            sources, 
            output_file=args.output_file,
            figsize=(12, 8)
        )
    else:
        print("No sources found in the CSV file.")

if __name__ == "__main__":
    main()
