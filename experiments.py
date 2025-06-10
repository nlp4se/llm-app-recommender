import csv
import subprocess
import time
from itertools import product

def read_csv_values(file_path):
    """Read values from a CSV file."""
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        # Skip header if exists
        next(reader, None)
        # Get all values from the first column
        return [row[0] for row in reader]

def run_command(k, category, output_dir, n=5, model="gpt-4o-search-preview", sleep=10):
    """Run the search command with given parameters."""
    cmd = [
        "python", "-m", "code.llm.openai.search_openai_uc1",
        "--output", output_dir,
        "--k", str(k),
        "--category", category,
        "--n", str(n),
        "--model", model,
        "--sleep", str(sleep)
    ]
    
    print(f"Running command with k={k}, category={category}")
    subprocess.run(cmd)
    time.sleep(sleep)  # Wait between runs

def main():
    # Read values from CSV files
    k_values = read_csv_values("data/input/use-case/k.csv")
    categories = read_csv_values("data/input/use-case/categories.csv")
    
    # Base output directory
    base_output_dir = "./data/output/search/uc1/openai"
    
    # Run command for each combination
    for k, category in product(k_values, categories):
        # Create a subdirectory for this combination
        output_dir = f"{base_output_dir}/k{k}_{category.replace(' ', '_')}"
        
        run_command(
            k=k,
            category=category,
            output_dir=output_dir
        )

if __name__ == "__main__":
    main()