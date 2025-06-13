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

def run_command(k, search, output_dir, n=10, model="gpt-4o-search-preview", sleep=10):
    """Run the search command with given parameters."""
    cmd = [
        "python", "-m", "code.llm.openai.search_openai_rq1",
        "--output", output_dir,
        "--k", str(k),
        "--search", search,
        "--n", str(n),
        "--model", model,
        "--sleep", str(sleep)
    ]
    
    print(f"Running command with k={k}, search={search}")
    subprocess.run(cmd)
    time.sleep(sleep)  # Wait between runs

def main():
    # Read values from CSV files
    k_values = read_csv_values("data/input/use-case/k.csv")
    categories = read_csv_values("data/input/use-case/features.csv")
    
    # Base output directory
    base_output_dir = "./data/output/features/rq1/openai"
    
    # Run command for each combination
    for k, search in product(k_values, categories):
        # Create a subdirectory for this combination
        output_dir = f"{base_output_dir}/k{k}_{search.replace(' ', '_')}"
        
        run_command(
            k=k,
            search=search,
            output_dir=output_dir
        )

if __name__ == "__main__":
    main()