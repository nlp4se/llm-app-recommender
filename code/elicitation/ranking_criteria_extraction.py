# Save this as ranking_criteria_to_csv.py
import json
import csv
import os
import argparse
from pathlib import Path
from typing import List, Dict

from code.experiments.io import list_experiment_json_files
from code.experiments.naming import parse_experiment_filename


def process_json_file(file_path: str, rq_id: str) -> List[Dict]:
    """Process a single JSON file and extract ranking criteria."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    meta = parse_experiment_filename(Path(file_path).name, rq_id)

    criteria_data = []
    for criterion in data.get("c", []):
        base_criterion = {
            "family": meta["family"],
            "model": meta["model"],
            "provider": meta["provider"],
            "mode": meta["mode"],
            "k": meta["k"],
            "feature": meta["feature"],
            "run": meta["run"],
            "name": criterion.get("n", ""),
            "description": criterion.get("d", ""),
            "type": criterion.get("t", ""),
        }

        sources = criterion.get("s", [])
        if isinstance(sources, list):
            sources = ", ".join(s.strip().rstrip(".,;:") for s in sources if s)
        elif not sources:
            sources = ""

        criteria_data.append({**base_criterion, "sources": sources})

    return criteria_data


def write_criteria_to_csv(criteria: List[Dict], output_file: str):
    """Write criteria to CSV file."""
    fieldnames = ["family", "model", "provider", "mode", "k", "feature", "run", "name", "description", "type", "sources"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(criteria)
    print(f"Successfully wrote {len(criteria)} criteria to {output_file}")


def process_files(input_folders: List[str], output_folder: str = ".", rq_id: str = "rq1"):
    """Process RQ output folders and create a consolidated CSV."""
    os.makedirs(output_folder, exist_ok=True)
    all_criteria = []

    for input_folder in input_folders:
        try:
            print(f"Processing folder: {input_folder}")
            json_files = list_experiment_json_files(input_folder)

            if not json_files:
                print(f"No JSON files found in {input_folder}")
                continue

            for json_file in json_files:
                try:
                    print(f"  Processing file: {json_file}")
                    criteria_data = process_json_file(str(json_file), rq_id)
                    all_criteria.extend(criteria_data)
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    continue

        except Exception as e:
            print(f"Error processing folder {input_folder}: {e}")
            continue

    if not all_criteria:
        print("No criteria found in any input files")
        return

    root_csv_path = os.path.join(output_folder, "rc_extracted.csv")
    write_criteria_to_csv(all_criteria, root_csv_path)


def main():
    parser = argparse.ArgumentParser(description="Extract ranking criteria from JSON files to CSV")
    parser.add_argument("--input-folders", nargs="+", required=True, help="RQ output folders to process")
    parser.add_argument("--output-folder", default=".", help="Output folder for CSV file")
    parser.add_argument("--rq", default="rq1", choices=["rq1", "rq3"], help="Research question (filename format)")

    args = parser.parse_args()

    for input_folder in args.input_folders:
        if not os.path.exists(input_folder):
            print(f"Error: Input folder '{input_folder}' does not exist")
            return

    process_files(args.input_folders, args.output_folder, args.rq)


if __name__ == "__main__":
    main()
