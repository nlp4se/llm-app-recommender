import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from code.experiments.io import expand_bundled_folders


def generate_app_rankings_csv(files_data: List[Dict[str, Any]], output_folder: str) -> None:
    """Generate app_rankings.csv with one row per record and columns for each app in array 'a'."""
    rows = []

    for file_info in files_data:
        json_data = file_info["json_data"]

        if "a" not in json_data or not isinstance(json_data["a"], list):
            print(f"Warning: No 'a' array found in {file_info['file_path']}")
            continue

        row = {
            "family": file_info["family"],
            "model": file_info["model"],
            "provider": file_info["provider"],
            "mode": file_info["mode"],
            "k": file_info["k"],
            "feature": file_info["feature"],
            "run": file_info["run"],
            "prefix": file_info["prefix"],
        }
        if file_info.get("criterion"):
            row["criterion"] = file_info["criterion"]

        apps = json_data["a"]
        for i, app in enumerate(apps, 1):
            row[str(i)] = app

        for i in range(len(apps) + 1, 21):
            row[str(i)] = ""

        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = Path(output_folder) / "app_rankings.csv"
    df.to_csv(output_path, index=False, quoting=1, escapechar="\\")
    print(f"Generated app_rankings.csv with {len(rows)} rows at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process bundled experiment JSON and generate app_rankings.csv")
    parser.add_argument("--input-folders", nargs="+", required=True, help="RQ output folders (e.g. .../rq1/open)")
    parser.add_argument("--output-folder", required=True, help="Output folder to save CSV files")
    parser.add_argument("--rq", default="rq1", choices=["rq1", "rq3"], help="Research question id")

    args = parser.parse_args()

    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(args.input_folders)} input folders...")
    print(f"Input folders: {args.input_folders}")
    print(f"Output folder: {args.output_folder}")

    files_data, failed_files_count = expand_bundled_folders(args.input_folders, args.rq)
    print(f"Expanded {len(files_data)} records from bundled files")
    print(f"Failed to read {failed_files_count} files")

    if not files_data:
        print("No bundled JSON files found. Exiting.")
        return

    generate_app_rankings_csv(files_data, args.output_folder)

    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
