import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from code.experiments.io import list_experiment_json_files
from code.experiments.naming import parse_experiment_filename


def extract_json_from_file_content(file_content: str) -> Dict[str, Any]:
    """
    Extract JSON content from file content. First tries to parse the entire content as JSON.
    If that fails, looks for markdown code blocks with ```json and parses the content between ```json and ```.
    If no closing ``` is found, extracts everything from ```json to the end of the file.
    If no markdown tags are found, tries to extract JSON from the beginning until the first non-JSON content.
    """
    try:
        return json.loads(file_content)
    except json.JSONDecodeError:
        start_pattern = "```json"
        end_pattern = "```"

        start_index = file_content.find(start_pattern)

        if start_index != -1:
            end_index = file_content.find(end_pattern, start_index + len(start_pattern))

            if end_index == -1:
                json_content = file_content[start_index + len(start_pattern):].strip()
            else:
                json_content = file_content[start_index + len(start_pattern):end_index].strip()
        else:
            lines = file_content.split("\n")
            json_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(("Explanation:", "Note:", "Based on", "**Explanation:**")):
                    break
                json_lines.append(line)

            json_content = "\n".join(json_lines).strip()

        if not json_content:
            raise ValueError("No content found to parse as JSON")

        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON content: {e}") from e


def extract_json_files_from_folders(
    input_folders: List[str],
    rq_id: str = "rq1",
) -> tuple[List[Dict[str, Any]], int]:
    """Extract all JSON files from flat RQ output folders."""
    all_files_data = []
    failed_files_count = 0

    for input_folder in input_folders:
        input_path = Path(input_folder)

        if not input_path.exists():
            print(f"Warning: Input folder {input_folder} does not exist. Skipping.")
            continue

        family_hint = input_path.name
        for file_path in list_experiment_json_files(input_path):
            try:
                with open(file_path, encoding="utf-8") as f:
                    file_content = f.read()

                try:
                    json_data = extract_json_from_file_content(file_content)
                except ValueError as e:
                    print(f"Failed to extract JSON from {file_path}: {e}")
                    failed_files_count += 1
                    continue

                meta = parse_experiment_filename(file_path.name, rq_id)
                file_info = {
                    "family": meta["family"] or family_hint,
                    "model": meta["model"],
                    "provider": meta["provider"],
                    "mode": meta["mode"],
                    "k": meta["k"],
                    "feature": meta["feature"],
                    "run": meta["run"],
                    "prefix": meta["prefix"],
                    "criterion": meta.get("criterion"),
                    "json_data": json_data,
                    "file_path": str(file_path),
                }
                all_files_data.append(file_info)

            except (IOError, OSError, ValueError) as e:
                print(f"Error reading {file_path}: {e}")
                failed_files_count += 1
                continue

    return all_files_data, failed_files_count


def generate_app_rankings_csv(files_data: List[Dict[str, Any]], output_folder: str) -> None:
    """Generate app_rankings.csv with one row per file and columns for each app in array 'a'."""
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


def generate_app_ranking_criteria_csv(files_data: List[Dict[str, Any]], output_folder: str) -> None:
    """Generate app_ranking_criteria.csv with N rows per file (one for each criterion in array 'c')."""
    rows = []

    for file_info in files_data:
        json_data = file_info["json_data"]

        if "c" not in json_data or not isinstance(json_data["c"], list):
            print(f"Warning: No 'c' array found in {file_info['file_path']}")
            continue

        for criterion in json_data["c"]:
            if isinstance(criterion, dict) and all(key in criterion for key in ["n", "d", "t", "s"]):
                row = {
                    "family": file_info["family"],
                    "model": file_info["model"],
                    "provider": file_info["provider"],
                    "mode": file_info["mode"],
                    "k": file_info["k"],
                    "feature": file_info["feature"],
                    "run": file_info["run"],
                    "prefix": file_info["prefix"],
                    "n": criterion["n"],
                    "d": criterion["d"],
                    "t": criterion["t"],
                    "s": criterion["s"],
                }
                rows.append(row)
            else:
                print(f"Warning: Invalid criterion format in {file_info['file_path']}")

    df = pd.DataFrame(rows)
    output_path = Path(output_folder) / "app_ranking_criteria.csv"
    df.to_csv(output_path, index=False, quoting=1, escapechar="\\")
    print(f"Generated app_ranking_criteria.csv with {len(rows)} rows at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process JSON files and generate CSV reports")
    parser.add_argument("--input-folders", nargs="+", required=True, help="RQ output folders to process")
    parser.add_argument("--output-folder", required=True, help="Output folder to save CSV files")
    parser.add_argument("--rq", default="rq1", choices=["rq1", "rq3"], help="Research question (filename format)")

    args = parser.parse_args()

    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(args.input_folders)} input folders...")
    print(f"Input folders: {args.input_folders}")
    print(f"Output folder: {args.output_folder}")

    files_data, failed_files_count = extract_json_files_from_folders(args.input_folders, args.rq)
    print(f"Found {len(files_data)} JSON files to process")
    print(f"Failed to parse {failed_files_count} files")

    if not files_data:
        print("No JSON files found. Exiting.")
        return

    generate_app_rankings_csv(files_data, args.output_folder)

    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
