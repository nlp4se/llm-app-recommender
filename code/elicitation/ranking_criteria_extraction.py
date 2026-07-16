# Save this as ranking_criteria_to_csv.py
import csv
import os
import argparse
from pathlib import Path
from typing import List, Dict

from code.experiments.io import expand_bundled_folders


def write_criteria_to_csv(criteria: List[Dict], output_file: str):
    """Write criteria to CSV file."""
    fieldnames = ["family", "model", "provider", "mode", "k", "feature", "run", "name", "description", "type", "sources"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(criteria)
    print(f"Successfully wrote {len(criteria)} criteria to {output_file}")


def extract_criteria_from_rows(rows: list[dict]) -> list[dict]:
    """Flatten bundled experiment rows into criterion records."""
    all_criteria: list[dict] = []
    for row in rows:
        payload = row.get("json_data") or {}
        for criterion in payload.get("c", []):
            if not isinstance(criterion, dict):
                continue
            sources = criterion.get("s", [])
            if isinstance(sources, list):
                sources = ", ".join(s.strip().rstrip(".,;:") for s in sources if s)
            elif not sources:
                sources = ""

            all_criteria.append(
                {
                    "family": row["family"],
                    "model": row["model"],
                    "provider": row["provider"],
                    "mode": row["mode"],
                    "k": row["k"],
                    "feature": row["feature"],
                    "run": row["run"],
                    "name": criterion.get("n", ""),
                    "description": criterion.get("d", ""),
                    "type": criterion.get("t", ""),
                    "sources": sources,
                }
            )
    return all_criteria


def extract_criteria_from_folders(
    input_folders: List[str],
    output_folder: str,
    rq_id: str = "rq1",
) -> str | None:
    """Expand bundled RQ outputs and write rc_extracted.csv. Returns output path or None."""
    os.makedirs(output_folder, exist_ok=True)
    rows, failed = expand_bundled_folders(input_folders, rq_id)
    if failed:
        print(f"Warning: failed to read {failed} bundled file(s)")

    all_criteria = extract_criteria_from_rows(rows)
    if not all_criteria:
        print("No criteria found in any bundled files")
        return None

    root_csv_path = os.path.join(output_folder, "rc_extracted.csv")
    write_criteria_to_csv(all_criteria, root_csv_path)
    return root_csv_path


def process_files(input_folders: List[str], output_folder: str = ".", rq_id: str = "rq1"):
    """Expand bundled RQ outputs and extract ranking criteria from payload field 'c'."""
    extract_criteria_from_folders(input_folders, output_folder, rq_id)


def main():
    parser = argparse.ArgumentParser(description="Extract ranking criteria from bundled JSON to CSV")
    parser.add_argument("--input-folders", nargs="+", required=True, help="RQ output folders to process")
    parser.add_argument("--output-folder", default=".", help="Output folder for CSV file")
    parser.add_argument("--rq", default="rq1", choices=["rq1", "rq3"], help="Research question id")

    args = parser.parse_args()

    for input_folder in args.input_folders:
        if not os.path.exists(input_folder):
            print(f"Error: Input folder '{input_folder}' does not exist")
            return

    process_files(args.input_folders, args.output_folder, args.rq)


if __name__ == "__main__":
    main()
