"""
RQ1 ranking-criteria elicitation pipeline (paper Steps 4–5 + prerequisites).

Step 0  — Extract criteria from bundled experiment JSON → rc_extracted.csv
Step 4  — Conservative filtering (exact dedup, singleton names, single-feature names)
Step 5  — SBERT + hierarchical clustering + representative voting → rc_consolidated.csv
Manual  — Inspect rc_consolidated.csv, edit → save as rc_wo_id.csv for RQ3
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pandas as pd

from code.elicitation.ranking_criteria_consolidation import run_consolidation
from code.elicitation.ranking_criteria_extraction import extract_criteria_from_folders
from code.elicitation.ranking_criteria_filtering import apply_filtering, print_filtering_report

ELICITATION_DEFAULT_OUTPUT = "data/output/features/rq1/rc/open_large"
RQ3_CRITERIA_DEFAULT_PATH = "data/output/features/rq1/rc/open_large/rc_wo_id_open_large.csv"

StepName = Literal["extract", "filter", "consolidate", "export-rq3-template", "all"]


def export_rq3_template(consolidated_csv: str, output_path: str) -> str:
    """
    Write a semicolon-separated criteria file for RQ3 (no id column).

    After manual review, copy/edit this file to rc_wo_id.csv.
    """
    df = pd.read_csv(consolidated_csv)
    cols = [c for c in ["name", "description"] if c in df.columns]
    if "name" not in cols or "description" not in cols:
        raise ValueError(f"Consolidated file must contain name and description: {consolidated_csv}")
    out = df[cols].copy()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False, sep=";")
    print(f"RQ3 template saved to {output_path} ({len(out)} criteria)")
    print("  → Manually review, then save final list as rc_wo_id.csv for run_experiments.py --rq rq3")
    return output_path


def run_criteria_elicitation(
    *,
    input_folders: list[str],
    output_folder: str = ELICITATION_DEFAULT_OUTPUT,
    rq_id: str = "rq1",
    steps: list[StepName] | None = None,
    k_range: tuple[int, int] = (2, 50),
    n_bootstrap: int = 10,
    rq3_template_path: str | None = None,
) -> dict[str, str]:
    """
    Run the criteria elicitation pipeline.

    Returns paths to produced artifacts.
    """
    steps = steps or ["all"]
    if "all" in steps:
        steps = ["extract", "filter", "consolidate", "export-rq3-template"]

    os.makedirs(output_folder, exist_ok=True)
    artifacts: dict[str, str] = {}

    extracted_path = os.path.join(output_folder, "rc_extracted.csv")
    filtered_path = os.path.join(output_folder, "criteria_after_basic_filtering.csv")
    consolidated_path = os.path.join(output_folder, "rc_consolidated.csv")
    template_path = rq3_template_path or os.path.join(
        os.path.dirname(RQ3_CRITERIA_DEFAULT_PATH),
        "rc_rq3_template.csv",
    )

    if "extract" in steps:
        print("\n=== Step 0: Extract criteria from experiment bundles ===")
        path = extract_criteria_from_folders(input_folders, output_folder, rq_id)
        if not path:
            raise RuntimeError("Extraction produced no criteria. Check input folders and bundle files.")
        artifacts["rc_extracted"] = path

    if "filter" in steps:
        print("\n=== Step 4: LLM criteria filtering (paper) ===")
        if not os.path.isfile(extracted_path):
            raise FileNotFoundError(f"Missing {extracted_path}. Run extract step first.")
        df = pd.read_csv(extracted_path)
        df, counts = apply_filtering(df)
        print_filtering_report(counts)
        df.to_csv(filtered_path, index=False)
        artifacts["criteria_after_basic_filtering"] = filtered_path
        print(f"Filtered criteria saved to {filtered_path}")

    if "consolidate" in steps:
        print("\n=== Step 5: LLM criteria consolidation (paper) ===")
        if not os.path.isfile(filtered_path):
            raise FileNotFoundError(f"Missing {filtered_path}. Run filter step first.")
        df = pd.read_csv(filtered_path)
        run_consolidation(
            df,
            output_folder,
            k_range=k_range,
            n_bootstrap=n_bootstrap,
        )
        artifacts["rc_consolidated"] = consolidated_path

    if "export-rq3-template" in steps:
        print("\n=== Export RQ3 criteria template (post manual review) ===")
        if not os.path.isfile(consolidated_path):
            raise FileNotFoundError(f"Missing {consolidated_path}. Run consolidate step first.")
        artifacts["rc_rq3_template"] = export_rq3_template(consolidated_path, template_path)

    print("\n=== Pipeline complete ===")
    for key, path in artifacts.items():
        print(f"  {key}: {path}")
    if "rc_consolidated" in artifacts:
        print(
            "\nManual step (paper): inspect rc_consolidated.csv, remove domain-specific "
            "or redundant entries, then save as:"
        )
        print(f"  {RQ3_CRITERIA_DEFAULT_PATH}")

    return artifacts
