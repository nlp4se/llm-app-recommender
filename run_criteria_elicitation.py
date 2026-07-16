#!/usr/bin/env python3
"""Unified entry point for RQ1 ranking-criteria elicitation (paper Steps 4–5)."""

from __future__ import annotations

import argparse
import sys

from code.hf_cache import configure_hf_cache

configure_hf_cache()

from code.elicitation.pipeline import (
    ELICITATION_DEFAULT_OUTPUT,
    RQ3_CRITERIA_DEFAULT_PATH,
    run_criteria_elicitation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RQ1 criteria elicitation pipeline: extract from bundles → filter (Step 4) "
            "→ consolidate (Step 5) → RQ3 template. Manual review produces rc_wo_id.csv."
        ),
    )
    parser.add_argument(
        "--input-folders",
        nargs="+",
        default=["data/output/features/rq1/apps/open_large"],
        help="RQ1 bundle folders (e.g. data/output/features/rq1/apps/open_large).",
    )
    parser.add_argument(
        "--output-folder",
        default=ELICITATION_DEFAULT_OUTPUT,
        help="RC artifact directory (e.g. data/output/features/rq1/rc/open_large).",
    )
    parser.add_argument(
        "--rq",
        default="rq1",
        choices=["rq1", "rq3"],
        help="Research question id for bundle parsing (use rq1 for criteria elicitation).",
    )
    parser.add_argument(
        "--steps",
        default="all",
        help=(
            "Comma-separated: extract, filter, consolidate, export-rq3-template, or all. "
            "Use subsets to resume (e.g. filter,consolidate)."
        ),
    )
    parser.add_argument(
        "--k-range",
        default="2,50",
        help="Cluster count search range for consolidation (min,max).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10,
        help="Bootstrap samples for gap statistic (Step 5).",
    )
    parser.add_argument(
        "--rq3-template",
        default=None,
        help=f"Path for semicolon CSV template (default: beside {RQ3_CRITERIA_DEFAULT_PATH}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    k_min, k_max = map(int, args.k_range.split(","))
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    run_criteria_elicitation(
        input_folders=args.input_folders,
        output_folder=args.output_folder,
        rq_id=args.rq,
        steps=steps,
        k_range=(k_min, k_max),
        n_bootstrap=args.n_bootstrap,
        rq3_template_path=args.rq3_template,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
