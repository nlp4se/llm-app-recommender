#!/usr/bin/env python3
"""RQ2: mobile app ranking consistency (RBO) — internal, intra-family external, cross-family external."""

from __future__ import annotations

import argparse
import logging
import sys

from code.consistency.runner import RQ2AnalysisScope, run_consistency_analysis
from code.experiments.config import Family, OUTPUT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_families(value: str) -> list[Family]:
    if value.lower() == "both":
        return ["proprietary", "open"]
    return [f.strip().lower() for f in value.split(",") if f.strip()]  # type: ignore[return-value]


def parse_scopes(value: str) -> list[RQ2AnalysisScope]:
    mapping = {
        "internal": "internal",
        "external-intra": "external_intra",
        "external_intra": "external_intra",
        "external-cross": "external_cross",
        "external_cross": "external_cross",
        "all": "all",
    }
    if value.lower() == "all":
        return ["all"]  # type: ignore[list-item]
    scopes: list[RQ2AnalysisScope] = []
    for part in value.split(","):
        key = part.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unknown analysis scope '{part}'. Choose from: internal, external-intra, external-cross, all")
        resolved = mapping[key]
        if resolved != "all" and resolved not in scopes:
            scopes.append(resolved)  # type: ignore[arg-type]
    return scopes or ["all"]  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RQ2 mobile app consistency using Rank-Biased Overlap (RBO). "
            "Reads RQ1 bundles from rq1/apps/{suite}/ and writes rq2/{suite}/."
        ),
    )
    parser.add_argument(
        "--families",
        default="both",
        help="Model families: open, proprietary, or both (comma-separated).",
    )
    parser.add_argument(
        "--model-keys",
        default="all",
        help="Comma-separated model keys, or 'all'.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="*",
        help="Ranking depth(s) for RBO evaluation (default: 20).",
    )
    parser.add_argument(
        "--rbo-p",
        type=float,
        default=0.9,
        help="Primary RBO persistence parameter p (default: 0.9).",
    )
    parser.add_argument(
        "--rbo-p-values",
        default=None,
        help=(
            "Optional comma-separated p values for sensitivity analysis "
            "(e.g. 0.8,0.9,0.95). Produces an aggregation plot; primary outputs use --rbo-p."
        ),
    )
    parser.add_argument(
        "--analysis",
        default="all",
        help=(
            "Which analyses to run: internal, external-intra, external-cross, all "
            "(comma-separated). Criteria consistency is excluded."
        ),
    )
    parser.add_argument(
        "--dataset-suite",
        default=None,
        help="Override suite folder for all families (e.g. open_large). Prefer --dataset-scale large.",
    )
    parser.add_argument(
        "--dataset-scale",
        choices=["large", "small"],
        default="large",
        help="Use rq1/apps/open_{scale} and proprietary_{scale} (default: large).",
    )
    parser.add_argument(
        "--experiment-root",
        default=OUTPUT_ROOT,
        help="Root containing rq1/apps/ input bundles.",
    )
    parser.add_argument(
        "--output-root",
        default="data/output/features/rq2",
        help="Root for RQ2 CSV/plot outputs.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Reuse existing app_rankings.csv when present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned inputs/outputs without computing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    families = parse_families(args.families)
    scopes = parse_scopes(args.analysis)
    if args.model_keys.lower() == "all":
        model_keys = None
    else:
        model_keys = [k.strip().lower() for k in args.model_keys.split(",") if k.strip()]

    rbo_p_values = None
    if args.rbo_p_values:
        rbo_p_values = [float(v.strip()) for v in args.rbo_p_values.split(",") if v.strip()]

    dataset_suite = args.dataset_suite
    if dataset_suite is None:
        dataset_suite = None  # resolved per family from scale below

    run_consistency_analysis(
        families=families,
        model_keys=model_keys,
        k_values=args.k,
        experiment_root=args.experiment_root,
        output_root=args.output_root,
        rbo_p=args.rbo_p,
        rbo_p_values=rbo_p_values,
        scopes=scopes,
        skip_extract=args.skip_extract,
        dry_run=args.dry_run,
        dataset_suite=dataset_suite,
        dataset_scale=args.dataset_scale,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
