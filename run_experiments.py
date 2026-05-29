#!/usr/bin/env python3
"""Unified entry point for feature-based ranking experiments (RQ1, RQ3)."""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from code.experiments.config import Family, OUTPUT_ROOT, RQ_CONFIGS
from code.experiments.runner import run_experiments

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_families(value: str) -> list[Family]:
    if value.lower() == "both":
        return ["proprietary", "open"]
    return [f.strip().lower() for f in value.split(",") if f.strip()]  # type: ignore[return-value]


def parse_models(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    result: dict[str, str] = {}
    for pair in value.split(","):
        if "=" not in pair:
            raise ValueError(f"Invalid model override '{pair}', expected model_key=model_id")
        model_key, model_name = pair.split("=", 1)
        result[model_key.strip().lower()] = model_name.strip()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LLM app-ranking experiments. Outputs one bundled JSON per "
            "(model, structured mode, k) under data/output/features/{rq}/{family}/."
        ),
    )
    parser.add_argument(
        "--rq",
        required=True,
        choices=sorted(RQ_CONFIGS.keys()),
        help="Research question: rq1 (rank + criteria) or rq3 (rank with given criteria).",
    )
    parser.add_argument(
        "--families",
        default="both",
        help="Model families: proprietary, open, or both (comma-separated).",
    )
    parser.add_argument(
        "--model-keys",
        default="all",
        help="Comma-separated model keys to run, or 'all'.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Optional overrides: model_key=model_id (e.g. llama31_8b=llama3.1:8b).",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="*",
        help="k values (default: from data/input/use-case/k.csv).",
    )
    parser.add_argument(
        "--search",
        nargs="*",
        help="Feature names to run (default: all rows in features.csv).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Runs per (feature) or per (feature, criterion) for RQ3 (default: 10 rq1, 4 rq3).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=10.0,
        help="Seconds to wait between API calls.",
    )
    parser.add_argument(
        "--output-root",
        default=OUTPUT_ROOT,
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--criteria-csv",
        default=None,
        help="RQ3 only: ranking criteria CSV (default: rc_wo_id.csv from RQ1 pipeline).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned bundled files without calling APIs.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max retries per record when JSON/schema validation fails.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Preflight selected models with a lightweight API call before the full run.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "On persistent failure for one cell, save an error placeholder and continue "
            "(re-run later to retry failed cells)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    families = parse_families(args.families)
    models = parse_models(args.models)
    if args.model_keys.lower() == "all":
        model_keys: list[str] | None = None
    else:
        model_keys = [k.strip().lower() for k in args.model_keys.split(",") if k.strip()]

    run_experiments(
        rq_id=args.rq,
        families=families,
        model_keys=model_keys,
        modes=["structured"],
        models=models,
        k_values=args.k,
        search_items=args.search,
        n=args.n,
        sleep=args.sleep,
        output_root=args.output_root,
        criteria_csv=args.criteria_csv,
        max_attempts=args.max_attempts,
        dry_run=args.dry_run,
        sanity_check=args.sanity_check,
        continue_on_error=args.continue_on_error,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
