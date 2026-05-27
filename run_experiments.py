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


def parse_modes(value: str) -> list[str]:
    if value.lower() == "both":
        return ["structured", "prompt"]
    return [m.strip().lower() for m in value.split(",") if m.strip()]


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
        description="Run LLM app-ranking experiments across model families and output modes.",
    )
    parser.add_argument(
        "--rq",
        required=True,
        choices=sorted(RQ_CONFIGS.keys()),
        help="Research question config to run (rq1 or rq3).",
    )
    parser.add_argument(
        "--families",
        default="both",
        help="Model families: proprietary, open, or both.",
    )
    parser.add_argument(
        "--modes",
        default="both",
        help="Output modes: structured, prompt, or both.",
    )
    parser.add_argument(
        "--model-keys",
        default="all",
        help="Comma-separated model keys to run, or 'all'.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Optional overrides: model_key=model_id (e.g. gemma4=google/gemma-4-31B).",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="*",
        help="Override k values (default: from data/input/use-case/k.csv).",
    )
    parser.add_argument(
        "--search",
        nargs="*",
        help="Override feature names (default: from features.csv).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Runs per item (default: 10 for rq1, 4 for rq3).",
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
        help="RQ3 only: CSV with ranking criteria (default: rc_wo_id.csv from RQ1 pipeline).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without calling APIs.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max retries per run when JSON/schema validation fails.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    families = parse_families(args.families)
    modes = parse_modes(args.modes)
    models = parse_models(args.models)
    if args.model_keys.lower() == "all":
        model_keys: list[str] | None = None
    else:
        model_keys = [k.strip().lower() for k in args.model_keys.split(",") if k.strip()]

    run_experiments(
        rq_id=args.rq,
        families=families,
        model_keys=model_keys,
        modes=modes,
        models=models,
        k_values=args.k,
        search_items=args.search,
        n=args.n,
        sleep=args.sleep,
        output_root=args.output_root,
        criteria_csv=args.criteria_csv,
        max_attempts=args.max_attempts,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
