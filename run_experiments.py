#!/usr/bin/env python3
"""Unified entry point: RQ1/RQ3 experiments, RQ2 consistency analysis."""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from code.consistency.runner import run_consistency_analysis
from code.experiments.config import Family, OUTPUT_ROOT, RQ_CONFIGS
from code.experiments.runner import run_experiments

ALL_RQ_CHOICES = sorted(list(RQ_CONFIGS.keys()) + ["rq2"])

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
            "RQ1/RQ3: LLM ranking experiments (bundled JSON). "
            "RQ2: mobile app RBO consistency (internal, intra-family external, cross-family external)."
        ),
    )
    parser.add_argument(
        "--rq",
        required=True,
        choices=ALL_RQ_CHOICES,
        help="rq1/rq3 experiments; rq2 consistency analysis on rq1 bundles.",
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
        help="Override features (default: features_large.csv for all families).",
    )
    parser.add_argument(
        "--features-csv",
        default=None,
        help="Feature list CSV for all families (overrides defaults).",
    )
    parser.add_argument(
        "--features-csv-proprietary",
        default=None,
        help="Optional smaller feature list for proprietary only (e.g. features_proprietary.csv).",
    )
    parser.add_argument(
        "--dataset-suite",
        default=None,
        help=(
            "Output suite folder name under rq*/apps/ and rq*/rc/, e.g. open_small. "
            "Default: inferred from --features-csv (features_small.csv → *_small)."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Runs per (feature) or per (feature, criterion) for RQ3 (default: 10 rq1, 5 rq3).",
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
        help=(
            "RQ3 only: ranking criteria CSV (default: rq1/rc/merge/rc_merge_unified.csv, "
            "else rq1/rc/<suite>/rc_wo_id_<suite>.csv)."
        ),
    )
    parser.add_argument(
        "--web-search",
        action="store_true",
        help=(
            "Enable provider web-search tools where supported (Gemini Google Search, "
            "Mistral web_search). Off by default to reduce cost."
        ),
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
        help="Preflight each model with one structured mock cell (same path as real runs).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "On persistent failure for one cell, save an error placeholder and continue "
            "(re-run later to retry failed cells)."
        ),
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="RQ2 only: reuse existing app_rankings.csv.",
    )
    parser.add_argument(
        "--rbo-p",
        type=float,
        default=0.9,
        help="RQ2 only: primary RBO persistence parameter (default: 0.9).",
    )
    parser.add_argument(
        "--rbo-p-values",
        default=None,
        help="RQ2 only: comma-separated p values for sensitivity plots (e.g. 0.8,0.9,0.95).",
    )
    parser.add_argument(
        "--analysis",
        default="all",
        help="RQ2 only: internal, external-intra, external-cross, or all (comma-separated).",
    )
    parser.add_argument(
        "--dataset-scale",
        choices=["large", "small"],
        default="large",
        help="RQ2 only: use rq1/apps/*_{scale} suites (default: large).",
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

    if args.rq == "rq2":
        from code.consistency.runner import run_consistency_analysis

        rbo_p_values = None
        if args.rbo_p_values:
            rbo_p_values = [float(v.strip()) for v in args.rbo_p_values.split(",") if v.strip()]

        run_consistency_analysis(
            families=families,
            model_keys=model_keys,
            k_values=args.k,
            experiment_root=args.output_root,
            output_root="data/output/features/rq2",
            rbo_p=args.rbo_p,
            rbo_p_values=rbo_p_values,
            scopes=[s.strip() for s in args.analysis.split(",")],  # type: ignore[arg-type]
            skip_extract=args.skip_extract,
            dry_run=args.dry_run,
            dataset_suite=args.dataset_suite,
            dataset_scale=args.dataset_scale,
        )
        return 0

    run_experiments(
        rq_id=args.rq,
        families=families,
        model_keys=model_keys,
        modes=["structured"],
        models=models,
        k_values=args.k,
        search_items=args.search,
        features_csv=args.features_csv,
        features_csv_proprietary=args.features_csv_proprietary,
        dataset_suite=args.dataset_suite,
        n=args.n,
        sleep=args.sleep,
        output_root=args.output_root,
        criteria_csv=args.criteria_csv,
        max_attempts=args.max_attempts,
        dry_run=args.dry_run,
        sanity_check=args.sanity_check,
        continue_on_error=args.continue_on_error,
        web_search=args.web_search,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
