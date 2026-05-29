#!/usr/bin/env python3
"""Report missing or failed cells in bundled experiment outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from code.experiments.config import K_VALUES_CSV, MODEL_SPECS, OUTPUT_ROOT, RQ_CONFIGS
from code.experiments.io import bundle_coverage
from code.experiments.naming import build_bundle_filename
from code.experiments.runner import read_csv_column


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check completeness of bundled experiment files.")
    parser.add_argument("--rq", required=True, choices=sorted(RQ_CONFIGS.keys()))
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--families", default="both", help="proprietary, open, or both")
    parser.add_argument("--model-keys", default="all")
    parser.add_argument("--k", type=int, nargs="*")
    parser.add_argument("--criteria-csv", default=None)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args(argv)

    rq = RQ_CONFIGS[args.rq]
    families = ["proprietary", "open"] if args.families.lower() == "both" else [args.families.lower()]
    model_keys = None if args.model_keys.lower() == "all" else [k.strip() for k in args.model_keys.split(",")]
    k_values = args.k or [int(v) for v in read_csv_column(K_VALUES_CSV)]
    features = read_csv_column(rq.search_items_csv)
    n = args.n if args.n is not None else rq.runs_per_item

    criteria_csv = args.criteria_csv or (rq.default_criteria_csv if args.rq == "rq3" else None)
    criteria_names = None
    if args.rq == "rq3" and criteria_csv:
        criteria_df = pd.read_csv(criteria_csv, encoding="utf-8", sep=";")
        criteria_names = [str(row.iloc[0]) for _, row in criteria_df.iterrows()]

    specs = [
        s for s in MODEL_SPECS.values()
        if s.family in families and (model_keys is None or s.key in model_keys)
    ]

    incomplete = 0
    for spec in specs:
        for k in k_values:
            bundle_name = build_bundle_filename(
                family=spec.family,
                provider=spec.provider.value,
                model_key=spec.key,
                mode="structured",
                k=k,
            )
            bundle_path = Path(args.output_root) / args.rq / spec.family / bundle_name
            cov = bundle_coverage(
                bundle_path,
                rq_id=args.rq,
                features=features,
                runs_per_item=n,
                criteria_names=criteria_names,
            )
            status = "OK" if cov["complete"] else "INCOMPLETE"
            if not cov["complete"]:
                incomplete += 1
            print(
                f"[{status}] {spec.key} k={k}: "
                f"{cov['successful']}/{cov['expected']} ok, "
                f"{cov['failed']} failed, {cov['missing']} missing -> {bundle_path.name}"
            )
            if cov["missing_keys"] and len(cov["missing_keys"]) <= 5:
                for key in cov["missing_keys"]:
                    print(f"    missing: {key}")
            elif cov["missing"]:
                print(f"    ({cov['missing']} missing cells — re-run run_experiments.py to fill)")

    if incomplete:
        print(f"\n{incomplete} bundle(s) need attention. Re-run the same experiment command to resume.")
        return 1
    print("\nAll bundles complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
