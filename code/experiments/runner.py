from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

from code.experiments.config import (
    K_VALUES_CSV,
    MAX_ATTEMPTS_PER_RUN,
    MODEL_SPECS,
    OUTPUT_ROOT,
    RQ_CONFIGS,
    Family,
    ModelSpec,
    OutputMode,
    RQConfig,
)
from code.experiments.io import parse_and_validate_response, save_json_response
from code.experiments.naming import (
    build_rq1_filename,
    build_rq3_filename,
    family_output_dir,
    to_camel_case,
)
from code.experiments.prompts import format_user_prompt, read_text, system_prompt_path
from code.experiments.providers import get_client  # lazy-loads provider SDKs
from code.experiments.schema import load_schema_for_run

logger = logging.getLogger(__name__)


def read_csv_column(path: str | Path, column: int = 0, skip_header: bool = True) -> list[str]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader, None)
        return [row[column] for row in reader if row]


def run_rq1_cell(
    *,
    rq: RQConfig,
    model_spec: ModelSpec,
    mode: OutputMode,
    k: int,
    search: str,
    n: int,
    sleep: float,
    output_root: str,
    max_attempts: int,
) -> None:
    out_dir = family_output_dir(output_root, rq.id, model_spec.family)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = load_schema_for_run(rq.schema_base, k)
    system = read_text(system_prompt_path(rq, mode))
    user = format_user_prompt(rq.user_prompt, k=k, search=search)
    client = get_client(model_spec.provider, model_spec.model_id)

    logger.info(
        "RQ1 %s | %s/%s | %s | k=%s | search=%s",
        rq.id,
        model_spec.family,
        model_spec.key,
        mode,
        k,
        search,
    )

    for run_idx in range(n):
        filename = build_rq1_filename(
            family=model_spec.family,
            provider=model_spec.provider.value,
            model_key=model_spec.key,
            mode=mode,
            k=k,
            feature=search,
            run=run_idx,
        )
        for attempt in range(1, max_attempts + 1):
            start = time.time()
            try:
                content = client.complete(system, user, structured=(mode == "structured"), schema=schema)
                payload = parse_and_validate_response(content, schema)
                save_json_response(json.dumps(payload, indent=2, ensure_ascii=False), out_dir / filename)
                logger.info(
                    "Run %s/%s attempt %s done (%.2fs) -> %s",
                    run_idx + 1,
                    n,
                    attempt,
                    time.time() - start,
                    filename,
                )
                break
            except Exception as exc:
                if attempt >= max_attempts:
                    logger.exception("Failed run %s for %s after %s attempts", run_idx, filename, max_attempts)
                    raise
                logger.warning(
                    "Invalid response for %s (attempt %s/%s): %s. Retrying...",
                    filename,
                    attempt,
                    max_attempts,
                    exc,
                )
                time.sleep(sleep)
        time.sleep(sleep)


def run_rq3_cell(
    *,
    rq: RQConfig,
    model_spec: ModelSpec,
    mode: OutputMode,
    k: int,
    search: str,
    n: int,
    sleep: float,
    output_root: str,
    criteria_csv: str,
    max_attempts: int,
) -> None:
    out_dir = family_output_dir(output_root, rq.id, model_spec.family)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = load_schema_for_run(rq.schema_base, k)
    system = read_text(system_prompt_path(rq, mode))
    client = get_client(model_spec.provider, model_spec.model_id)

    criteria_df = pd.read_csv(criteria_csv, encoding="utf-8", sep=";")

    logger.info(
        "RQ3 %s | %s/%s | %s | k=%s | search=%s | criteria rows=%s",
        rq.id,
        model_spec.family,
        model_spec.key,
        mode,
        k,
        search,
        len(criteria_df),
    )

    for _, row in criteria_df.iterrows():
        ranking_criteria = json.dumps([row.to_dict()], indent=2)
        criterion_name = str(row.iloc[0])
        user = format_user_prompt(
            rq.user_prompt,
            k=k,
            search=search,
            ranking_criteria=ranking_criteria,
        )

        for run_idx in range(n):
            filename = build_rq3_filename(
                family=model_spec.family,
                provider=model_spec.provider.value,
                model_key=model_spec.key,
                mode=mode,
                k=k,
                feature=search,
                criterion=criterion_name,
                run=run_idx,
            )
            for attempt in range(1, max_attempts + 1):
                start = time.time()
                try:
                    content = client.complete(
                        system,
                        user,
                        structured=(mode == "structured"),
                        schema=schema,
                    )
                    payload = parse_and_validate_response(content, schema)
                    save_json_response(json.dumps(payload, indent=2, ensure_ascii=False), out_dir / filename)
                    logger.info(
                        "Criteria %s run %s/%s attempt %s done (%.2fs) -> %s",
                        to_camel_case(criterion_name),
                        run_idx + 1,
                        n,
                        attempt,
                        time.time() - start,
                        filename,
                    )
                    break
                except Exception as exc:
                    if attempt >= max_attempts:
                        logger.exception("Failed %s run %s after %s attempts", criterion_name, run_idx, max_attempts)
                        raise
                    logger.warning(
                        "Invalid response for %s (attempt %s/%s): %s. Retrying...",
                        filename,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    time.sleep(sleep)
            time.sleep(sleep)


def run_experiments(
    *,
    rq_id: str,
    families: list[Family],
    model_keys: list[str] | None = None,
    modes: list[OutputMode],
    models: dict[str, str] | None = None,
    k_values: list[int] | None = None,
    search_items: list[str] | None = None,
    n: int | None = None,
    sleep: float = 10.0,
    output_root: str = OUTPUT_ROOT,
    criteria_csv: str | None = None,
    max_attempts: int = MAX_ATTEMPTS_PER_RUN,
    dry_run: bool = False,
) -> None:
    rq = RQ_CONFIGS[rq_id]
    models = models or {}
    selected_specs = [
        spec for spec in MODEL_SPECS.values()
        if spec.family in families and (model_keys is None or spec.key in model_keys)
    ]
    if not selected_specs:
        raise ValueError("No models selected after applying families/model_keys filters.")
    selected_specs = [
        ModelSpec(
            key=spec.key,
            family=spec.family,
            provider=spec.provider,
            model_id=models.get(spec.key, spec.model_id),
        )
        for spec in selected_specs
    ]

    k_values = k_values or [int(v) for v in read_csv_column(K_VALUES_CSV)]
    search_items = search_items or read_csv_column(rq.search_items_csv)
    n = n if n is not None else rq.runs_per_item

    if rq_id == "rq3" and not criteria_csv:
        criteria_csv = rq.default_criteria_csv
    if rq_id == "rq3" and criteria_csv and not Path(criteria_csv).exists():
        raise FileNotFoundError(
            f"RQ3 requires ranking criteria CSV at {criteria_csv}. "
            "Generate it from RQ1 analysis or pass --criteria-csv."
        )

    combos = list(product(selected_specs, modes, k_values, search_items))
    logger.info(
        "Starting %s at %s - %s combinations (n=%s per item)",
        rq_id,
        datetime.now().isoformat(timespec="seconds"),
        len(combos),
        n,
    )

    if dry_run:
        out_dir = Path(output_root) / rq.id
        for spec, mode, k, search in combos:
            if rq_id == "rq1":
                example = build_rq1_filename(
                    family=spec.family,
                    provider=spec.provider.value,
                    model_key=spec.key,
                    mode=mode,
                    k=k,
                    feature=search,
                    run=0,
                )
            else:
                example = build_rq3_filename(
                    family=spec.family,
                    provider=spec.provider.value,
                    model_key=spec.key,
                    mode=mode,
                    k=k,
                    feature=search,
                    criterion="ExampleCriterion",
                    run=0,
                )
            print(f"[dry-run] {spec.family}/{spec.key}/{mode} model={spec.model_id} -> {out_dir / spec.family / example}")
        return

    for spec, mode, k, search in combos:
        common = dict(
            rq=rq,
            model_spec=spec,
            mode=mode,
            k=k,
            search=search,
            n=n,
            sleep=sleep,
            output_root=output_root,
            max_attempts=max_attempts,
        )
        if rq_id == "rq1":
            run_rq1_cell(**common)
        else:
            run_rq3_cell(**common, criteria_csv=criteria_csv)  # type: ignore[arg-type]

    logger.info("All experiments for %s completed.", rq_id)
