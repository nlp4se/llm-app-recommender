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
    OllamaSettings,
    OutputMode,
    RQConfig,
)
from code.experiments.io import parse_and_validate_response, save_json_response
from code.experiments.naming import build_bundle_filename, family_output_dir
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


def _model_label(spec: ModelSpec) -> str:
    return f"{spec.family}/{spec.key} ({spec.model_id})"


def check_model_availability(selected_specs: list[ModelSpec]) -> tuple[list[ModelSpec], dict[str, str]]:
    """Run a lightweight ping against each selected model."""
    available: list[ModelSpec] = []
    unavailable: dict[str, str] = {}
    for spec in selected_specs:
        label = _model_label(spec)
        try:
            client = get_client(spec)
            client.complete(
                "You are a helpful assistant.",
                "Return exactly: ok",
                structured=False,
            )
            logger.info("Sanity check OK: %s", label)
            available.append(spec)
        except Exception as exc:  # pragma: no cover - provider-specific runtime errors
            reason = str(exc) or repr(exc)
            unavailable[spec.key] = reason
            logger.warning("Sanity check FAILED: %s -> %s", label, reason)
    return available, unavailable


def _complete_validated(
    *,
    client,
    system: str,
    user: str,
    structured: bool,
    schema: dict,
    max_attempts: int,
    sleep: float,
    label: str,
) -> dict:
    for attempt in range(1, max_attempts + 1):
        start = time.time()
        try:
            content = client.complete(system, user, structured=structured, schema=schema)
            payload = parse_and_validate_response(content, schema)
            logger.info("%s attempt %s done (%.2fs)", label, attempt, time.time() - start)
            return payload
        except Exception as exc:
            if attempt >= max_attempts:
                logger.exception("%s failed after %s attempts", label, max_attempts)
                raise
            logger.warning(
                "%s invalid (attempt %s/%s): %s. Retrying...",
                label,
                attempt,
                max_attempts,
                exc,
            )
            time.sleep(sleep)
    raise RuntimeError(f"Unreachable: exhausted attempts for {label}")


def run_setting_bundle(
    *,
    rq: RQConfig,
    model_spec: ModelSpec,
    mode: OutputMode,
    k: int,
    search_items: list[str],
    n: int,
    sleep: float,
    output_root: str,
    max_attempts: int,
    criteria_csv: str | None = None,
) -> None:
    """Run one model/mode/k setting; write a single bundled *_ALL.json file."""
    out_dir = family_output_dir(output_root, rq.id, model_spec.family)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = load_schema_for_run(rq.schema_base, k)
    system = read_text(system_prompt_path(rq, mode))
    client = get_client(model_spec)
    structured = mode == "structured"

    records: list[dict[str, object]] = []
    criteria_rows: list[dict] | None = None

    if rq.id == "rq3":
        if not criteria_csv:
            raise ValueError("RQ3 requires criteria_csv")
        criteria_df = pd.read_csv(criteria_csv, encoding="utf-8", sep=";")
        criteria_rows = [row.to_dict() for _, row in criteria_df.iterrows()]
        logger.info(
            "RQ3 BUNDLE %s | %s/%s | %s | k=%s | features=%s | criteria=%s",
            rq.id,
            model_spec.family,
            model_spec.key,
            mode,
            k,
            len(search_items),
            len(criteria_rows),
        )
    else:
        logger.info(
            "RQ1 BUNDLE %s | %s/%s | %s | k=%s | features=%s",
            rq.id,
            model_spec.family,
            model_spec.key,
            mode,
            k,
            len(search_items),
        )

    for search in search_items:
        if rq.id == "rq1":
            user = format_user_prompt(rq.user_prompt, k=k, search=search)
            for run_idx in range(n):
                label = f"{search} run {run_idx + 1}/{n}"
                payload = _complete_validated(
                    client=client,
                    system=system,
                    user=user,
                    structured=structured,
                    schema=schema,
                    max_attempts=max_attempts,
                    sleep=sleep,
                    label=label,
                )
                records.append({"feature": search, "run": run_idx, "payload": payload})
                time.sleep(sleep)
            continue

        assert criteria_rows is not None
        for criterion_row in criteria_rows:
            criterion_name = str(list(criterion_row.values())[0])
            ranking_criteria = json.dumps([criterion_row], indent=2)
            user = format_user_prompt(
                rq.user_prompt,
                k=k,
                search=search,
                ranking_criteria=ranking_criteria,
            )
            for run_idx in range(n):
                label = f"{search} / {criterion_name} run {run_idx + 1}/{n}"
                payload = _complete_validated(
                    client=client,
                    system=system,
                    user=user,
                    structured=structured,
                    schema=schema,
                    max_attempts=max_attempts,
                    sleep=sleep,
                    label=label,
                )
                records.append(
                    {
                        "feature": search,
                        "criterion": criterion_name,
                        "run": run_idx,
                        "payload": payload,
                    }
                )
                time.sleep(sleep)

    bundle_filename = build_bundle_filename(
        family=model_spec.family,
        provider=model_spec.provider.value,
        model_key=model_spec.key,
        mode=mode,
        k=k,
    )
    bundle_payload: dict[str, object] = {
        "rq": rq.id,
        "family": model_spec.family,
        "provider": model_spec.provider.value,
        "model_key": model_spec.key,
        "model_id": model_spec.model_id,
        "mode": mode,
        "k": k,
        "features": search_items,
        "runs_per_item": n,
        "records": records,
    }
    if rq.id == "rq3" and criteria_csv:
        bundle_payload["criteria_csv"] = criteria_csv

    save_json_response(
        json.dumps(bundle_payload, indent=2, ensure_ascii=False),
        out_dir / bundle_filename,
    )
    logger.info("Saved bundled file -> %s (%s records)", bundle_filename, len(records))


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
    sanity_check: bool = False,
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
            model_id=models[spec.key] if spec.key in models else spec.model_id,
            hf=spec.hf,
            ollama=(
                OllamaSettings(
                    base_url=spec.ollama.base_url if spec.ollama else "http://localhost:11434",
                    use_json_schema=spec.ollama.use_json_schema if spec.ollama else True,
                    max_tokens=spec.ollama.max_tokens if spec.ollama else 8192,
                )
                if spec.key in models
                else spec.ollama
            ),
        )
        for spec in selected_specs
    ]

    if sanity_check:
        selected_specs, unavailable = check_model_availability(selected_specs)
        if unavailable:
            unavailable_models = ",".join(sorted(unavailable))
            raise ValueError(f"Sanity check failed for models: {unavailable_models}")
        if not selected_specs:
            raise ValueError("No available models after sanity check.")
        logger.info("Sanity check complete: %s models available", len(selected_specs))

    k_values = k_values or [int(v) for v in read_csv_column(K_VALUES_CSV)]
    search_items = search_items or read_csv_column(rq.search_items_csv)
    n = n if n is not None else rq.runs_per_item

    if rq_id == "rq3":
        if not criteria_csv:
            criteria_csv = rq.default_criteria_csv
        if not criteria_csv or not Path(criteria_csv).exists():
            raise FileNotFoundError(
                f"RQ3 requires ranking criteria CSV at {criteria_csv}. "
                "Generate it from RQ1 analysis or pass --criteria-csv."
            )

    criteria_count = 1
    if rq_id == "rq3" and criteria_csv:
        criteria_count = len(pd.read_csv(criteria_csv, encoding="utf-8", sep=";"))

    bundle_combos = list(product(selected_specs, modes, k_values))
    records_per_bundle = len(search_items) * criteria_count * n

    logger.info(
        "Starting %s at %s - %s bundled files (%s records each, n=%s)",
        rq_id,
        datetime.now().isoformat(timespec="seconds"),
        len(bundle_combos),
        records_per_bundle,
        n,
    )

    if dry_run:
        out_family = Path(output_root) / rq.id
        for spec, mode, k in bundle_combos:
            example = build_bundle_filename(
                family=spec.family,
                provider=spec.provider.value,
                model_key=spec.key,
                mode=mode,
                k=k,
            )
            print(
                f"  {spec.key}/{mode} k={k}: 1 file, {records_per_bundle} records "
                f"-> {out_family / spec.family / example}"
            )
        print(
            f"[dry-run] {rq_id}: {len(selected_specs)} models × {len(modes)} modes × "
            f"{len(k_values)} k = {len(bundle_combos)} bundled files"
        )
        return

    for spec, mode, k in bundle_combos:
        run_setting_bundle(
            rq=rq,
            model_spec=spec,
            mode=mode,
            k=k,
            search_items=search_items,
            n=n,
            sleep=sleep,
            output_root=output_root,
            max_attempts=max_attempts,
            criteria_csv=criteria_csv if rq_id == "rq3" else None,
        )

    logger.info("All experiments for %s completed.", rq_id)
