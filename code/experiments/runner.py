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
from code.experiments.io import (
    bundle_coverage,
    expected_record_keys,
    index_successful_records,
    load_bundle,
    parse_and_validate_response,
    save_bundle,
)
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


def _build_bundle_payload(
    *,
    rq: RQConfig,
    model_spec: ModelSpec,
    mode: OutputMode,
    k: int,
    search_items: list[str],
    n: int,
    records_by_key: dict,
    key_order: list,
    criteria_csv: str | None,
) -> dict[str, object]:
    records = [records_by_key[key] for key in key_order if key in records_by_key]
    payload: dict[str, object] = {
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
        payload["criteria_csv"] = criteria_csv
    return payload


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
    continue_on_error: bool = False,
) -> None:
    """
    Run one model/mode/k setting; write a bundled *_ALL.json file.

    Resumes automatically: loads an existing bundle, skips successful cells, and
    checkpoints to disk after every completed record.
    """
    out_dir = family_output_dir(output_root, rq.id, model_spec.family)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_filename = build_bundle_filename(
        family=model_spec.family,
        provider=model_spec.provider.value,
        model_key=model_spec.key,
        mode=mode,
        k=k,
    )
    bundle_path = out_dir / bundle_filename

    criteria_names: list[str] | None = None
    criteria_rows: list[dict] | None = None
    if rq.id == "rq3":
        if not criteria_csv:
            raise ValueError("RQ3 requires criteria_csv")
        criteria_df = pd.read_csv(criteria_csv, encoding="utf-8", sep=";")
        criteria_rows = [row.to_dict() for _, row in criteria_df.iterrows()]
        criteria_names = [str(list(row.values())[0]) for row in criteria_rows]

    key_order = expected_record_keys(
        rq_id=rq.id,
        features=search_items,
        runs_per_item=n,
        criteria_names=criteria_names,
    )

    coverage = bundle_coverage(
        bundle_path,
        rq_id=rq.id,
        features=search_items,
        runs_per_item=n,
        criteria_names=criteria_names,
    )
    if coverage["complete"]:
        logger.info(
            "Bundle already complete (%s/%s records) -> skipping %s",
            coverage["successful"],
            coverage["expected"],
            bundle_filename,
        )
        return

    records_by_key = index_successful_records([], rq.id)
    if coverage["exists"]:
        existing = load_bundle(bundle_path)
        assert existing is not None
        records_by_key = index_successful_records(existing.get("records", []), rq.id)
        logger.info(
            "Resuming %s: %s/%s successful, %s missing, %s failed placeholders",
            bundle_filename,
            coverage["successful"],
            coverage["expected"],
            coverage["missing"],
            coverage["failed"],
        )
    else:
        logger.info(
            "Starting %s | %s/%s | k=%s | %s cells",
            bundle_filename,
            model_spec.family,
            model_spec.key,
            k,
            len(key_order),
        )

    schema = load_schema_for_run(rq.schema_base, k)
    system = read_text(system_prompt_path(rq, mode))
    client = get_client(model_spec)
    structured = mode == "structured"

    def persist() -> None:
        save_bundle(
            bundle_path,
            _build_bundle_payload(
                rq=rq,
                model_spec=model_spec,
                mode=mode,
                k=k,
                search_items=search_items,
                n=n,
                records_by_key=records_by_key,
                key_order=key_order,
                criteria_csv=criteria_csv,
            ),
        )

    for cell_key in key_order:
        if cell_key in records_by_key:
            continue

        feature = cell_key[0]
        if rq.id == "rq1":
            run_idx = int(cell_key[1])
            user = format_user_prompt(rq.user_prompt, k=k, search=feature)
            label = f"{feature} run {run_idx + 1}/{n}"
            record_stub: dict[str, object] = {"feature": feature, "run": run_idx}
        else:
            criterion_name = str(cell_key[1])
            run_idx = int(cell_key[2])
            criterion_row = next(
                row for row in (criteria_rows or []) if str(list(row.values())[0]) == criterion_name
            )
            ranking_criteria = json.dumps([criterion_row], indent=2)
            user = format_user_prompt(
                rq.user_prompt,
                k=k,
                search=feature,
                ranking_criteria=ranking_criteria,
            )
            label = f"{feature} / {criterion_name} run {run_idx + 1}/{n}"
            record_stub = {
                "feature": feature,
                "criterion": criterion_name,
                "run": run_idx,
            }

        try:
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
            records_by_key[cell_key] = {**record_stub, "payload": payload}
        except Exception as exc:
            if not continue_on_error:
                persist()
                raise
            logger.error("%s: storing error placeholder (%s)", label, exc)
            records_by_key[cell_key] = {**record_stub, "payload": None, "error": str(exc)}
        persist()
        time.sleep(sleep)

    final = bundle_coverage(
        bundle_path,
        rq_id=rq.id,
        features=search_items,
        runs_per_item=n,
        criteria_names=criteria_names,
    )
    logger.info(
        "Finished %s -> %s/%s successful (%s missing)",
        bundle_filename,
        final["successful"],
        final["expected"],
        final["missing"],
    )


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
    continue_on_error: bool = False,
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

    criteria_names: list[str] | None = None
    if rq_id == "rq3" and criteria_csv:
        criteria_df = pd.read_csv(criteria_csv, encoding="utf-8", sep=";")
        criteria_names = [str(row.iloc[0]) for _, row in criteria_df.iterrows()]

    criteria_count = len(criteria_names) if criteria_names else 1
    bundle_combos = list(product(selected_specs, modes, k_values))
    records_per_bundle = len(search_items) * criteria_count * n

    logger.info(
        "Starting %s at %s - %s bundled files (%s records each, n=%s, resume=on)",
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
            bundle_path = out_family / spec.family / example
            cov = bundle_coverage(
                bundle_path,
                rq_id=rq_id,
                features=search_items,
                runs_per_item=n,
                criteria_names=criteria_names,
            )
            status = "complete" if cov["exists"] and cov["complete"] else f"{cov['successful']}/{cov['expected']} done"
            print(
                f"  {spec.key}/{mode} k={k}: {status} "
                f"-> {bundle_path}"
            )
        print(
            f"[dry-run] {rq_id}: {len(selected_specs)} models × {len(modes)} modes × "
            f"{len(k_values)} k = {len(bundle_combos)} bundled files (re-run fills gaps)"
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
            continue_on_error=continue_on_error,
        )

    logger.info("All experiments for %s completed.", rq_id)
