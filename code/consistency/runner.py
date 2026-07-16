"""RQ2: mobile app ranking RBO consistency (internal, intra-family external, cross-family external)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from code.consistency.plots import (
    plot_all_models_external_rbo,
    plot_cross_family_external_rbo,
    plot_external_rbo_multi_k,
    plot_internal_rbo_multi_k,
    plot_rbo_p_sensitivity,
)
from code.consistency.apps import (
    DEFAULT_RBO_P,
    RQ2_K_VALUES,
    cross_family_external_app_rbo_by_feature,
    cross_family_external_app_rbo_summary,
    external_app_rbo_by_feature,
    external_app_rbo_summary,
    internal_app_rbo_by_feature,
    internal_app_rbo_summary,
)
from code.elicitation.merge_recommendations import generate_app_rankings_csv
from code.experiments.config import DEFAULT_MODEL_KEYS_BY_FAMILY, MODEL_SPECS, Family
from code.experiments.naming import apps_output_dir

logger = logging.getLogger(__name__)

RQ2_OUTPUT_ROOT = "data/output/features/rq2"
SOURCE_RQ = "rq1"

RQ2AnalysisScope = Literal["internal", "external_intra", "external_cross", "all"]


def _normalize_scopes(scopes: list[RQ2AnalysisScope] | None) -> set[str]:
    if not scopes or scopes == ["all"]:
        return {"internal", "external_intra", "external_cross"}
    return set(scopes)


def _source_k_for_eval(available: list[int], eval_k: int) -> int | None:
    """Pick ranking depth in CSV to use when evaluating RBO at eval_k."""
    if eval_k in available:
        return eval_k
    candidates = [s for s in available if s >= eval_k]
    return max(candidates) if candidates else None


def _resolve_models(families: list[Family], model_keys: list[str] | None) -> list[str]:
    keys: list[str] = []
    for family in families:
        if model_keys is None:
            keys.extend(DEFAULT_MODEL_KEYS_BY_FAMILY.get(family, []))
        else:
            keys.extend(k for k in model_keys if MODEL_SPECS.get(k) and MODEL_SPECS[k].family == family)
    seen: set[str] = set()
    ordered: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered


def _suites_for_scale(
    *,
    dataset_suite_override: str | None,
    dataset_scale: str,
) -> tuple[str, str]:
    if dataset_suite_override:
        if dataset_suite_override.startswith("open"):
            return dataset_suite_override, f"proprietary_{dataset_scale}"
        if dataset_suite_override.startswith("proprietary"):
            return f"open_{dataset_scale}", dataset_suite_override
        raise ValueError(
            f"--dataset-suite must start with open or proprietary, got '{dataset_suite_override}'"
        )
    return f"open_{dataset_scale}", f"proprietary_{dataset_scale}"


def _extract_rankings(
    *,
    experiment_root: str,
    suite: str,
    output_dir: Path,
    skip_extract: bool,
) -> pd.DataFrame:
    apps_csv = output_dir / "app_rankings.csv"
    if not skip_extract or not apps_csv.is_file():
        fam_folders = [str(apps_output_dir(experiment_root, SOURCE_RQ, suite))]
        from code.experiments.io import expand_bundled_folders

        rows, _ = expand_bundled_folders(fam_folders, SOURCE_RQ)
        if not rows:
            raise RuntimeError(f"No bundled RQ1 files in {fam_folders}")
        generate_app_rankings_csv(rows, str(output_dir))
    return pd.read_csv(apps_csv)


def _rank_df_for_eval(apps_df: pd.DataFrame, eval_k: int) -> tuple[pd.DataFrame, int]:
    available_k = sorted(int(v) for v in apps_df["k"].unique().tolist())
    source_k = _source_k_for_eval(available_k, eval_k)
    if source_k is None:
        raise RuntimeError(f"Cannot evaluate k={eval_k}; available ranking depths: {available_k}")
    rank_df = apps_df[apps_df["k"] == source_k]
    return rank_df, source_k


def _run_family_analyses(
    *,
    family: Family,
    suite: str,
    experiment_root: str,
    output_root: Path,
    models: list[str],
    k_values: list[int],
    rbo_p: float,
    rbo_p_values: list[float] | None,
    scopes: set[str],
    skip_extract: bool,
) -> None:
    family_out = output_root / suite
    plots_dir = family_out / "heatmaps"
    plots_dir.mkdir(parents=True, exist_ok=True)

    apps_df = _extract_rankings(
        experiment_root=experiment_root,
        suite=suite,
        output_dir=family_out,
        skip_extract=skip_extract,
    )
    family_models = [m for m in models if m in apps_df["model"].unique()]
    features = apps_df["feature"].unique().tolist()

    external_by_k: dict[int, pd.DataFrame] = {}
    internal_by_k: dict[int, pd.DataFrame] = {}
    internal_detail_by_k: dict[int, pd.DataFrame] = {}
    active_k: list[int] = []

    for eval_k in k_values:
        rank_df, source_k = _rank_df_for_eval(apps_df, eval_k)
        if source_k != eval_k:
            logger.info(
                "RQ2 %s: evaluating k=%s using top-%s from k=%s rankings",
                suite,
                eval_k,
                eval_k,
                source_k,
            )

        if "internal" in scopes:
            int_detail = internal_app_rbo_by_feature(rank_df, family_models, features, eval_k, p=rbo_p)
            int_summary = internal_app_rbo_summary(rank_df, family_models, features, eval_k, p=rbo_p)
            internal_detail_by_k[eval_k] = int_detail
            internal_by_k[eval_k] = int_summary
            int_detail.to_csv(family_out / f"internal_rbo_by_feature_k{eval_k}.csv", index=False)
            int_summary.to_csv(family_out / f"internal_rbo_k{eval_k}.csv", index=False)

        if "external_intra" in scopes:
            ext_detail = external_app_rbo_by_feature(rank_df, family_models, features, eval_k, p=rbo_p)
            ext_summary = external_app_rbo_summary(rank_df, family_models, features, eval_k, p=rbo_p)
            external_by_k[eval_k] = ext_summary
            ext_detail.to_csv(family_out / f"external_rbo_by_feature_k{eval_k}.csv", index=False)
            ext_summary.to_csv(family_out / f"external_rbo_k{eval_k}.csv", index=False)

        active_k.append(eval_k)
        logger.info(
            "RQ2 %s eval_k=%s: %s models, %s features, p=%s",
            suite,
            eval_k,
            len(family_models),
            len(features),
            rbo_p,
        )

    if not active_k:
        raise RuntimeError(f"RQ2 {suite}: no eval k values produced outputs.")

    suffix = f"k{active_k[0]}_p{rbo_p}" if len(active_k) == 1 else f"multi_k_p{rbo_p}"
    family_label = "Open-source" if family == "open" else "Proprietary"

    if "external_intra" in scopes and external_by_k:
        plot_external_rbo_multi_k(
            external_by_k,
            k_values=[k for k in k_values if k in active_k],
            title=f"{family_label} — external consistency (within family)",
            output_path=plots_dir / f"apps_external_rbo_{suffix}.png",
            models=family_models,
            p=rbo_p,
        )

    if "internal" in scopes and internal_detail_by_k:
        plot_internal_rbo_multi_k(
            internal_detail_by_k,
            internal_by_k,
            k_values=[k for k in k_values if k in active_k],
            title=f"{family_label} — internal consistency",
            output_path=plots_dir / f"apps_internal_rbo_{suffix}.png",
            models=family_models,
        )

    if rbo_p_values and "external_intra" in scopes:
        summaries_by_p: dict[float, pd.DataFrame] = {}
        rank_df, _ = _rank_df_for_eval(apps_df, active_k[0])
        for p_val in rbo_p_values:
            summaries_by_p[p_val] = external_app_rbo_summary(
                rank_df, family_models, features, active_k[0], p=p_val
            )
        plot_rbo_p_sensitivity(
            summaries_by_p,
            title=f"{family_label} — external RBO vs p (k={active_k[0]})",
            output_path=plots_dir / f"apps_external_rbo_p_sensitivity_k{active_k[0]}.png",
            analysis_label="Mean pairwise external RBO",
        )


def _run_cross_family_analysis(
    *,
    experiment_root: str,
    output_root: Path,
    open_suite: str,
    proprietary_suite: str,
    open_models: list[str],
    proprietary_models: list[str],
    k_values: list[int],
    rbo_p: float,
    rbo_p_values: list[float] | None,
    skip_extract: bool,
) -> None:
    cross_out = output_root / f"cross_{open_suite.removeprefix('open_')}"
    plots_dir = cross_out / "heatmaps"
    plots_dir.mkdir(parents=True, exist_ok=True)

    open_df = _extract_rankings(
        experiment_root=experiment_root,
        suite=open_suite,
        output_dir=output_root / open_suite,
        skip_extract=skip_extract,
    )
    prop_df = _extract_rankings(
        experiment_root=experiment_root,
        suite=proprietary_suite,
        output_dir=output_root / proprietary_suite,
        skip_extract=skip_extract,
    )

    open_models = [m for m in open_models if m in open_df["model"].unique()]
    proprietary_models = [m for m in proprietary_models if m in prop_df["model"].unique()]
    features = sorted(set(open_df["feature"]).intersection(set(prop_df["feature"])))
    if not features:
        raise RuntimeError("No shared features between open and proprietary rankings.")

    for eval_k in k_values:
        open_rank_df, open_source_k = _rank_df_for_eval(open_df, eval_k)
        prop_rank_df, prop_source_k = _rank_df_for_eval(prop_df, eval_k)
        logger.info(
            "RQ2 cross %s vs %s: k=%s (open source k=%s, proprietary source k=%s), %s shared features",
            open_suite,
            proprietary_suite,
            eval_k,
            open_source_k,
            prop_source_k,
            len(features),
        )

        detail = cross_family_external_app_rbo_by_feature(
            open_rank_df,
            prop_rank_df,
            open_models,
            proprietary_models,
            features,
            eval_k,
            p=rbo_p,
        )
        summary = cross_family_external_app_rbo_summary(
            open_rank_df,
            prop_rank_df,
            open_models,
            proprietary_models,
            features,
            eval_k,
            p=rbo_p,
        )
        detail.to_csv(cross_out / f"external_cross_rbo_by_feature_k{eval_k}.csv", index=False)
        summary.to_csv(cross_out / f"external_cross_rbo_k{eval_k}.csv", index=False)

        plot_cross_family_external_rbo(
            summary,
            open_models=open_models,
            proprietary_models=proprietary_models,
            title="Open vs proprietary — external consistency",
            output_path=plots_dir / f"apps_external_cross_rbo_k{eval_k}_p{rbo_p}.png",
            k=eval_k,
            p=rbo_p,
        )

        combined_df = pd.concat([open_rank_df, prop_rank_df], ignore_index=True)
        all_models = open_models + [m for m in proprietary_models if m not in open_models]
        combined_detail = external_app_rbo_by_feature(
            combined_df, all_models, features, eval_k, p=rbo_p
        )
        combined_summary = external_app_rbo_summary(
            combined_df, all_models, features, eval_k, p=rbo_p
        )
        combined_detail.to_csv(cross_out / f"external_all_models_rbo_by_feature_k{eval_k}.csv", index=False)
        combined_summary.to_csv(cross_out / f"external_all_models_rbo_k{eval_k}.csv", index=False)
        plot_all_models_external_rbo(
            combined_summary,
            open_models=open_models,
            proprietary_models=proprietary_models,
            title="All models — external consistency",
            output_path=plots_dir / f"apps_external_all_models_rbo_k{eval_k}_p{rbo_p}.png",
            k=eval_k,
            p=rbo_p,
        )

        if rbo_p_values:
            summaries_by_p: dict[float, pd.DataFrame] = {}
            for p_val in rbo_p_values:
                summaries_by_p[p_val] = cross_family_external_app_rbo_summary(
                    open_rank_df,
                    prop_rank_df,
                    open_models,
                    proprietary_models,
                    features,
                    eval_k,
                    p=p_val,
                )
            plot_rbo_p_sensitivity(
                summaries_by_p,
                title=f"Cross-family external RBO vs p (k={eval_k})",
                output_path=plots_dir / f"apps_external_cross_rbo_p_sensitivity_k{eval_k}.png",
                analysis_label="Mean open × proprietary RBO",
            )


def run_consistency_analysis(
    *,
    families: list[Family],
    model_keys: list[str] | None = None,
    k_values: list[int] | None = None,
    experiment_root: str = "data/output/features",
    output_root: str = RQ2_OUTPUT_ROOT,
    rbo_p: float = DEFAULT_RBO_P,
    rbo_p_values: list[float] | None = None,
    scopes: list[RQ2AnalysisScope] | None = None,
    skip_extract: bool = False,
    dry_run: bool = False,
    dataset_suite: str | None = None,
    dataset_scale: str = "large",
    features_csv: str | None = None,
    features_csv_proprietary: str | None = None,
) -> None:
    """RQ2 app-ranking RBO consistency for mobile apps only."""
    active_scopes = _normalize_scopes(scopes)
    models = _resolve_models(families, model_keys)
    if not models:
        raise ValueError("No models selected.")

    k_values = k_values or list(RQ2_K_VALUES)
    k_values = sorted(dict.fromkeys(k_values))
    out_base = Path(output_root)

    open_suite, proprietary_suite = _suites_for_scale(
        dataset_suite_override=dataset_suite,
        dataset_scale=dataset_scale,
    )

    if dry_run:
        print(f"[dry-run] RQ2 app RBO | scopes={sorted(active_scopes)} | families={families}")
        print(f"  Input:  {experiment_root}/rq1/apps/{{open,proprietary}}_{dataset_scale}/")
        print(f"  Output: {out_base}/")
        print(f"  k={k_values} | primary p={rbo_p}")
        if rbo_p_values:
            print(f"  p sensitivity: {rbo_p_values}")
        if "internal" in active_scopes or "external_intra" in active_scopes:
            for family in families:
                suite = open_suite if family == "open" else proprietary_suite
                print(f"    - {suite}/heatmaps/  (internal + intra-family external)")
        if "external_cross" in active_scopes:
            print(f"    - cross_{dataset_scale}/heatmaps/  (open × proprietary + all-models)")
        return

    for family in families:
        if family not in ("open", "proprietary"):
            continue
        if "internal" not in active_scopes and "external_intra" not in active_scopes:
            continue
        suite = open_suite if family == "open" else proprietary_suite
        _run_family_analyses(
            family=family,
            suite=suite,
            experiment_root=experiment_root,
            output_root=out_base,
            models=models,
            k_values=k_values,
            rbo_p=rbo_p,
            rbo_p_values=rbo_p_values,
            scopes=active_scopes,
            skip_extract=skip_extract,
        )

    if "external_cross" in active_scopes:
        if "open" not in families or "proprietary" not in families:
            logger.warning("Cross-family external analysis requires both open and proprietary families.")
        else:
            open_models = [m for m in models if MODEL_SPECS.get(m) and MODEL_SPECS[m].family == "open"]
            prop_models = [m for m in models if MODEL_SPECS.get(m) and MODEL_SPECS[m].family == "proprietary"]
            _run_cross_family_analysis(
                experiment_root=experiment_root,
                output_root=out_base,
                open_suite=open_suite,
                proprietary_suite=proprietary_suite,
                open_models=open_models,
                proprietary_models=prop_models,
                k_values=k_values,
                rbo_p=rbo_p,
                rbo_p_values=rbo_p_values,
                skip_extract=skip_extract,
            )

    logger.info("RQ2 consistency analysis completed.")
