"""App-ranking RBO consistency with per-feature and aggregated scores."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from code.consistency.metrics import mean_pairwise_metric, rbo

DEFAULT_RBO_P = 0.9
RQ2_K_VALUES = [20]


def _rank_lists(df: pd.DataFrame, model: str, feature: str, k: int) -> list[list[str]]:
    subset = df[(df["model"] == model) & (df["feature"] == feature)]
    cols = [str(i) for i in range(1, k + 1) if str(i) in subset.columns]
    if not cols:
        return []
    lists = subset[cols].values.tolist()
    return [[app for app in row if pd.notna(app) and str(app).strip()] for row in lists]


def external_app_rbo_by_feature(
    df: pd.DataFrame,
    models: list[str],
    features: list[str],
    k: int,
    p: float = DEFAULT_RBO_P,
) -> pd.DataFrame:
    """Per (model pair, feature) mean RBO across run pairs at fixed p."""
    rows: list[dict] = []

    for model1, model2 in combinations(models, 2):
        for feature in features:
            lists1 = _rank_lists(df, model1, feature, k)
            lists2 = _rank_lists(df, model2, feature, k)
            if not lists1 or not lists2:
                continue
            rbo_pairs = ((a, b) for a in lists1 for b in lists2)
            score = mean_pairwise_metric(rbo_pairs, lambda a, b, pv=p: rbo(a, b, p=pv))
            rows.append(
                {
                    "model1": model1,
                    "model2": model2,
                    "feature": feature,
                    "rbo": score,
                }
            )

    return pd.DataFrame(rows)


def internal_app_rbo_by_feature(
    df: pd.DataFrame,
    models: list[str],
    features: list[str],
    k: int,
    p: float = DEFAULT_RBO_P,
) -> pd.DataFrame:
    """Per (model, feature) mean RBO across within-model run pairs at fixed p."""
    rows: list[dict] = []

    for model in models:
        for feature in features:
            lists = _rank_lists(df, model, feature, k)
            if len(lists) < 2:
                continue
            rbo_pairs = ((lists[i], lists[j]) for i, j in combinations(range(len(lists)), 2))
            score = mean_pairwise_metric(rbo_pairs, lambda a, b, pv=p: rbo(a, b, p=pv))
            rows.append({"model": model, "feature": feature, "rbo": score})

    return pd.DataFrame(rows)


def _summarize_external(by_feature: pd.DataFrame) -> pd.DataFrame:
    if by_feature.empty:
        return pd.DataFrame(columns=["model1", "model2", "rbo_mean", "rbo_std", "n_features"])

    grouped = by_feature.groupby(["model1", "model2"], as_index=False)["rbo"]
    return grouped.agg(rbo_mean="mean", rbo_std="std", n_features="count")


def _summarize_internal(by_feature: pd.DataFrame) -> pd.DataFrame:
    if by_feature.empty:
        return pd.DataFrame(columns=["model", "rbo_mean", "rbo_std", "n_features"])

    grouped = by_feature.groupby("model", as_index=False)["rbo"]
    return grouped.agg(rbo_mean="mean", rbo_std="std", n_features="count")


def external_app_rbo(
    df: pd.DataFrame,
    models: list[str],
    features: list[str],
    k: int,
    p: float = DEFAULT_RBO_P,
) -> pd.DataFrame:
    """Mean RBO between model pairs (avg over features) at fixed p."""
    by_feature = external_app_rbo_by_feature(df, models, features, k, p=p)
    summary = _summarize_external(by_feature)
    if summary.empty:
        return pd.DataFrame(columns=["model1", "model2", "rbo"])
    out = summary.rename(columns={"rbo_mean": "rbo"})
    return out[["model1", "model2", "rbo"]]


def internal_app_rbo(
    df: pd.DataFrame,
    models: list[str],
    features: list[str],
    k: int,
    p: float = DEFAULT_RBO_P,
) -> pd.DataFrame:
    """Mean within-model RBO across run pairs (avg over features) at fixed p."""
    by_feature = internal_app_rbo_by_feature(df, models, features, k, p=p)
    summary = _summarize_internal(by_feature)
    if summary.empty:
        return pd.DataFrame(columns=["model", "rbo"])
    out = summary.rename(columns={"rbo_mean": "rbo"})
    return out[["model", "rbo"]]


def external_app_rbo_summary(
    df: pd.DataFrame,
    models: list[str],
    features: list[str],
    k: int,
    p: float = DEFAULT_RBO_P,
) -> pd.DataFrame:
    """Model-pair RBO mean and std across features."""
    return _summarize_external(external_app_rbo_by_feature(df, models, features, k, p=p))


def cross_family_external_app_rbo_by_feature(
    open_df: pd.DataFrame,
    proprietary_df: pd.DataFrame,
    open_models: list[str],
    proprietary_models: list[str],
    features: list[str],
    k: int,
    p: float = DEFAULT_RBO_P,
) -> pd.DataFrame:
    """Per (open model, proprietary model, feature) mean RBO across run pairs."""
    rows: list[dict] = []

    for open_model in open_models:
        for prop_model in proprietary_models:
            for feature in features:
                lists_open = _rank_lists(open_df, open_model, feature, k)
                lists_prop = _rank_lists(proprietary_df, prop_model, feature, k)
                if not lists_open or not lists_prop:
                    continue
                rbo_pairs = ((a, b) for a in lists_open for b in lists_prop)
                score = mean_pairwise_metric(rbo_pairs, lambda a, b, pv=p: rbo(a, b, p=pv))
                rows.append(
                    {
                        "open_model": open_model,
                        "proprietary_model": prop_model,
                        "feature": feature,
                        "rbo": score,
                    }
                )

    return pd.DataFrame(rows)


def cross_family_external_app_rbo_summary(
    open_df: pd.DataFrame,
    proprietary_df: pd.DataFrame,
    open_models: list[str],
    proprietary_models: list[str],
    features: list[str],
    k: int,
    p: float = DEFAULT_RBO_P,
) -> pd.DataFrame:
    """Open × proprietary model-pair RBO mean and std across features."""
    by_feature = cross_family_external_app_rbo_by_feature(
        open_df,
        proprietary_df,
        open_models,
        proprietary_models,
        features,
        k,
        p=p,
    )
    if by_feature.empty:
        return pd.DataFrame(
            columns=["open_model", "proprietary_model", "rbo_mean", "rbo_std", "n_features"]
        )
    grouped = by_feature.groupby(["open_model", "proprietary_model"], as_index=False)["rbo"]
    return grouped.agg(rbo_mean="mean", rbo_std="std", n_features="count")


def internal_app_rbo_summary(
    df: pd.DataFrame,
    models: list[str],
    features: list[str],
    k: int,
    p: float = DEFAULT_RBO_P,
) -> pd.DataFrame:
    """Per-model RBO mean and std across features."""
    return _summarize_internal(internal_app_rbo_by_feature(df, models, features, k, p=p))
