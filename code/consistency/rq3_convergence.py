"""RQ3 publication analysis: impact of ranking criteria on recommendation convergence.

Small-scale dataset (16 features), knowledge-only, 10 models, 16 ranking criteria,
5 guided runs per feature x criterion x model, k=20, RBO p=0.9.

Blind baselines are recomputed from the RQ1 bundles (open_small +
proprietary_small_wo_websearch, 10 runs each) using the exact same RBO
definition as the RQ2 pipeline, and cross-checked against the stored RQ2 CSVs.

Outputs:
  data/output/features/rq3/analysis/      CSV tables (A, B, C)
  data/output/features/rq3/publication/   figures (PDF + PNG, RQ2 styling)
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from code.consistency.publication_figures import (
    COL_WIDTH_IN,
    COHORT_BY_KEY,
    COLOR_OPEN,
    COLOR_OPEN_ERR,
    COLOR_PROPRIETARY,
    COLOR_PROPRIETARY_ERR,
    DISPLAY_NAMES,
    MODEL_KEYS_OPEN,
    MODEL_KEYS_PROPRIETARY,
    _save,
    _style_pub,
)

K = 20
P = 0.9
ROOT = Path("data/output/features")
BLIND_DIRS = [
    ROOT / "rq1/apps/open_small",
    ROOT / "rq1/apps/proprietary_small_wo_websearch",
]
GUIDED_DIRS = [
    ROOT / "rq3/apps/open_small",
    ROOT / "rq3/apps/proprietary_small",
]
ANALYSIS_DIR = ROOT / "rq3/analysis"
FIG_DIR = ROOT / "rq3/publication"

ALL_MODEL_KEYS = MODEL_KEYS_OPEN + MODEL_KEYS_PROPRIETARY
COLOR_CROSS = "#009E73"  # Okabe-Ito green for cross-cohort pairs
COLOR_CRITERION_BAR = "#5A9AB5"
COLOR_CRITERION_ERR = "#1E4D63"
DOUBLE_WIDTH_IN = 17.8 / 2.54

CRITERION_SHORT = {
    "Accessibility Features": "Accessibility",
    "Cost and Pricing": "Cost & Pricing",
    "Cross-Platform Synchronization": "Cross-Platform Sync",
    "Customer Support": "Customer Support",
    "Customization Options": "Customization",
    "Feature Breadth and Richness": "Feature Breadth",
    "Integration with Other Services": "Integrations",
    "Performance and Stability": "Performance",
    "Platform and Device Compatibility": "Compatibility",
    "Popularity and Market Reach": "Popularity",
    "Regional and Geographical Availability": "Regional Availability",
    "Regular Updates and Maintenance": "Updates & Maintenance",
    "Security and Privacy": "Security & Privacy",
    "User Engagement": "User Engagement",
    "User Interface and Experience": "UI & Experience",
    "User Ratings": "User Ratings",
}


# ---------------------------------------------------------------------------
# RBO (identical definition to code/consistency/metrics.py, with prefix-set
# precomputation for speed; verified to reproduce the stored RQ2 CSVs).
# ---------------------------------------------------------------------------

_WEIGHTS = [P ** (d - 1) for d in range(1, K + 1)]
_NORM_CUM = list(np.cumsum(_WEIGHTS))


class RList:
    """A ranked list with precomputed prefix sets."""

    __slots__ = ("items", "prefixes")

    def __init__(self, items: list[str]):
        self.items = items
        self.prefixes: list[set[str]] = []
        seen: set[str] = set()
        for x in items:
            seen.add(x)
            self.prefixes.append(set(seen))


def rbo_pre(a: RList, b: RList) -> float:
    if not a.items or not b.items:
        return 0.0
    if a.items == b.items:
        return 1.0
    la, lb = len(a.items), len(b.items)
    max_depth = max(la, lb)
    agreement = 0.0
    for d in range(1, max_depth + 1):
        s1 = a.prefixes[min(d, la) - 1]
        s2 = b.prefixes[min(d, lb) - 1]
        agreement += (len(s1 & s2) / d) * _WEIGHTS[d - 1]
    return agreement / _NORM_CUM[max_depth - 1]


def mean_pairwise(lists: list[RList]) -> float | None:
    if len(lists) < 2:
        return None
    scores = [rbo_pre(a, b) for a, b in combinations(lists, 2)]
    return float(np.mean(scores))


def mean_cross(lists1: list[RList], lists2: list[RList]) -> float | None:
    if not lists1 or not lists2:
        return None
    scores = [rbo_pre(a, b) for a in lists1 for b in lists2]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _clean(apps: list) -> list[str]:
    return [str(x).strip() for x in apps if x is not None and str(x).strip()]


def load_blind() -> dict[str, dict[str, list[RList]]]:
    """blind[model][feature] -> ranked lists (up to 10 runs)."""
    out: dict[str, dict[str, list[RList]]] = {}
    for folder in BLIND_DIRS:
        for path in sorted(folder.glob("*.json")):
            data = json.loads(path.read_text())
            model = data["model_key"]
            per_feat = out.setdefault(model, {})
            for rec in data["records"]:
                payload = rec.get("payload") or {}
                apps = _clean(payload.get("a") or [])
                if not apps:
                    continue
                per_feat.setdefault(rec["feature"], []).append(RList(apps[:K]))
    return out


def load_guided() -> dict[str, dict[str, dict[str, list[RList]]]]:
    """guided[model][criterion][feature] -> ranked lists (up to 5 runs)."""
    out: dict[str, dict[str, dict[str, list[RList]]]] = {}
    for folder in GUIDED_DIRS:
        for path in sorted(folder.glob("*.json")):
            data = json.loads(path.read_text())
            model = data["model_key"]
            per_crit = out.setdefault(model, {})
            for rec in data["records"]:
                payload = rec.get("payload") or {}
                apps = _clean(payload.get("a") or [])
                if not apps:
                    continue
                crit = rec["criterion"]["n"]
                per_crit.setdefault(crit, {}).setdefault(rec["feature"], []).append(
                    RList(apps[:K])
                )
    return out


# ---------------------------------------------------------------------------
# Part A: internal convergence (determinism gain)
# ---------------------------------------------------------------------------

def blind_internal_by_feature(blind) -> pd.DataFrame:
    rows = []
    for model in ALL_MODEL_KEYS:
        for feature, lists in blind[model].items():
            score = mean_pairwise(lists)
            if score is not None:
                rows.append({"model": model, "feature": feature, "rbo": score})
    return pd.DataFrame(rows)


def guided_internal_by_feature(guided, criteria) -> pd.DataFrame:
    rows = []
    for model in ALL_MODEL_KEYS:
        for crit in criteria:
            for feature, lists in guided[model].get(crit, {}).items():
                score = mean_pairwise(lists)
                if score is not None:
                    rows.append(
                        {"model": model, "criterion": crit, "feature": feature, "rbo": score}
                    )
    return pd.DataFrame(rows)


def internal_delta_table(blind_int: pd.DataFrame, guided_int: pd.DataFrame) -> pd.DataFrame:
    """Per model x criterion: guided mean, blind mean, matched-feature paired delta."""
    blind_map = blind_int.set_index(["model", "feature"])["rbo"]
    blind_mean = blind_int.groupby("model")["rbo"].mean()
    rows = []
    for (model, crit), grp in guided_int.groupby(["model", "criterion"]):
        matched = grp[grp.apply(lambda r: (r["model"], r["feature"]) in blind_map.index, axis=1)]
        deltas = [
            r["rbo"] - blind_map[(r["model"], r["feature"])] for _, r in matched.iterrows()
        ]
        rows.append(
            {
                "model": model,
                "criterion": crit,
                "guided_internal_mean": grp["rbo"].mean(),
                "guided_internal_std": grp["rbo"].std(),
                "n_features_guided": len(grp),
                "blind_internal_mean": blind_mean[model],
                "delta_matched": float(np.mean(deltas)) if deltas else np.nan,
                "n_features_matched": len(deltas),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part B: external convergence (cross-model agreement)
# ---------------------------------------------------------------------------

def _pair_cohort(m1: str, m2: str) -> str:
    c1, c2 = COHORT_BY_KEY[m1], COHORT_BY_KEY[m2]
    if c1 == c2:
        return c1
    return "cross"


def blind_external_by_pair_feature(blind, features) -> pd.DataFrame:
    rows = []
    for m1, m2 in combinations(ALL_MODEL_KEYS, 2):
        for feature in features:
            score = mean_cross(blind[m1].get(feature, []), blind[m2].get(feature, []))
            if score is not None:
                rows.append(
                    {
                        "model1": m1,
                        "model2": m2,
                        "cohort_pair": _pair_cohort(m1, m2),
                        "feature": feature,
                        "rbo": score,
                    }
                )
    return pd.DataFrame(rows)


def guided_external_by_pair_feature(guided, criteria, features) -> pd.DataFrame:
    rows = []
    for crit in criteria:
        for m1, m2 in combinations(ALL_MODEL_KEYS, 2):
            g1 = guided[m1].get(crit, {})
            g2 = guided[m2].get(crit, {})
            for feature in features:
                score = mean_cross(g1.get(feature, []), g2.get(feature, []))
                if score is not None:
                    rows.append(
                        {
                            "criterion": crit,
                            "model1": m1,
                            "model2": m2,
                            "cohort_pair": _pair_cohort(m1, m2),
                            "feature": feature,
                            "rbo": score,
                        }
                    )
    return pd.DataFrame(rows)


def external_per_criterion(
    guided_ext: pd.DataFrame, blind_ext_feat_mean: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary per criterion + paired stats over features (guided vs blind)."""
    # per criterion x feature: mean over the 45 model pairs
    per_cf = guided_ext.groupby(["criterion", "feature"])["rbo"].mean().reset_index()
    summary_rows = []
    stats_rows = []
    blind_overall = blind_ext_feat_mean.mean()
    for crit, grp in per_cf.groupby("criterion"):
        feats = grp.set_index("feature")["rbo"]
        common = feats.index.intersection(blind_ext_feat_mean.index)
        g = feats.loc[common].values
        b = blind_ext_feat_mean.loc[common].values
        diff = g - b
        t_stat, p_t = stats.ttest_rel(g, b)
        try:
            w_stat, p_w = stats.wilcoxon(g, b)
        except ValueError:
            w_stat, p_w = np.nan, np.nan
        dz = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else np.nan
        summary_rows.append(
            {
                "criterion": crit,
                "guided_external_mean": feats.mean(),
                "guided_external_std_across_features": feats.std(),
                "n_features": len(feats),
                "blind_external_mean": blind_overall,
                "delta": feats.mean() - blind_overall,
            }
        )
        stats_rows.append(
            {
                "criterion": crit,
                "mean_guided": float(np.mean(g)),
                "mean_blind": float(np.mean(b)),
                "mean_paired_delta": float(np.mean(diff)),
                "t_stat": float(t_stat),
                "p_ttest": float(p_t),
                "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else np.nan,
                "p_wilcoxon": float(p_w) if not np.isnan(p_w) else np.nan,
                "cohen_dz": dz,
                "n_features": len(common),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("guided_external_mean", ascending=False)
    stats_df = pd.DataFrame(stats_rows)
    # Holm correction across the 16 criteria (on Wilcoxon p-values)
    for col, out in [("p_ttest", "p_ttest_holm"), ("p_wilcoxon", "p_wilcoxon_holm")]:
        pvals = stats_df[col].values
        order = np.argsort(pvals)
        m = len(pvals)
        adj = np.empty(m)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * pvals[idx])
            adj[idx] = min(1.0, running)
        stats_df[out] = adj
    return summary, stats_df


def external_cohort_breakdown(
    guided_ext: pd.DataFrame, blind_ext: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Per criterion x cohort-pair guided external RBO; blind baseline per cohort-pair."""
    # feature-mean within cohort, then mean over features (equal feature weighting)
    g = (
        guided_ext.groupby(["criterion", "cohort_pair", "feature"])["rbo"]
        .mean()
        .groupby(["criterion", "cohort_pair"])
        .mean()
        .reset_index()
        .rename(columns={"rbo": "guided_external_mean"})
    )
    b = (
        blind_ext.groupby(["cohort_pair", "feature"])["rbo"]
        .mean()
        .groupby("cohort_pair")
        .mean()
    )
    g["blind_external_mean"] = g["cohort_pair"].map(b)
    g["delta"] = g["guided_external_mean"] - g["blind_external_mean"]
    return g, b


# ---------------------------------------------------------------------------
# Part C: guided-vs-blind displacement
# ---------------------------------------------------------------------------

def displacement_by_model_criterion_feature(blind, guided, criteria) -> pd.DataFrame:
    rows = []
    for model in ALL_MODEL_KEYS:
        for crit in criteria:
            g_per_feat = guided[model].get(crit, {})
            for feature, g_lists in g_per_feat.items():
                b_lists = blind[model].get(feature, [])
                score = mean_cross(g_lists, b_lists)
                if score is not None:
                    rows.append(
                        {
                            "model": model,
                            "cohort": COHORT_BY_KEY[model],
                            "criterion": crit,
                            "feature": feature,
                            "rbo": score,
                        }
                    )
    return pd.DataFrame(rows)


def mine_new_app_examples(blind, guided, criteria, top_n: int = 5) -> pd.DataFrame:
    """Apps appearing in guided top-N (majority of runs) but absent from all blind top-20."""
    rows = []
    for model in ALL_MODEL_KEYS:
        for crit in criteria:
            for feature, g_lists in guided[model].get(crit, {}).items():
                b_lists = blind[model].get(feature, [])
                if not b_lists:
                    continue
                blind_apps = set().union(*(set(bl.items) for bl in b_lists))
                counts: dict[str, list[int]] = {}
                for gl in g_lists:
                    for pos, app in enumerate(gl.items[:top_n], 1):
                        counts.setdefault(app, []).append(pos)
                for app, positions in counts.items():
                    if len(positions) >= 3 and app not in blind_apps:
                        rows.append(
                            {
                                "model": model,
                                "criterion": crit,
                                "feature": feature,
                                "app": app,
                                "n_runs_in_top5": len(positions),
                                "median_guided_rank": float(np.median(positions)),
                            }
                        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = (
        df.groupby(["criterion", "feature", "app"])
        .agg(
            n_models=("model", "nunique"),
            models=("model", lambda s: ", ".join(sorted(set(s)))),
            best_median_rank=("median_guided_rank", "min"),
        )
        .reset_index()
        .sort_values(["n_models", "best_median_rank"], ascending=[False, True])
    )
    return agg


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_internal_delta_heatmap(delta_df: pd.DataFrame, criteria: list[str]) -> None:
    _style_pub(base_size=8.0)
    model_order = MODEL_KEYS_OPEN + MODEL_KEYS_PROPRIETARY
    n_open = len(MODEL_KEYS_OPEN)
    matrix = delta_df.pivot(index="model", columns="criterion", values="delta_matched")
    matrix = matrix.reindex(index=model_order, columns=criteria)
    # append per-model mean column
    matrix["Mean"] = matrix.mean(axis=1)
    col_labels = [CRITERION_SHORT[c] for c in criteria] + ["Mean"]
    row_labels = [DISPLAY_NAMES[m] for m in model_order]

    vmax = float(np.nanmax(np.abs(matrix.values)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(DOUBLE_WIDTH_IN, 3.4))
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        norm=norm,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 5.8},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": f"Δ internal RBO (guided − blind, k={K}, p={P})",
            "shrink": 0.85,
            "pad": 0.015,
        },
        ax=ax,
    )
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7.0)
    ax.set_yticklabels(row_labels, rotation=0, fontsize=7.0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0, pad=3)
    ax.axhline(n_open, color="#374151", linewidth=2.0, zorder=4)
    ax.axvline(len(criteria), color="#374151", linewidth=1.4, zorder=4)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_fontsize(7.5)
    cbar.ax.tick_params(labelsize=7.0)
    plt.tight_layout(pad=0.5)
    _save(fig, FIG_DIR / "rq3_internal_delta_heatmap")


def fig_external_convergence_per_criterion(
    summary: pd.DataFrame, per_cf: pd.DataFrame, blind_baseline: float
) -> None:
    _style_pub(base_size=7.5)
    data = summary.sort_values("guided_external_mean", ascending=True).reset_index(drop=True)
    # 95% CI across the 16 features
    ci = []
    for crit in data["criterion"]:
        vals = per_cf.loc[per_cf["criterion"] == crit, "rbo"].values
        ci.append(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))
    data["ci95"] = ci

    fig_h = max(2.4, len(data) * 0.17 + 0.55)
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, fig_h))
    y_pos = np.arange(len(data))
    ax.barh(
        y_pos,
        data["guided_external_mean"],
        height=0.55,
        color=COLOR_CRITERION_BAR,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    for y, val, err in zip(y_pos, data["guided_external_mean"], data["ci95"]):
        ax.errorbar(
            val, y, xerr=err, fmt="none", ecolor=COLOR_CRITERION_ERR,
            elinewidth=0.9, capsize=1.6, capthick=0.8, zorder=4,
        )
    ax.axvline(blind_baseline, color="#374151", linestyle="--", linewidth=1.1, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([CRITERION_SHORT[c] for c in data["criterion"]])
    ax.set_xlim(0, max(0.65, float((data["guided_external_mean"] + data["ci95"]).max()) * 1.10))
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.set_xlabel(f"Mean external RBO (k={K}, p={P})", fontsize=7.0)
    ax.grid(axis="x", color="#E8ECF0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=COLOR_CRITERION_BAR, edgecolor="white", label="Guided (per criterion)"),
        Line2D([0], [0], color="#374151", linestyle="--", linewidth=1.1,
               label=f"Blind baseline = {blind_baseline:.3f}"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=6.5,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    plt.tight_layout(pad=0.4, rect=(0, 0, 1, 0.95))
    _save(fig, FIG_DIR / "rq3_external_convergence_per_criterion")


def fig_guided_vs_blind_boxplot(disp: pd.DataFrame, criteria_order: list[str]) -> None:
    _style_pub(base_size=7.5)
    df = disp.copy()
    df["criterion_short"] = df["criterion"].map(CRITERION_SHORT)
    df["Cohort"] = df["cohort"].map({"proprietary": "Proprietary", "open": "Open-source"})
    order = [CRITERION_SHORT[c] for c in criteria_order]

    fig, ax = plt.subplots(figsize=(DOUBLE_WIDTH_IN, 2.9))
    sns.boxplot(
        data=df,
        x="criterion_short",
        y="rbo",
        hue="Cohort",
        hue_order=["Proprietary", "Open-source"],
        order=order,
        palette={"Proprietary": COLOR_PROPRIETARY, "Open-source": COLOR_OPEN},
        linewidth=0.6,
        fliersize=1.0,
        width=0.72,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#1F2937",
            "markersize": 2.6,
            "markeredgewidth": 0.6,
        },
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(f"RBO guided vs. blind (k={K}, p={P})", fontsize=7.0)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=45, labelsize=6.8)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.grid(axis="y", color="#E8ECF0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=6.5,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    plt.tight_layout(pad=0.4, rect=(0, 0, 1, 0.96))
    _save(fig, FIG_DIR / "rq3_guided_vs_blind_boxplot")


def fig_external_cohort_breakdown(
    cohort_df: pd.DataFrame, blind_cohort: pd.Series, criteria_order: list[str]
) -> None:
    _style_pub(base_size=7.5)
    groups = ["proprietary", "open", "cross"]
    colors = {"proprietary": COLOR_PROPRIETARY, "open": COLOR_OPEN, "cross": COLOR_CROSS}
    labels = {"proprietary": "Proprietary pairs", "open": "Open-source pairs", "cross": "Cross-cohort pairs"}

    pivot = cohort_df.pivot(index="criterion", columns="cohort_pair", values="guided_external_mean")
    pivot = pivot.reindex(index=criteria_order, columns=groups)

    x = np.arange(len(criteria_order))
    width = 0.26
    fig, ax = plt.subplots(figsize=(DOUBLE_WIDTH_IN, 2.7))
    err_colors = {
        "proprietary": COLOR_PROPRIETARY_ERR,
        "open": COLOR_OPEN_ERR,
        "cross": "#00563F",
    }
    for i, g in enumerate(groups):
        ax.bar(
            x + (i - 1) * width,
            pivot[g],
            width=width,
            color=colors[g],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
            label=labels[g],
        )
        ax.axhline(
            blind_cohort[g],
            color=err_colors[g],
            linestyle="--",
            linewidth=1.3,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CRITERION_SHORT[c] for c in criteria_order], rotation=45, ha="right", fontsize=6.8)
    ax.set_ylabel(f"Mean external RBO (k={K}, p={P})", fontsize=7.0)
    ax.set_ylim(0, max(0.8, float(pivot.max().max()) * 1.12))
    ax.grid(axis="y", color="#E8ECF0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [Patch(facecolor=colors[g], edgecolor="white", label=labels[g]) for g in groups]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        fontsize=6.5,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    plt.tight_layout(pad=0.4, rect=(0, 0, 1, 0.96))
    _save(fig, FIG_DIR / "rq3_external_cohort_breakdown")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading blind (RQ1, knowledge-only) and guided (RQ3) bundles ...")
    blind = load_blind()
    guided = load_guided()
    criteria = sorted({c for m in guided.values() for c in m})
    features = sorted({f for m in blind.values() for f in m})
    print(f"  models={len(blind)}, criteria={len(criteria)}, features={len(features)}")

    # ---------------- Part A ----------------
    print("Part A: internal convergence ...")
    blind_int = blind_internal_by_feature(blind)
    blind_int_summary = (
        blind_int.groupby("model")["rbo"]
        .agg(rbo_mean="mean", rbo_std="std", n_features="count")
        .reset_index()
    )
    blind_int_summary.to_csv(ANALYSIS_DIR / "rq3_blind_internal_baseline.csv", index=False)

    guided_int = guided_internal_by_feature(guided, criteria)
    guided_int.to_csv(ANALYSIS_DIR / "rq3_internal_guided_by_feature.csv", index=False)

    delta = internal_delta_table(blind_int, guided_int)
    delta.to_csv(ANALYSIS_DIR / "rq3_internal_delta.csv", index=False)

    delta_matrix = delta.pivot(index="model", columns="criterion", values="delta_matched")
    delta_matrix = delta_matrix.reindex(index=MODEL_KEYS_OPEN + MODEL_KEYS_PROPRIETARY)
    delta_matrix["MODEL_MEAN"] = delta_matrix.mean(axis=1)
    delta_matrix.loc["CRITERION_MEAN"] = delta_matrix.mean(axis=0)
    delta_matrix.to_csv(ANALYSIS_DIR / "rq3_internal_delta_matrix.csv")

    # ---------------- Part B ----------------
    print("Part B: external convergence (blind baseline) ...")
    blind_ext = blind_external_by_pair_feature(blind, features)
    blind_ext.to_csv(ANALYSIS_DIR / "rq3_blind_external_by_pair_feature.csv", index=False)
    blind_ext_feat_mean = blind_ext.groupby("feature")["rbo"].mean()
    blind_ext_feat_mean.rename("mean_external_rbo").reset_index().to_csv(
        ANALYSIS_DIR / "rq3_blind_external_by_feature.csv", index=False
    )
    blind_overall = float(blind_ext_feat_mean.mean())
    print(f"  blind external baseline (mean of feature means) = {blind_overall:.4f}")

    # Cross-check against stored baseline CSV
    stored = ROOT / "rq2/rq3_baseline_small_ko_external_by_feature_k20_p0.9.csv"
    if stored.is_file():
        st = pd.read_csv(stored).set_index("feature")["mean_external_rbo"]
        diff = (blind_ext_feat_mean - st).abs().max()
        print(f"  max |diff| vs stored rq2 baseline file: {diff:.6f}")

    print("Part B: external convergence (guided, 16 criteria x 45 pairs) ...")
    guided_ext = guided_external_by_pair_feature(guided, criteria, features)
    guided_ext.to_csv(ANALYSIS_DIR / "rq3_external_guided_by_pair_feature.csv", index=False)

    per_cf = guided_ext.groupby(["criterion", "feature"])["rbo"].mean().reset_index()
    per_cf.to_csv(ANALYSIS_DIR / "rq3_external_guided_by_criterion_feature.csv", index=False)

    ext_summary, ext_stats = external_per_criterion(guided_ext, blind_ext_feat_mean)
    ext_summary.to_csv(ANALYSIS_DIR / "rq3_external_guided_per_criterion.csv", index=False)
    ext_stats.to_csv(ANALYSIS_DIR / "rq3_external_stats_per_criterion.csv", index=False)

    cohort_df, blind_cohort = external_cohort_breakdown(guided_ext, blind_ext)
    cohort_df.to_csv(ANALYSIS_DIR / "rq3_external_cohort_breakdown.csv", index=False)
    blind_cohort.rename("blind_external_mean").reset_index().to_csv(
        ANALYSIS_DIR / "rq3_blind_external_cohort_baseline.csv", index=False
    )

    # ---------------- Part C ----------------
    print("Part C: guided-vs-blind displacement ...")
    disp = displacement_by_model_criterion_feature(blind, guided, criteria)
    disp.to_csv(ANALYSIS_DIR / "rq3_guided_vs_blind_by_model_criterion_feature.csv", index=False)

    disp_crit = (
        disp.groupby("criterion")["rbo"]
        .agg(rbo_mean="mean", rbo_median="median", rbo_std="std", n="count")
        .sort_values("rbo_mean")
        .reset_index()
    )
    disp_crit.to_csv(ANALYSIS_DIR / "rq3_guided_vs_blind_per_criterion.csv", index=False)
    disp_model = (
        disp.groupby(["model", "criterion"])["rbo"].mean().reset_index()
        .pivot(index="model", columns="criterion", values="rbo")
        .reindex(index=MODEL_KEYS_OPEN + MODEL_KEYS_PROPRIETARY)
    )
    disp_model.to_csv(ANALYSIS_DIR / "rq3_guided_vs_blind_model_x_criterion.csv")
    print(f"  overall guided-vs-blind RBO mean = {disp['rbo'].mean():.4f}")

    examples = mine_new_app_examples(blind, guided, criteria)
    examples.to_csv(ANALYSIS_DIR / "rq3_new_app_examples.csv", index=False)

    # ---------------- Figures ----------------
    print("Figures ...")
    fig_internal_delta_heatmap(delta, criteria)
    fig_external_convergence_per_criterion(ext_summary, per_cf, blind_overall)
    disp_order = disp_crit["criterion"].tolist()  # most to least displacing
    fig_guided_vs_blind_boxplot(disp, disp_order)
    ext_order = ext_summary.sort_values("guided_external_mean", ascending=False)[
        "criterion"
    ].tolist()
    fig_external_cohort_breakdown(cohort_df, blind_cohort, ext_order)

    print("Done. Tables in", ANALYSIS_DIR, "figures in", FIG_DIR)


if __name__ == "__main__":
    main()
