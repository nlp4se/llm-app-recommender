"""Publication-quality RQ2 consistency figures (PDF + PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from code.consistency.apps import internal_app_rbo_summary, external_app_rbo_summary

K = 20
P = 0.9
DPI = 300

# Runtime keys (bundle) → short display labels for paper figures
MODEL_KEYS_PROPRIETARY = ["openai", "gemini", "anthropic", "mistral"]
MODEL_KEYS_OPEN = [
    "llama31_8b",
    "gemma3_4b",
    "qwen3_8b",
    "gptoss20b",
    "mistral_open",
    "deepseekr1_8b",
]
ALL_MODEL_KEYS = MODEL_KEYS_PROPRIETARY + MODEL_KEYS_OPEN

DISPLAY_NAMES: dict[str, str] = {
    "openai": "GPT-5.3",
    "gemini": "Gemini-3",
    "anthropic": "Claude-Opus-4.6",
    "mistral": "Mistral-Large",
    "llama31_8b": "Llama4-Scout",
    "gemma3_4b": "Gemma3-27B",
    "qwen3_8b": "Qwen3-30B",
    "gptoss20b": "GPT-OSS-20B",
    "mistral_open": "Mistral-Small-24B",
    "deepseekr1_8b": "DeepSeek-R1-8B",
}

COHORT_BY_KEY: dict[str, str] = {
    **{k: "proprietary" for k in MODEL_KEYS_PROPRIETARY},
    **{k: "open" for k in MODEL_KEYS_OPEN},
}

# Okabe–Ito colorblind-safe cohort colors (consistent across bar charts)
COLOR_PROPRIETARY = "#0072B2"
COLOR_OPEN = "#E69F00"
COHORT_COLORS = {"proprietary": COLOR_PROPRIETARY, "open": COLOR_OPEN}
# Darker cohort shades for std error bars — readable on both bar fills
COLOR_PROPRIETARY_ERR = "#003F6B"
COLOR_OPEN_ERR = "#B35C00"
COHORT_ERR_COLORS = {"proprietary": COLOR_PROPRIETARY_ERR, "open": COLOR_OPEN_ERR}
COLOR_WEB_AUG = "#56B4E9"
COLOR_WEB_AUG_ERR = "#1A6B94"

_BG = "#FAFBFC"
_TEXT = "#1F2937"
_TICK = "#1F2937"
_ANNOT_LIGHT = "#FFFFFF"
_ANNOT_DARK = "#1F2937"
_DIAG_FILL = "#ECEFF3"

RBO_CMAP = LinearSegmentedColormap.from_list(
    "rbo_teal",
    ["#D8EAF2", "#B9D6E8", "#5A9AB5", "#1E4D63"],
    N=256,
)

# Single-column ~8.5 cm; wide heatmap ~13 cm
COL_WIDTH_IN = 8.5 / 2.54
WIDE_WIDTH_IN = 13.0 / 2.54

# Compact horizontal bar layout (shared publication bar charts)
PUB_BAR_HEIGHT = 0.44
PUB_BASE_SIZE = 7.5
PUB_LEGEND_SIZE = 6.5
FIG_ROW_H = 0.17
FIG_BASE_H = 0.55
BAR_LABEL_SIZE = 7.0
# Paired bars (ablation): two thick bars per row, combined ≈ PUB_BAR_HEIGHT
PUB_PAIR_BAR_HEIGHT = 0.40
PUB_PAIR_OFFSET = PUB_PAIR_BAR_HEIGHT / 2 + 0.005


def _compact_fig_height(n_models: int) -> float:
    return max(2.0, n_models * FIG_ROW_H + FIG_BASE_H)


def _paired_fig_height(n_models: int) -> float:
    """Same row pitch as main bar charts, without the 10-model minimum height."""
    return n_models * FIG_ROW_H + FIG_BASE_H


def _add_setting_error_bars(
    ax: plt.Axes,
    y_pos: np.ndarray,
    values: pd.Series | np.ndarray,
    errors: pd.Series | np.ndarray,
    *,
    err_color: str,
) -> None:
    for y, val, err in zip(y_pos, values, errors):
        ax.errorbar(
            float(val),
            float(y),
            xerr=float(err),
            fmt="none",
            ecolor=err_color,
            elinewidth=0.8,
            capsize=1.2,
            capthick=0.7,
            zorder=4,
        )


def _add_std_error_bars(
    ax: plt.Axes,
    y_pos: np.ndarray,
    values: pd.Series | np.ndarray,
    errors: pd.Series | np.ndarray,
    cohorts: list[str],
) -> None:
    """Per-cohort error bars in a darker shade of each bar color."""
    for y, val, err, cohort in zip(y_pos, values, errors, cohorts):
        ax.errorbar(
            float(val),
            float(y),
            xerr=float(err),
            fmt="none",
            ecolor=COHORT_ERR_COLORS[cohort],
            elinewidth=0.9,
            capsize=2.0,
            capthick=0.9,
            zorder=4,
        )


def _annotate_horizontal_bar_values(
    ax: plt.Axes,
    bars,
    values: np.ndarray | pd.Series,
    *,
    xlim_max: float,
    fmt: str = "{:.2f}",
    min_inside_frac: float = 0.10,
) -> None:
    """Uniform in-bar labels with stroke for contrast; outside when bar is too narrow."""
    for bar, val in zip(bars, values):
        frac = float(val) / xlim_max if xlim_max > 0 else 0.0
        y = bar.get_y() + bar.get_height() / 2
        label = fmt.format(val)
        if frac >= min_inside_frac:
            x = float(val) - xlim_max * 0.015
            txt = ax.text(
                x,
                y,
                label,
                va="center",
                ha="right",
                fontsize=BAR_LABEL_SIZE,
                color="white",
                fontweight="bold",
                zorder=5,
            )
            txt.set_path_effects(
                [pe.withStroke(linewidth=1.6, foreground="#1F2937", alpha=0.9)]
            )
        else:
            x = float(val) + xlim_max * 0.015
            txt = ax.text(
                x,
                y,
                label,
                va="center",
                ha="left",
                fontsize=BAR_LABEL_SIZE,
                color=_TEXT,
                fontweight="bold",
                zorder=5,
            )


def _cohort_legend_handles(prop_mean: float, open_mean: float) -> list:
    """Legend entries for cohort colors plus dashed mean reference lines."""
    return [
        Patch(facecolor=COLOR_PROPRIETARY, edgecolor="white", label="Proprietary"),
        Patch(facecolor=COLOR_OPEN, edgecolor="white", label="Open-source"),
        Line2D(
            [0],
            [0],
            color=COLOR_PROPRIETARY,
            linestyle="--",
            linewidth=1.2,
            label=f"Proprietary μ = {prop_mean:.3f}",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_OPEN,
            linestyle="--",
            linewidth=1.2,
            label=f"Open-source μ = {open_mean:.3f}",
        ),
    ]


def _style_pub(*, base_size: float = 8.5) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": base_size,
            "axes.titlesize": base_size,
            "axes.labelsize": base_size,
            "xtick.labelsize": base_size - 0.5,
            "ytick.labelsize": base_size - 0.5,
            "legend.fontsize": base_size - 1.0,
            "axes.edgecolor": "#D0D7DE",
            "axes.labelcolor": _TEXT,
            "xtick.color": _TICK,
            "ytick.color": _TICK,
            "text.color": _TEXT,
            "figure.facecolor": _BG,
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=DPI,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"Saved {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")


def _load_large_rankings(experiment_root: Path) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(experiment_root / "rq2/proprietary_large/app_rankings.csv"),
            pd.read_csv(experiment_root / "rq2/open_large/app_rankings.csv"),
        ],
        ignore_index=True,
    )


def _distinct_apps_per_model(rank_df: pd.DataFrame, k: int = K) -> pd.DataFrame:
    rank_cols = [str(i) for i in range(1, k + 1)]
    rows: list[dict] = []
    for model, feat_df in rank_df.groupby(["model", "feature"]):
        model_key, feature = model
        apps: set[str] = set()
        for _, row in feat_df.iterrows():
            for col in rank_cols:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    apps.add(str(val).strip())
        rows.append(
            {
                "model": model_key,
                "feature": feature,
                "n_distinct": len(apps),
                "n_runs": len(feat_df),
            }
        )
    detail = pd.DataFrame(rows)
    agg = (
        detail.groupby("model")
        .agg(
            mean_distinct=("n_distinct", "mean"),
            std_distinct=("n_distinct", "std"),
            n_features=("feature", "count"),
        )
        .reset_index()
    )
    agg["display"] = agg["model"].map(DISPLAY_NAMES)
    agg["cohort"] = agg["model"].map(COHORT_BY_KEY)
    return agg


def _text_color_for_cell(value: float, vmin: float, vmax: float) -> str:
    span = vmax - vmin
    norm = 0.5 if span <= 0 else (float(value) - vmin) / span
    norm = min(1.0, max(0.0, norm))
    rgba = RBO_CMAP(norm)
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return _ANNOT_LIGHT if lum < 0.58 else _ANNOT_DARK


def _mean_matrix(summary: pd.DataFrame, model_keys: list[str]) -> pd.DataFrame:
    labels = [DISPLAY_NAMES[k] for k in model_keys]
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    for _, row in summary.iterrows():
        m1, m2 = row["model1"], row["model2"]
        v = float(row["rbo_mean"])
        l1, l2 = DISPLAY_NAMES[m1], DISPLAY_NAMES[m2]
        matrix.loc[l1, l2] = v
        matrix.loc[l2, l1] = v
    return matrix.fillna(0.0)


def plot_distinct_apps_per_model(rank_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Figure 1: horizontal bar chart, sorted ascending, cohort colors."""
    _style_pub(base_size=PUB_BASE_SIZE)
    data = _distinct_apps_per_model(rank_df)
    prop_mean = data.loc[data["cohort"] == "proprietary", "mean_distinct"].mean()
    open_mean = data.loc[data["cohort"] == "open", "mean_distinct"].mean()
    data = data.sort_values("mean_distinct", ascending=True)

    bar_height = PUB_BAR_HEIGHT
    fig_h = _compact_fig_height(len(data))
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, fig_h))

    y_pos = np.arange(len(data))
    colors = [COHORT_COLORS[c] for c in data["cohort"]]
    ax.barh(
        y_pos,
        data["mean_distinct"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=bar_height,
        zorder=3,
    )
    _add_std_error_bars(
        ax,
        y_pos,
        data["mean_distinct"],
        data["std_distinct"],
        data["cohort"].tolist(),
    )

    xlim_max = float((data["mean_distinct"] + data["std_distinct"]).max() * 1.08)

    ax.axvline(prop_mean, color=COLOR_PROPRIETARY, linestyle="--", linewidth=1.1, zorder=2)
    ax.axvline(open_mean, color=COLOR_OPEN, linestyle="--", linewidth=1.1, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(data["display"])
    ax.set_xlabel("")
    ax.set_xlim(0, xlim_max)
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.margins(y=0)
    ax.grid(axis="x", color="#E8ECF0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=COLOR_PROPRIETARY, edgecolor="white", label="Proprietary"),
        Patch(facecolor=COLOR_OPEN, edgecolor="white", label="Open-source"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=PUB_LEGEND_SIZE,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    plt.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94))
    _save(fig, output_dir / "distinct_apps_per_model")
    return data


def plot_internal_rbo_per_model(rank_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Figure 2: internal RBO bars, descending, cohort means as dashed lines."""
    _style_pub(base_size=PUB_BASE_SIZE)
    features = sorted(rank_df.feature.unique())
    summary = internal_app_rbo_summary(rank_df, ALL_MODEL_KEYS, features, K, p=P)
    summary["display"] = summary["model"].map(DISPLAY_NAMES)
    summary["cohort"] = summary["model"].map(COHORT_BY_KEY)
    summary = summary.sort_values("rbo_mean", ascending=True)

    prop_mean = summary.loc[summary["cohort"] == "proprietary", "rbo_mean"].mean()
    open_mean = summary.loc[summary["cohort"] == "open", "rbo_mean"].mean()

    bar_height = PUB_BAR_HEIGHT
    fig_h = _compact_fig_height(len(summary))
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, fig_h))

    y_pos = np.arange(len(summary))
    colors = [COHORT_COLORS[c] for c in summary["cohort"]]
    ax.barh(
        y_pos,
        summary["rbo_mean"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=bar_height,
        zorder=3,
    )
    _add_std_error_bars(
        ax,
        y_pos,
        summary["rbo_mean"],
        summary["rbo_std"],
        summary["cohort"].tolist(),
    )

    ax.axvline(prop_mean, color=COLOR_PROPRIETARY, linestyle="--", linewidth=1.1, zorder=2)
    ax.axvline(open_mean, color=COLOR_OPEN, linestyle="--", linewidth=1.1, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary["display"])
    ax.set_xlabel("")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(summary) - 0.4)
    ax.margins(y=0)
    ax.grid(axis="x", color="#E8ECF0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=COLOR_PROPRIETARY, edgecolor="white", label="Proprietary"),
        Patch(facecolor=COLOR_OPEN, edgecolor="white", label="Open-source"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=PUB_LEGEND_SIZE,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    plt.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94))
    _save(fig, output_dir / "internal_rbo_per_model")
    return summary


def plot_external_rbo_heatmap(rank_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Figure 3: 10×10 external RBO heatmap, open block then proprietary."""
    _style_pub(base_size=9.0)
    features = sorted(rank_df.feature.unique())
    # Order: open-source first, then proprietary (matches existing separator layout)
    model_keys = MODEL_KEYS_OPEN + MODEL_KEYS_PROPRIETARY
    display_labels = [DISPLAY_NAMES[k] for k in model_keys]
    n_open = len(MODEL_KEYS_OPEN)

    summary = external_app_rbo_summary(rank_df, model_keys, features, K, p=P)
    matrix = _mean_matrix(summary, model_keys)
    n = len(model_keys)
    diag_mask = np.eye(n, dtype=bool)

    off_vals = [
        float(matrix.iloc[i, j])
        for i in range(n)
        for j in range(n)
        if i != j
    ]
    vmin, vmax = min(off_vals), max(off_vals)
    pad = max(0.01, (vmax - vmin) * 0.06)
    vmin, vmax = vmin, vmax + pad

    fig_size = max(WIDE_WIDTH_IN, WIDE_WIDTH_IN * 0.95)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.92))

    sns.heatmap(
        matrix,
        annot=False,
        cmap=RBO_CMAP,
        vmin=vmin,
        vmax=vmax,
        mask=diag_mask,
        ax=ax,
        square=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Mean RBO (k=20, p=0.9)", "shrink": 0.78},
    )

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(
                    Rectangle(
                        (j, i),
                        1,
                        1,
                        fill=True,
                        facecolor=_DIAG_FILL,
                        edgecolor="white",
                        linewidth=0.6,
                        zorder=2,
                    )
                )
                continue
            val = float(matrix.iloc[i, j])
            color = _text_color_for_cell(val, vmin, vmax)
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8.0,
                color=color,
                zorder=3,
            )

    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8.5, color=_TICK)
    ax.set_yticklabels(display_labels, rotation=0, fontsize=8.5, color=_TICK)
    ax.tick_params(axis="both", colors=_TICK, length=0, pad=6)

    if 0 < n_open < n:
        ax.axhline(n_open, color="#374151", linewidth=2.2, zorder=4)
        ax.axvline(n_open, color="#374151", linewidth=2.2, zorder=4)

    cbar = ax.collections[0].colorbar if ax.collections else None
    if cbar is not None:
        cbar.ax.yaxis.label.set_color(_TEXT)
        cbar.ax.yaxis.label.set_fontsize(8.5)
        cbar.ax.tick_params(colors=_TICK, labelsize=8.0)

    plt.tight_layout(pad=0.8)
    _save(fig, output_dir / "external_rbo_heatmap")
    return summary


def plot_ablation_internal_rbo(experiment_root: Path, output_dir: Path) -> pd.DataFrame:
    """Figure 4 (optional): grouped horizontal bars, KO vs web-augmented."""
    _style_pub(base_size=PUB_BASE_SIZE)
    wo_path = experiment_root / "rq2/proprietary_small_wo_websearch/app_rankings.csv"
    w_path = experiment_root / "rq2/proprietary_small/app_rankings.csv"
    wo_df = pd.read_csv(wo_path)
    w_df = pd.read_csv(w_path)
    features = sorted(set(wo_df.feature) & set(w_df.feature))

    int_wo = internal_app_rbo_summary(wo_df, MODEL_KEYS_PROPRIETARY, features, K, p=P)
    int_w = internal_app_rbo_summary(w_df, MODEL_KEYS_PROPRIETARY, features, K, p=P)
    merged = int_wo.merge(int_w, on="model", suffixes=("_wo", "_w"))
    merged["display"] = merged["model"].map(DISPLAY_NAMES)
    merged = merged.sort_values("rbo_mean_wo", ascending=True)

    ko_mean = merged["rbo_mean_wo"].mean()
    web_mean = merged["rbo_mean_w"].mean()

    n_models = len(merged)
    fig_h = _paired_fig_height(n_models) + 0.30
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, fig_h))

    y_pos = np.arange(n_models)
    y_ko = y_pos + PUB_PAIR_OFFSET
    y_web = y_pos - PUB_PAIR_OFFSET

    ax.barh(
        y_ko,
        merged["rbo_mean_wo"],
        height=PUB_PAIR_BAR_HEIGHT,
        color=COLOR_PROPRIETARY,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.barh(
        y_web,
        merged["rbo_mean_w"],
        height=PUB_PAIR_BAR_HEIGHT,
        color=COLOR_WEB_AUG,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    _add_setting_error_bars(
        ax, y_ko, merged["rbo_mean_wo"], merged["rbo_std_wo"], err_color=COLOR_PROPRIETARY_ERR
    )
    _add_setting_error_bars(
        ax, y_web, merged["rbo_mean_w"], merged["rbo_std_w"], err_color=COLOR_WEB_AUG_ERR
    )

    ax.axvline(ko_mean, color=COLOR_PROPRIETARY, linestyle="--", linewidth=1.1, zorder=2)
    ax.axvline(web_mean, color=COLOR_WEB_AUG, linestyle="--", linewidth=1.1, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["display"])
    ax.set_xlabel("")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, n_models - 0.4)
    ax.margins(y=0)
    ax.grid(axis="x", color="#E8ECF0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=COLOR_PROPRIETARY, edgecolor="white", label="Knowledge-only"),
        Patch(facecolor=COLOR_WEB_AUG, edgecolor="white", label="Web-augmented"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=PUB_LEGEND_SIZE,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    plt.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94))
    _save(fig, output_dir / "ablation_internal_rbo")
    return merged


def generate_rq2_publication_figures(
    *,
    experiment_root: str | Path = "data/output/features",
    output_dir: str | Path | None = None,
) -> None:
    """Generate all RQ2 publication figures from source ranking bundles."""
    root = Path(experiment_root)
    out = Path(output_dir) if output_dir else root / "rq2" / "publication"
    rank_df = _load_large_rankings(root)

    print("Recomputing distinct apps per model …")
    da = plot_distinct_apps_per_model(rank_df, out)
    print(da[["display", "mean_distinct", "std_distinct"]].to_string(index=False))

    print("Recomputing internal RBO per model …")
    ir = plot_internal_rbo_per_model(rank_df, out)
    print(ir[["display", "rbo_mean", "rbo_std"]].to_string(index=False))

    print("Recomputing external RBO heatmap …")
    plot_external_rbo_heatmap(rank_df, out)

    print("Recomputing ablation internal RBO …")
    plot_ablation_internal_rbo(root, out)


if __name__ == "__main__":
    generate_rq2_publication_figures()
