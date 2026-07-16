"""Model-level RBO plots — minimal multi-k layouts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

DEFAULT_RBO_P = 0.9

# Minimal palette
_BG = "#FAFBFC"
_BOX_FILL = "#C5DCEB"
_BOX_EDGE = "#4A7C9B"
_MEDIAN = "#1E3A4F"
_TEXT = "#1F2937"
_TICK = "#1F2937"
_ANNOT_LIGHT = "#FFFFFF"
_ANNOT_DARK = "#1F2937"

# Soft teal sequential (low = light teal, high = deep teal)
RBO_CMAP = LinearSegmentedColormap.from_list(
    "rbo_teal",
    ["#D8EAF2", "#B9D6E8", "#5A9AB5", "#1E4D63"],
    N=256,
)

# Typography — heatmap figures
_HM_TICK = 14
_HM_ANNOT = 14
_HM_TITLE = 16
_HM_SUPTITLE = 15
_HM_CBAR = 14


def _style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#D0D7DE",
            "axes.labelcolor": _TEXT,
            "xtick.color": _TICK,
            "ytick.color": _TICK,
            "text.color": _TEXT,
            "figure.facecolor": _BG,
            "axes.facecolor": "white",
        }
    )


_DIAG_FILL = "#ECEFF3"


def _text_color_for_cell(
    value: float,
    cmap: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
) -> str:
    span = vmax - vmin
    norm = 0.5 if span <= 0 else (float(value) - vmin) / span
    norm = min(1.0, max(0.0, norm))
    rgba = cmap(norm)
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return _ANNOT_LIGHT if lum < 0.58 else _ANNOT_DARK


def _off_diagonal_vrange(matrices: list[pd.DataFrame]) -> tuple[float, float]:
    vals: list[float] = []
    for matrix in matrices:
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if i != j:
                    vals.append(float(matrix.iloc[i, j]))
    if not vals:
        return 0.0, 1.0
    vmin, vmax = min(vals), max(vals)
    pad = max(0.01, (vmax - vmin) * 0.06)
    return vmin, vmax + pad


def _annotate_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    cmap: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
) -> None:
    from matplotlib.patches import Rectangle

    n_rows, n_cols = matrix.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val = float(matrix.iloc[i, j])
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
            color = _text_color_for_cell(val, cmap, vmin, vmax)
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=_HM_ANNOT,
                color=color,
                zorder=3,
            )


def _style_heatmap_axes(
    ax: plt.Axes,
    models: list[str],
    *,
    tick_fontsize: float = _HM_TICK,
) -> None:
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=tick_fontsize, color=_TICK)
    ax.set_yticklabels(models, rotation=0, fontsize=tick_fontsize, color=_TICK)
    ax.tick_params(axis="both", colors=_TICK, length=0, pad=8)


def _draw_diagonal_placeholders(ax: plt.Axes, n: int) -> None:
    from matplotlib.patches import Rectangle

    for i in range(n):
        ax.add_patch(
            Rectangle(
                (i, i),
                1,
                1,
                fill=True,
                facecolor=_DIAG_FILL,
                edgecolor="white",
                linewidth=0.6,
                zorder=2,
            )
        )


def _annotate_upper_triangle(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    cmap: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
    *,
    fontsize: float = _HM_ANNOT,
) -> None:
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            val = float(matrix.iloc[i, j])
            color = _text_color_for_cell(val, cmap, vmin, vmax)
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
                zorder=3,
            )


def _boxplot_kwargs(*, widths: float = 0.34) -> dict:
    return dict(
        patch_artist=True,
        showfliers=False,
        widths=widths,
        boxprops={"facecolor": _BOX_FILL, "edgecolor": _BOX_EDGE, "linewidth": 0.9},
        medianprops={"color": _MEDIAN, "linewidth": 1.3},
        whiskerprops={"color": _BOX_EDGE, "linewidth": 0.9},
        capprops={"color": _BOX_EDGE, "linewidth": 0.9},
    )


def _format_stats(mean: float, std: float) -> str:
    return f"μ={mean:.2f}  σ={std:.2f}"


def _mean_matrix(summary: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=models, columns=models)
    for _, row in summary.iterrows():
        m1, m2 = row["model1"], row["model2"]
        v = float(row["rbo_mean"])
        matrix.loc[m1, m2] = v
        matrix.loc[m2, m1] = v
    for m in models:
        matrix.loc[m, m] = 1.0
    return matrix.fillna(0.0)


def _cross_family_matrix(
    summary: pd.DataFrame,
    open_models: list[str],
    proprietary_models: list[str],
) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=open_models, columns=proprietary_models)
    for _, row in summary.iterrows():
        matrix.loc[row["open_model"], row["proprietary_model"]] = float(row["rbo_mean"])
    return matrix.fillna(0.0)


def _matrix_vrange(matrix: pd.DataFrame) -> tuple[float, float]:
    vals = [float(v) for v in matrix.to_numpy().flatten() if not np.isnan(v)]
    if not vals:
        return 0.0, 1.0
    vmin, vmax = min(vals), max(vals)
    pad = max(0.015, (vmax - vmin) * 0.08)
    return vmin - pad, vmax + pad


def plot_all_models_external_rbo(
    summary: pd.DataFrame,
    *,
    open_models: list[str],
    proprietary_models: list[str],
    title: str,
    output_path: Path,
    k: int,
    p: float = DEFAULT_RBO_P,
) -> None:
    """Square heatmap for all models (open block, then proprietary), family dividers."""
    all_models = open_models + [m for m in proprietary_models if m not in open_models]
    if summary.empty or len(all_models) < 2:
        return

    _style()
    matrix = _mean_matrix(summary, all_models)
    n_open = len(open_models)
    n = len(all_models)
    diag_mask = np.eye(n, dtype=bool)
    vmin, vmax = _off_diagonal_vrange([matrix])

    fig, ax = plt.subplots(figsize=(max(8.5, n * 1.05 + 2.0), max(8.0, n * 1.0 + 2.0)))
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
        cbar_kws={"label": "Mean RBO", "shrink": 0.82},
    )
    _annotate_heatmap(ax, matrix, RBO_CMAP, vmin, vmax)
    _style_heatmap_axes(ax, all_models)

    if 0 < n_open < n:
        ax.axhline(n_open, color="#4A5568", linewidth=2.0, zorder=4)
        ax.axvline(n_open, color="#4A5568", linewidth=2.0, zorder=4)
        ax.text(
            n_open / 2,
            -0.95,
            "Open-source",
            ha="center",
            va="top",
            transform=ax.transData,
            fontsize=_HM_TICK,
            color=_TEXT,
            clip_on=False,
        )
        ax.text(
            n_open + (n - n_open) / 2,
            -0.95,
            "Proprietary",
            ha="center",
            va="top",
            transform=ax.transData,
            fontsize=_HM_TICK,
            color=_TEXT,
            clip_on=False,
        )

    cbar = ax.collections[0].colorbar if ax.collections else None
    if cbar is not None:
        cbar.ax.yaxis.label.set_color(_TEXT)
        cbar.ax.yaxis.label.set_fontsize(_HM_CBAR + 1)
        cbar.ax.tick_params(colors=_TICK, labelsize=_HM_CBAR)

    ax.set_title(
        f"All models — external RBO | k={k}, p={p}",
        fontsize=_HM_TITLE,
        fontweight="500",
        pad=16,
        color=_TEXT,
    )
    fig.suptitle(title, y=1.02, fontsize=_HM_SUPTITLE, fontweight="500", color=_TEXT)
    plt.tight_layout(pad=1.6)
    fig.subplots_adjust(bottom=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_cross_family_external_rbo(
    summary: pd.DataFrame,
    *,
    open_models: list[str],
    proprietary_models: list[str],
    title: str,
    output_path: Path,
    k: int,
    p: float = DEFAULT_RBO_P,
) -> None:
    """Rectangular heatmap: open models (rows) × proprietary models (cols)."""
    if summary.empty:
        return

    _style()
    matrix = _cross_family_matrix(summary, open_models, proprietary_models)
    vmin, vmax = _matrix_vrange(matrix)
    n_open, n_prop = len(open_models), len(proprietary_models)

    fig, ax = plt.subplots(
        figsize=(max(6.0, n_prop * 1.05 + 1.8), max(5.0, n_open * 0.75 + 1.8)),
    )
    sns.heatmap(
        matrix,
        annot=False,
        cmap=RBO_CMAP,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Mean RBO", "shrink": 0.85},
    )
    for i in range(n_open):
        for j in range(n_prop):
            val = float(matrix.iloc[i, j])
            color = _text_color_for_cell(val, RBO_CMAP, vmin, vmax)
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=9.5,
                color=color,
                zorder=3,
            )

    ax.set_xticklabels(proprietary_models, rotation=45, ha="right", fontsize=9.5, color=_TICK)
    ax.set_yticklabels(open_models, rotation=0, fontsize=9.5, color=_TICK)
    ax.set_xlabel("Proprietary models", fontsize=10, color=_TEXT, labelpad=8)
    ax.set_ylabel("Open models", fontsize=10, color=_TEXT, labelpad=8)
    ax.tick_params(axis="both", colors=_TICK, length=0, pad=6)
    ax.set_title(f"Cross-family external RBO | k={k}, p={p}", fontsize=11, fontweight="500", pad=12, color=_TEXT)
    fig.suptitle(title, y=1.02, fontsize=12, fontweight="500", color=_TEXT)
    plt.tight_layout(pad=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_rbo_p_sensitivity(
    summaries_by_p: dict[float, pd.DataFrame],
    *,
    title: str,
    output_path: Path,
    analysis_label: str,
) -> None:
    """Line plot of mean off-diagonal / overall mean RBO vs persistence parameter p."""
    if not summaries_by_p:
        return

    _style()
    p_values = sorted(summaries_by_p)
    means: list[float] = []
    for p in p_values:
        summary = summaries_by_p[p]
        if summary.empty:
            means.append(float("nan"))
            continue
        if "open_model" in summary.columns:
            vals = summary["rbo_mean"].astype(float).tolist()
        elif "model1" in summary.columns:
            vals = summary["rbo_mean"].astype(float).tolist()
        else:
            vals = summary["rbo_mean"].astype(float).tolist()
        means.append(float(np.mean(vals)) if vals else float("nan"))

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(p_values, means, marker="o", color="#1E4D63", linewidth=1.8, markersize=7)
    for x, y in zip(p_values, means):
        if not np.isnan(y):
            ax.text(x, y + 0.012, f"{y:.3f}", ha="center", va="bottom", fontsize=9, color=_TEXT)
    ax.set_xlabel("RBO persistence parameter (p)", fontsize=10, color=_TEXT)
    ax.set_ylabel("Mean RBO", fontsize=10, color=_TEXT)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="#E8ECF0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(analysis_label, fontsize=11, fontweight="500", pad=10, color=_TEXT)
    fig.suptitle(title, y=1.02, fontsize=12, fontweight="500", color=_TEXT)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_external_rbo_multi_k(
    summaries_by_k: dict[int, pd.DataFrame],
    *,
    k_values: list[int],
    title: str,
    output_path: Path,
    models: list[str],
    p: float = DEFAULT_RBO_P,
) -> None:
    """One heatmap per k: mean RBO across features (model × model)."""
    plot_k = [
        k
        for k in k_values
        if k in summaries_by_k and summaries_by_k[k] is not None and not summaries_by_k[k].empty
    ]
    if not plot_k:
        return

    _style()
    n = len(models)
    n_k = len(plot_k)
    matrices = [_mean_matrix(summaries_by_k[k], models) for k in plot_k]
    vmin, vmax = _off_diagonal_vrange(matrices)
    diag_mask = np.eye(n, dtype=bool)

    fig, axes = plt.subplots(
        1,
        n_k,
        figsize=(max(4.8, n * 0.9) * n_k + 0.3, max(4.5, n * 0.85)),
        squeeze=False,
    )

    for col, k in enumerate(plot_k):
        ax = axes[0, col]
        matrix = matrices[col]
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
            cbar=col == n_k - 1,
            cbar_kws={"label": "Mean RBO", "shrink": 0.85} if col == n_k - 1 else {},
        )
        _annotate_heatmap(ax, matrix, RBO_CMAP, vmin, vmax)
        _style_heatmap_axes(ax, models)
        ax.set_title(f"k = {k}", fontsize=11, fontweight="500", pad=12, color=_TEXT)

    if n_k > 0 and axes[0, n_k - 1].collections:
        cbar = axes[0, n_k - 1].collections[0].colorbar
        if cbar is not None:
            cbar.ax.yaxis.label.set_color(_TEXT)
            cbar.ax.tick_params(colors=_TICK, labelsize=9)

    fig.suptitle(
        title,
        y=1.04,
        fontsize=12,
        fontweight="500",
        color=_TEXT,
    )
    if n_k == 1:
        fig.axes[0].set_title(f"External RBO | k={plot_k[0]}, p={p}", fontsize=11, fontweight="500", pad=12, color=_TEXT)
    plt.tight_layout(pad=1.2)
    fig.subplots_adjust(bottom=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output_path}")


def _internal_panel(
    ax: plt.Axes,
    detail: pd.DataFrame,
    summary: pd.DataFrame | None,
    models: list[str],
    *,
    k: int,
) -> None:
    plot_df = detail[detail["model"].isin(models)]
    order = [m for m in models if m in plot_df["model"].unique()]
    if not order:
        ax.set_title(f"k = {k} (no data)")
        return

    positions = np.arange(len(order))
    data = [plot_df.loc[plot_df["model"] == m, "rbo"].to_numpy(dtype=float) for m in order]

    ax.boxplot(data, positions=positions, vert=False, **_boxplot_kwargs())

    stats_by_model: dict[str, tuple[float, float]] = {}
    if summary is not None and not summary.empty:
        for _, row in summary.iterrows():
            stats_by_model[row["model"]] = (float(row["rbo_mean"]), float(row["rbo_std"]))

    for i, m in enumerate(order):
        vals = data[i]
        if m in stats_by_model:
            mean, std = stats_by_model[m]
        elif vals.size:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        else:
            continue
        ax.text(
            1.01,
            i,
            _format_stats(mean, std),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8.5,
            color=_TEXT,
        )
        ax.plot(mean, i, "o", color=_MEDIAN, markersize=5, zorder=5, markeredgecolor="white", markeredgewidth=0.6)

    n_models = len(order)
    ax.set_yticks(positions)
    ax.set_yticklabels(order, fontsize=9.5, color=_TICK)
    ax.set_ylim(-0.5, n_models - 0.5)
    ax.set_xlim(0, 1.12)
    ax.margins(y=0)
    ax.tick_params(axis="x", colors=_TICK, labelsize=9)
    ax.set_xlabel("RBO", fontsize=9.5, color=_TEXT)
    ax.set_title(f"k = {k}", fontsize=11, fontweight="500", pad=8)
    ax.grid(axis="x", color="#E8ECF0", linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E0E6ED")
    ax.spines["bottom"].set_color("#E0E6ED")


def plot_internal_rbo_multi_k(
    by_feature_by_k: dict[int, pd.DataFrame],
    summaries_by_k: dict[int, pd.DataFrame],
    *,
    k_values: list[int],
    title: str,
    output_path: Path,
    models: list[str],
) -> None:
    """One column per k: horizontal boxplots with μ/σ labels."""
    plot_k = [k for k in k_values if k in by_feature_by_k and not by_feature_by_k[k].empty]
    if not plot_k:
        return

    _style()
    n_k = len(plot_k)
    n_models = len(models)
    row_h = max(2.5, n_models * 0.34 + 0.6)
    fig, axes = plt.subplots(
        1,
        n_k,
        figsize=(4.2 * n_k + 0.4, row_h),
        squeeze=False,
    )
    for col, k in enumerate(plot_k):
        _internal_panel(
            axes[0, col],
            by_feature_by_k[k],
            summaries_by_k.get(k),
            models,
            k=k,
        )
        if col > 0:
            axes[0, col].set_yticklabels([])
            axes[0, col].set_ylabel("")
        if col < n_k - 1:
            axes[0, col].set_xlabel("")

    fig.suptitle(title, y=1.03, fontsize=12, fontweight="500", color=_TEXT)
    if n_k == 1:
        fig.axes[0].set_title(f"Internal RBO | k={plot_k[0]}", fontsize=11, fontweight="500", pad=8, color=_TEXT)
    plt.tight_layout(pad=0.6)
    fig.subplots_adjust(top=0.88)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output_path}")
