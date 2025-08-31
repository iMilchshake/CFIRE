#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# --------------------------- CLI ----------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Use unified_before_after_deltas.csv (per dataset) and plot BEFORE metrics."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("./experiments/2_grid/results"),
        help="Root directory with per-dataset folders (default: ./experiments/2_grid/results)",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("./experiments/2_grid/plots/log_metrics"),
        help="Output directory for PNGs (default: ./experiments/2_grid/plots/log_metrics)",
    )
    p.add_argument("--show", action="store_true", help="Show plots interactively.")
    return p.parse_args()


# --------------------------- helpers ------------------------------------------
EXPL_CANON = {
    "ig": "IG",
    "integratedgradients": "IG",
    "integrated_gradients": "IG",
    "kernelshap": "KernelSHAP",
    "shap": "KernelSHAP",
    "lime": "LIME",
}
EXPL_ORDER = ["IG", "KernelSHAP", "LIME"]


def ensure_outdir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def coalesce_cols(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def canon_expl(x: str) -> str:
    if not isinstance(x, str):
        return str(x)
    key = x.strip().lower().replace(" ", "").replace("-", "_")
    return EXPL_CANON.get(key, x)


def parse_bin_label(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    m = re.search(r"\(([^)]*)\)", s)
    inside = m.group(1) if m else ""
    if "threshold" in inside.lower():
        m2 = re.search(r"threshold\s*=\s*([0-9.eE+-]+)", inside, re.IGNORECASE)
        if m2:
            return f"threshold={m2.group(1)}"
    if "top" in s.lower() or "k=" in inside.lower():
        m2 = re.search(r"k\s*=\s*([0-9]+)", inside, re.IGNORECASE)
        if m2:
            return f"topk={m2.group(1)}"
    return inside or s


def order_bin_labels(labels: List[str]) -> List[str]:
    def key(lbl: str) -> Tuple[int, float]:
        if lbl.startswith("threshold="):
            try:
                return (0, float(lbl.split("=", 1)[1]))
            except Exception:
                return (0, float("inf"))
        if lbl.startswith("topk="):
            try:
                return (1, float(lbl.split("=", 1)[1]))
            except Exception:
                return (1, float("inf"))
        return (2, float("inf"))
    return sorted(labels, key=key)


# ---- BEFORE-metric resolution tailored for deltas files -----------------------
def resolve_before(df: pd.DataFrame, base: str, fuzzy_terms: Optional[List[str]] = None) -> Optional[str]:
    candidates = [
        f"before_{base}",
        f"{base}_before",
        f"Before:{base}",
        f"before:{base}",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    if fuzzy_terms:
        terms = [t.lower() for t in fuzzy_terms]
        for c in df.columns:
            lc = c.lower()
            if all(t in lc for t in terms) and ("before" in lc or "prior" in lc):
                return c
    return None


def compute_pct_if_possible(df: pd.DataFrame, count_col: Optional[str], total_col: Optional[str]) -> Optional[pd.Series]:
    if count_col and total_col and (count_col in df.columns) and (total_col in df.columns):
        denom = df[total_col].replace(0, np.nan).astype(float)
        return 100.0 * df[count_col].astype(float) / denom
    return None


# --------------------------- loading per dataset ------------------------------
def load_from_deltas(ds_dir: Path) -> Optional[pd.DataFrame]:
    ds_name = ds_dir.name
    f = ds_dir / "unified_before_after_deltas.csv"
    if not f.exists():
        print(f"[skip] {ds_name}: {f.name} not found")
        return None

    df = pd.read_csv(f)

    # Core config columns (usually unprefixed in deltas)
    expl_col = coalesce_cols(df, ["expl_method", "explainer"])
    ft_col   = coalesce_cols(df, ["freq_threshold", "frequency_threshold"])
    bc_col   = coalesce_cols(df, ["bin_config", "binarization", "binarization_config"])
    depth_col= coalesce_cols(df, ["max_dt_depth", "max_depth", "dt_depth"])

    missing_core = [name for name, col in [
        ("expl_method", expl_col),
        ("freq_threshold", ft_col),
        ("bin_config", bc_col),
        ("max_dt_depth", depth_col),
    ] if col is None]
    if missing_core:
        print(f"[warn] {ds_name}: cannot resolve core columns in {f.name}: {missing_core}")
        return None

    # BEFORE metrics
    m_collision_intra = resolve_before(df, "collision_intra_pct", ["collision", "intra", "pct"])
    m_collision_inter = resolve_before(df, "collision_inter_pct", ["collision", "inter", "pct"])
    m_distinct_winner = resolve_before(df, "distinct_winner_clauses", ["distinct", "winner", "clause"])
    m_clause_acc_mean = resolve_before(df, "clause_accuracy_mean", ["clause", "accuracy", "mean"])

    m_smm_pct = resolve_before(df, "samples_multi_match_pct", ["samples", "multi", "match", "pct"])
    if m_smm_pct is None:
        m_smm   = resolve_before(df, "samples_multi_match", ["samples", "multi", "match"])
        m_total = resolve_before(df, "samples_total", ["samples", "total"])
        smm_pct_series = compute_pct_if_possible(df, m_smm, m_total)
    else:
        smm_pct_series = df[m_smm_pct].astype(float)

    norm = pd.DataFrame({
        "dataset": ds_name,
        "explainer": df[expl_col].map(canon_expl),
        "freq_threshold": pd.to_numeric(df[ft_col], errors="coerce"),
        "bin_config": df[bc_col].astype(str),
        "max_dt_depth": pd.to_numeric(df[depth_col], errors="coerce"),
        "collision_intra_pct": df[m_collision_intra].astype(float) if m_collision_intra else np.nan,
        "collision_inter_pct": df[m_collision_inter].astype(float) if m_collision_inter else np.nan,
        "distinct_winner_clauses": df[m_distinct_winner].astype(float) if m_distinct_winner else np.nan,
        "clause_accuracy_mean": df[m_clause_acc_mean].astype(float) if m_clause_acc_mean else np.nan,
        "samples_multi_match_pct": smm_pct_series if smm_pct_series is not None else np.nan,
    })

    norm["bin_label"] = norm["bin_config"].map(parse_bin_label)
    return norm


def scan_all_datasets(root: Path) -> pd.DataFrame:
    rows = []
    for ds_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        df = load_from_deltas(ds_dir)
        if df is not None and not df.empty:
            rows.append(df)
    if not rows:
        raise RuntimeError(f"No datasets loaded from {root}")
    df_all = pd.concat(rows, ignore_index=True)

    ordered_bins = order_bin_labels(sorted(df_all["bin_label"].dropna().unique().tolist()))
    df_all["bin_label"] = pd.Categorical(df_all["bin_label"], categories=ordered_bins, ordered=True)
    return df_all


def get_palette_by_dataset(datasets: List[str]) -> Dict[str, Tuple[float, float, float]]:
    """Stable mapping: dataset -> color from HLS palette (full spectrum, no repeats)."""
    n = max(len(datasets), 1)
    cols = sns.color_palette("hls", n_colors=n)
    return {ds: tuple(cols[i]) for i, ds in enumerate(datasets)}



# ---- aggregation & plotting ---------------------------------------------------
def aggregate_equal_weight_across_explainers(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    lvl1 = (
        df.groupby(["dataset", "explainer", x_col], dropna=False, observed=True)[y_col]
        .mean()
        .reset_index()
    )
    lvl2 = (
        lvl1.groupby(["dataset", x_col], dropna=False, observed=True)[y_col]
        .mean()
        .reset_index()
        .sort_values(["dataset", x_col])
    )
    return lvl2


def aggregate_per_explainer(df: pd.DataFrame, x_col: str, y_col: str, explainer_label: str) -> pd.DataFrame:
    subset = df[df["explainer"] == explainer_label]
    if subset.empty:
        return subset.assign(dummy=np.nan)
    agg = (
        subset.groupby(["dataset", x_col], dropna=False, observed=True)[y_col]
        .mean()
        .reset_index()
        .sort_values(["dataset", x_col])
    )
    return agg


def lineplot_on_ax(ax, data: pd.DataFrame, x_col: str, y_col: str, hue_col: str,
                   palette: Dict[str, Tuple[float, float, float]], x_label_text: Optional[str] = None):
    sns.lineplot(
        data=data, x=x_col, y=y_col, hue=hue_col,
        marker="o", palette=palette, errorbar=None, ax=ax,
    )
    ax.set_xlabel(x_label_text if x_label_text is not None else x_col)
    ax.set_ylabel(y_col)
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def build_four_panel(df: pd.DataFrame, x_col: str, y_col: str, title: str,
                     palette: Dict[str, Tuple[float, float, float]], rotate_xticks: bool = False,
                     x_label_override: Optional[str] = None):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), sharey=True)
    (ax_all, ax_ig, ax_kshap, ax_lime) = axes

    # All (avg across explainers)
    data_all = aggregate_equal_weight_across_explainers(df, x_col=x_col, y_col=y_col)
    lineplot_on_ax(ax_all, data_all, x_col, y_col, hue_col="dataset", palette=palette, x_label_text=x_label_override)
    ax_all.set_title("All (avg)")

    # IG / KernelSHAP / LIME
    for ax, lbl in zip([ax_ig, ax_kshap, ax_lime], EXPL_ORDER):
        data_e = aggregate_per_explainer(df, x_col=x_col, y_col=y_col, explainer_label=lbl)
        if data_e.empty:
            ax.text(0.5, 0.5, f"No data: {lbl}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(lbl)
            ax.set_xlabel(x_label_override if x_label_override is not None else x_col)
            ax.set_ylabel(y_col)
        else:
            lineplot_on_ax(ax, data_e, x_col, y_col, hue_col="dataset", palette=palette, x_label_text=x_label_override)
            ax.set_title(lbl)

    if rotate_xticks:
        for ax in axes:
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha("right")

    fig.suptitle(title, y=1.02, fontsize=12)

    handles, labels = ax_all.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_ig.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="dataset", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


# --------- NEW: filter to fixed defaults for non-plotted hyperparameters -------
DEFAULTS = {
    "freq_threshold": 0.01,
    "max_dt_depth": 7,
    "bin_label": "threshold=0.01",
}

def init_theme(n_colors: int) -> None:
    sns.set_theme(
        palette=sns.color_palette("hls", n_colors=n_colors),
        context="paper",
        style="whitegrid",
    )

def filter_to_defaults(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """Hold non-x hyperparameters at defaults. Uses np.isclose for float comparisons."""
    keep = df.copy()
    if x_col != "freq_threshold":
        # numeric float → use isclose
        keep = keep[np.isfinite(keep["freq_threshold"]) & np.isclose(keep["freq_threshold"], DEFAULTS["freq_threshold"])]
    if x_col != "max_dt_depth":
        keep = keep[keep["max_dt_depth"] == DEFAULTS["max_dt_depth"]]
    if x_col != "bin_label":
        # categorical/string compare
        keep = keep[keep["bin_label"].astype(str) == DEFAULTS["bin_label"]]
    return keep


# --------------------------- main ---------------------------------------------
def main():
    args = parse_args()
    # Build list first so we can size the palette
    df = scan_all_datasets(args.root)
    datasets = sorted(df["dataset"].unique().tolist())

    # Init theme to match coworker
    init_theme(n_colors=len(datasets))

    # Then build explicit mapping (keeps colors consistent across all figures)
    palette = get_palette_by_dataset(datasets)
    ensure_outdir(args.outdir)

    df = scan_all_datasets(args.root)

    # Coerce numeric axes
    df["freq_threshold"] = pd.to_numeric(df["freq_threshold"], errors="coerce")
    df["max_dt_depth"] = pd.to_numeric(df["max_dt_depth"], errors="coerce")

    # Palette by dataset
    datasets = sorted(df["dataset"].unique().tolist())
    palette = get_palette_by_dataset(datasets)

    # Metrics to plot (BEFORE columns resolved earlier)
    metrics = [
        ("collision_intra_pct", "collision_intra_pct"),
        ("collision_inter_pct", "collision_inter_pct"),
        ("distinct_winner_clauses", "distinct_winner_clauses"),
        ("clause_accuracy_mean", "clause_accuracy_mean"),
        ("samples_multi_match_pct", "samples_multi_match_pct"),
    ]

    # X-axes (hyperparameters)
    x_axes = [
        ("bin_label", "binarization", True),
        ("freq_threshold", "freq_threshold", False),
        ("max_dt_depth", "max_dt_depth", False),
    ]

    # 5 metrics × 3 x-axes × 2 variants (avg + fixed) = 30 figures
    for y_col, y_label in metrics:
        if y_col not in df.columns:
            print(f"[warn] Global df missing column '{y_col}' — skipping.")
            continue
        for x_col, x_label, rotate in x_axes:
            # -------- Averaged version (current behavior) --------
            title_avg = f"{y_col} vs {x_label} — averaged others"
            x_label_override = x_label if x_col == "bin_label" else None
            fig = build_four_panel(
                df=df,
                x_col=x_col,
                y_col=y_col,
                title=title_avg,
                palette=palette,
                rotate_xticks=rotate,
                x_label_override=x_label_override,
            )
            outpath = args.outdir / f"{y_col}__vs__{x_label}__avg.png"
            fig.savefig(outpath, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {outpath}")

            # -------- Fixed-defaults version --------
            df_fixed = filter_to_defaults(df, x_col=x_col)
            if df_fixed.empty:
                print(f"[warn] No rows for fixed defaults with x={x_col}; skipping fixed figure.")
                continue
            title_fixed = f"{y_col} vs {x_label} — fixed defaults"
            fig2 = build_four_panel(
                df=df_fixed,
                x_col=x_col,
                y_col=y_col,
                title=title_fixed,
                palette=palette,
                rotate_xticks=rotate,
                x_label_override=x_label_override,
            )
            outpath2 = args.outdir / f"{y_col}__vs__{x_label}__fixed.png"
            fig2.savefig(outpath2, dpi=200, bbox_inches="tight")
            plt.close(fig2)
            print(f"[saved] {outpath2}")

    if args.show:
        print("Saved figures to:", args.outdir)


if __name__ == "__main__":
    main()
