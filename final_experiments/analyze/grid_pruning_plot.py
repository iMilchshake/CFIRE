# grid_pruning_before_after.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

# use a non-interactive backend for servers/CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from final_experiments.analyze.utils import (
    ALL_PARAMS,
    filter_other_params_to_default,
    merge_local_explainers,
    get_metric_table,
    load_csv_files,
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
results_dir = Path("./experiments/2_grid/results/")
out_path = Path("./experiments/2_grid/results_grid2_pruning.txt")
plots_root = Path("./experiments/2_grid/plots/")

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
METRIC_SELECTION: List[str] = [
    "val_acc",
    "val_f1_weighted",
    "test_f1_weighted",
    "test_acc",
    "rule_size",
    "rule_count",
    "literal_count",
    "unique_literal_count",
    "max_iou",
    "mean_iou",
    "t_rule_candidates",
    "t_compose_rules",
    "missing_class_rules",
    "missing_pred_val",
    "missing_pred_test",
    "mean_coverage_ratio",
    "mean_single_coverage_ratio",
    "mean_nodes_per_sample",
    "mean_duplicate_nodes_ratio",
    "total_frequent_node_count",
    "attr_mean_absolute_attribution",
    "attr_attribution_variance",
    "attr_sparsity",
    "attr_class_separation",
    "bin_mean_active_features_per_sample",
    "bin_mean_active_features_ratio",
    "bin_mean_feature_activation_ratio",
    "bin_features_inactive_ratio",
    "bin_mean_feature_class_specificity",
    "bin_mean_within_class_jaccard",
    "bin_mean_across_class_jaccard",
    "bin_class_separation_score",
    "bin_all_features_active_ratio",
    "bin_all_features_inactive_ratio",
]

EXPLAINERS = ["IG", "lime", "kernelshap", "merged"]

# Plots: what to visualize
PLOT_PARAMS = ["freq_threshold", "max_dt_depth", "bin_config"]
PLOT_METRICS = ["test_f1_weighted", "rule_size", "max_iou"]

# nice consistent styling
EXPLAINER_COLORS = {
    "IG": "#1f77b4",          # blue
    "lime": "#2ca02c",        # green
    "kernelshap": "#d62728",  # red
}
PHASE_MARKERS = {
    "before": "o",
    "after_best": "s",
    "after_safe": "^",
}

# ordering helpers for x-axis
ORDER_FREQ = [0.001, 0.01, 0.1, 0.25]
ORDER_DEPTH = [2, 3, 7, 14]
ORDER_BIN = [
    "ThresholdBinarization(threshold=0.01)",
    "ThresholdBinarization(threshold=0.1)",
    "ThresholdBinarization(threshold=0.25)",
    "TopKBinarization(k=2)",
    "TopKBinarization(k=3)",
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _parse_mean(value) -> Optional[float]:
    """Extract the mean from strings like '0.76±0.03' or return float if numeric."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        for sep in ("±", "+/-", "+–", "+-"):
            if sep in s:
                s = s.split(sep)[0].strip()
                break
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _with_subcolumns(
        base_table: pd.DataFrame,
        best_table: pd.DataFrame,
        safe_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every parameter value column, create a 5-subcolumn block:
      before | after_best | after_safe | Δbest | Δsafe
    Δ columns are differences of MEANS only (signed).
    """
    metrics_index = base_table.index
    all_cols = list(
        dict.fromkeys(
            list(base_table.columns)
            + list(best_table.columns)
            + list(safe_table.columns)
        )
    )

    base_table = base_table.reindex(columns=all_cols)
    best_table = best_table.reindex(columns=all_cols)
    safe_table = safe_table.reindex(columns=all_cols)

    blocks = []
    for col in all_cols:
        before = base_table[col]
        after_b = best_table[col]
        after_s = safe_table[col]

        d_best, d_safe = [], []
        for b, ab, as_ in zip(before.tolist(), after_b.tolist(), after_s.tolist()):
            mb = _parse_mean(b)
            mab = _parse_mean(ab)
            mas = _parse_mean(as_)
            db = (mab - mb) if (mb is not None and mab is not None) else None
            ds = (mas - mb) if (mb is not None and mas is not None) else None
            d_best.append("" if db is None else f"{db:+.3f}")
            d_safe.append("" if ds is None else f"{ds:+.3f}")

        block = pd.DataFrame(
            {
                (str(col), "before"): before,
                (str(col), "after_best"): after_b,
                (str(col), "after_safe"): after_s,
                (str(col), "Δbest"): d_best,
                (str(col), "Δsafe"): d_safe,
            },
            index=metrics_index,
        )
        blocks.append(block)

    out = pd.concat(blocks, axis=1)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def _subset_for_expl(
        df_base: pd.DataFrame, df_best: pd.DataFrame, df_safe: pd.DataFrame, expl: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if expl == "merged":
        return df_base, df_best, df_safe
    return (
        df_base[df_base["expl_method"] == expl],
        df_best[df_best["expl_method"] == expl],
        df_safe[df_safe["expl_method"] == expl],
    )


def _order_labels(param: str, labels: List[str]) -> List[str]:
    """Return labels ordered per spec, preserving any unexpected values at the end."""
    if param == "freq_threshold":
        desired = [str(x) for x in ORDER_FREQ]
        labels_s = [str(x) for x in labels]
        rest = [x for x in labels_s if x not in desired]
        return [x for x in desired if x in labels_s] + sorted(rest)
    if param == "max_dt_depth":
        desired = [str(x) for x in ORDER_DEPTH]
        labels_s = [str(x) for x in labels]
        rest = [x for x in labels_s if x not in desired]
        return [x for x in desired if x in labels_s] + sorted(rest, key=lambda z: int(float(z)))
    if param == "bin_config":
        desired = ORDER_BIN[:]
        rest = [x for x in labels if x not in desired]
        return [x for x in desired if x in labels] + rest
    # fallback: keep natural order
    return list(labels)


def _plot_one(
        dataset: str,
        param: str,
        metric: str,
        df_base: pd.DataFrame,
        df_best: pd.DataFrame,
        df_safe: pd.DataFrame,
) -> None:
    """
    Build a single figure for (dataset, param, metric).
    X-axis: param values; Y-axis: metric mean
    Lines: per explainer (IG/lime/kernelshap), each with 3 markers (before / after_best / after_safe)
    """
    # Prepare output dir
    out_dir = plots_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # We'll unify X across phases & explainers by taking union of columns from "before" tables
    # to ensure we show all tried values.
    # We build per-explainer series then plot.
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for expl in ["IG", "lime", "kernelshap"]:
        sub_b, sub_best, sub_safe = _subset_for_expl(df_base, df_best, df_safe, expl)

        # build compact tables containing only the metric row
        tbl_b = get_metric_table(sub_b, param, [metric])
        tbl_best = get_metric_table(sub_best, param, [metric])
        tbl_safe = get_metric_table(sub_safe, param, [metric])

        # union of all labels for this explainer (stringified for consistent keys)
        all_labels = list(
            dict.fromkeys(
                [str(c) for c in tbl_b.columns]
                + [str(c) for c in tbl_best.columns]
                + [str(c) for c in tbl_safe.columns]
            )
        )
        ordered_labels = _order_labels(param, all_labels)

        def extract_numeric(tbl: pd.DataFrame, ordered: List[str]) -> List[float]:
            series = []
            for lab in ordered:
                val = None
                if lab in [str(c) for c in tbl.columns]:
                    # match actual column name by string
                    real_col = [c for c in tbl.columns if str(c) == lab][0]
                    cell = tbl.loc[metric, real_col] if metric in tbl.index else None
                    val = _parse_mean(cell)
                series.append(np.nan if val is None else float(val))
            return series

        y_before = extract_numeric(tbl_b, ordered_labels)
        y_best = extract_numeric(tbl_best, ordered_labels)
        y_safe = extract_numeric(tbl_safe, ordered_labels)

        x = np.arange(len(ordered_labels))
        color = EXPLAINER_COLORS.get(expl, None)

        ax.plot(
            x, y_before,
            marker=PHASE_MARKERS["before"],
            label=f"{expl} • before",
            color=color, linewidth=1.8, alpha=0.95,
        )
        ax.plot(
            x, y_best,
            marker=PHASE_MARKERS["after_best"],
            label=f"{expl} • after_best",
            color=color, linewidth=1.8, alpha=0.95, linestyle="--",
        )
        ax.plot(
            x, y_safe,
            marker=PHASE_MARKERS["after_safe"],
            label=f"{expl} • after_safe",
            color=color, linewidth=1.8, alpha=0.95, linestyle=":",
        )

    ax.set_title(f"{dataset} — {metric} vs {param}")
    ax.set_xlabel(param)
    ax.set_ylabel(metric)
    ax.set_xticks(np.arange(len(ordered_labels)))
    ax.set_xticklabels(ordered_labels, rotation=20, ha="right")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    # keep legend readable: 3 explainers * 3 phases
    # Legend outside (right)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),   # outside right
        frameon=True,
        fontsize=9,
        title="explainer • phase",
    )

    # Leave room on the right for the legend
    fig.tight_layout(rect=[0, 0, 0.78, 1])

    out_file = out_dir / f"{dataset}__{metric}__{param}.png"
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _analyze_one_dataset(
        name: str,
        base_df: pd.DataFrame,
        best_df: pd.DataFrame,
        safe_df: pd.DataFrame,
) -> str:
    """Builds the text section for a single dataset (same format as before) and creates plots."""
    lines: List[str] = []
    lines.append(f"\n\n{'='*15} ANALYSIS FOR DATASET: '{name}' {'='*15}")

    # --- tables (unchanged from your previous script) ---
    for param in ALL_PARAMS:
        df_base = filter_other_params_to_default(base_df, param)
        df_best = filter_other_params_to_default(best_df, param)
        df_safe = filter_other_params_to_default(safe_df, param)

        df_cfire_base = merge_local_explainers(df_base)
        df_cfire_best = merge_local_explainers(df_best)
        df_cfire_safe = merge_local_explainers(df_safe)

        for expl in EXPLAINERS:
            if expl == "merged":
                sub_b, sub_best, sub_safe = df_cfire_base, df_cfire_best, df_cfire_safe
            else:
                sub_b, sub_best, sub_safe = _subset_for_expl(df_base, df_best, df_safe, expl)

            tbl_b = get_metric_table(sub_b, param, METRIC_SELECTION)
            tbl_best = get_metric_table(sub_best, param, METRIC_SELECTION)
            tbl_safe = get_metric_table(sub_safe, param, METRIC_SELECTION)

            combined = _with_subcolumns(tbl_b, tbl_best, tbl_safe)

            lines.append(f"\n ======= [{name}] param={param} | expl={expl} ======")
            try:
                lines.append(combined.to_string())
            except Exception:
                lines.append(str(combined))

        # --- plots (new) ---
        if param in PLOT_PARAMS:
            # re-use the same filtered frames for plotting
            for metric in PLOT_METRICS:
                _plot_one(name, param, metric, df_base, df_best, df_safe)

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    # Each dataset dir contains: metrics.csv, metrics_best_prune.csv, metrics_safe_prune.csv
    base: Dict[str, pd.DataFrame] = load_csv_files(results_dir, csv_file_name="metrics.csv")
    best: Dict[str, pd.DataFrame] = load_csv_files(results_dir, csv_file_name="metrics_best_prune.csv")
    safe: Dict[str, pd.DataFrame] = load_csv_files(results_dir, csv_file_name="metrics_safe_prune.csv")

    blocks: List[str] = []
    for dataset in sorted(base.keys()):
        if dataset not in best or dataset not in safe:
            print(f"[WARN] Skipping dataset '{dataset}' — best/safe prune files missing.")
            continue
        print(f"[INFO] Analyzing and plotting dataset: {dataset}")
        blocks.append(_analyze_one_dataset(dataset, base[dataset], best[dataset], safe[dataset]))

    # write the text file (same as before)
    text = "\n".join(blocks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    print(f"[OK] Wrote pruning-aware grid to: {out_path.resolve()}")
    print(f"[OK] Plots saved under: {plots_root.resolve()}/<dataset>/*.png")


if __name__ == "__main__":
    main()
