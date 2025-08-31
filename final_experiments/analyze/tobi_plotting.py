# grid_plots_seaborn.py  (updated)
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from final_experiments.analyze.utils import (
    ALL_PARAMS,
    load_csv_files,
    merge_local_explainers,
    filter_all_params_to_default,
    filter_other_params_to_default,
)

# ------------------------------------------------------------------------------
# Paths & global config
# ------------------------------------------------------------------------------
results_dir = Path("./experiments/4_max_dt_depth/results/")
plots_root = Path("./experiments/4_max_dt_depth/plots/")
plots_root.mkdir(parents=True, exist_ok=True)

sns.set_theme()  # default seaborn theme

# For ordering on PLOT4
ORDER_FREQ = [0.001, 0.01, 0.1, 0.25]
ORDER_DEPTH = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]
ORDER_BIN = [
    "ThresholdBinarization(threshold=0.01)",
    "ThresholdBinarization(threshold=0.1)",
    "ThresholdBinarization(threshold=0.25)",
    "TopKBinarization(k=2)",
    "TopKBinarization(k=3)",
]

# Hardcoded mean model accuracies (from test_models.py output)
MODEL_MEAN_ACCS: Dict[str, float] = {
    "abalone": 0.64, "breastw": 0.99, "spambase": 0.93, "beans": 0.90,
    "ionosphere": 0.93, "breastcancer": 0.99, "btsc": 0.80, "spf": 0.75,
    "wine": 0.99, "diggle": 0.95, "iris": 0.92, "vehicle": 0.79, "autouniv": 0.41,
}

EXPLAINERS = ["IG", "lime", "kernelshap", "merged"]  # "merged" = CFIRE view

# ------------------------------------------------------------------------------
# Helpers (re-usable)
# ------------------------------------------------------------------------------
def _save(fig: plt.Figure, fname: str) -> None:
    out = plots_root / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _default_views_all(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Default-params views for CFIRE (merged) and individual local explainers."""
    d = filter_all_params_to_default(df)
    return {
        "merged": merge_local_explainers(d),
        "IG": d[d["expl_method"] == "IG"],
        "lime": d[d["expl_method"] == "lime"],
        "kernelshap": d[d["expl_method"] == "kernelshap"],
    }

def _concat_with_dataset_col(per_ds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for ds, d in per_ds.items():
        if d is None or d.empty:
            continue
        dd = d.copy()
        dd["dataset"] = ds
        frames.append(dd)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _order_for_param(param: str, labels: List[str]) -> List[str]:
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
    return labels

def _views_lock_others(df: pd.DataFrame, vary_param: str) -> Dict[str, pd.DataFrame]:
    """Return per-explainer dfs when varying `vary_param` (others fixed to defaults)."""
    d = filter_other_params_to_default(df, vary_param)
    return {
        "merged": merge_local_explainers(d),
        "IG": d[d["expl_method"] == "IG"],
        "lime": d[d["expl_method"] == "lime"],
        "kernelshap": d[d["expl_method"] == "kernelshap"],
    }

# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------
def load_all() -> Dict[str, pd.DataFrame]:
    return load_csv_files(results_dir, csv_file_name="metrics.csv")

# ------------------------------------------------------------------------------
# PLOT 1 — Val vs Test F1 (defaults), dot per model, colored by dataset
# ------------------------------------------------------------------------------
def plot1_val_vs_test_scatter(data: Dict[str, pd.DataFrame]) -> None:
    per_ds = {}
    for ds, d in data.items():
        per_ds[ds] = _default_views_all(d)["merged"]
    df = _concat_with_dataset_col(per_ds)
    if df.empty:
        return
    df = _ensure_numeric(df, ["val_f1_weighted", "test_f1_weighted"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df,
        x="val_f1_weighted",
        y="test_f1_weighted",
        hue="dataset",
        ax=ax,
        alpha=0.6,          # minor transparency
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlim(0, 1)       # limit to [0,1]
    ax.set_ylim(0, 1)       # limit to [0,1]
    ax.set_title("Default CFIRE: Validation vs Test F1 (per model)")
    _save(fig, "PLOT1_val_vs_test_scatter.png")

# ------------------------------------------------------------------------------
# PLOT 2 — Local explainer performance (defaults), faceted by dataset; show x labels on all
# ------------------------------------------------------------------------------
def _plot2_one_metric(data: Dict[str, pd.DataFrame], metric: str, fname: str, ylabel: str | None = None) -> None:
    recs = []
    for ds, d in data.items():
        views = _default_views_all(d)
        for method, dd in views.items():
            if dd is None or dd.empty:
                continue
            ddm = _ensure_numeric(dd, [metric])
            for v in ddm[metric].dropna().tolist():
                recs.append({"dataset": ds, "method": method, "value": v})
    if not recs:
        return
    df = pd.DataFrame(recs)
    g = sns.catplot(
        data=df, x="method", y="value", col="dataset",
        kind="bar", col_wrap=4, sharey=False, height=3.2, aspect=1.1,
        errorbar="sd"
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Method", ylabel or metric)
    g.fig.suptitle(f"Default configuration — {metric}", y=1.03)
    _save(g.fig, fname)

def plot2_local_explainers_bars(data: Dict[str, pd.DataFrame]) -> None:
    _plot2_one_metric(data, "val_f1_weighted", "PLOT2A_local_bar_val_f1.png", "val_f1_weighted")
    _plot2_one_metric(data, "test_f1_weighted", "PLOT2B_local_bar_test_f1.png", "test_f1_weighted")
    _plot2_one_metric(data, "rule_size", "PLOT2C_local_bar_rule_size.png", "rule_size")

# ------------------------------------------------------------------------------
# PLOT 3 — IOU boxplots (defaults), show all local explainers, larger layout
# ------------------------------------------------------------------------------
def plot3_iou_boxplots(data: Dict[str, pd.DataFrame]) -> None:
    recs = []
    for ds, d in data.items():
        views = _default_views_all(d)
        for expl, dd in views.items():
            if dd is None or dd.empty:
                continue
            dd = _ensure_numeric(dd, ["mean_iou", "max_iou"])
            for m in ["mean_iou", "max_iou"]:
                for v in dd[m].dropna().tolist():
                    recs.append({"dataset": ds, "metric": m, "value": v, "explainer": expl})
    if not recs:
        return
    df = pd.DataFrame(recs)
    g = sns.catplot(
        data=df, x="dataset", y="value",
        row="explainer", col="metric",
        kind="box", sharey=False,
        height=3.8, aspect=1.4  # increase size
    )
    g.set_titles("{row_name} — {col_name}")
    g.set_axis_labels("Dataset", "Value")
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
    g.fig.suptitle("Default CFIRE — IOU metrics across explainers", y=1.02)
    _save(g.fig, "PLOT3_iou_boxplots.png")

# ------------------------------------------------------------------------------
# PLOT 4 — Hyperparameter sweeps (others default), show all local explainers
#          NOW: relative gains/losses (%) vs each dataset+explainer default
# ------------------------------------------------------------------------------
def _baseline_defaults_by_explainer(data: Dict[str, pd.DataFrame],
                                    metric: str = "test_f1_weighted") -> Dict[tuple, float]:
    """Return {(dataset, explainer): baseline_mean} for default params."""
    baselines: Dict[tuple, float] = {}
    for ds, d in data.items():
        d_def = filter_all_params_to_default(d)
        views = {
            "merged": merge_local_explainers(d_def),
            "IG": d_def[d_def["expl_method"] == "IG"],
            "lime": d_def[d_def["expl_method"] == "lime"],
            "kernelshap": d_def[d_def["expl_method"] == "kernelshap"],
        }
        for expl, dd in views.items():
            if dd is None or dd.empty:
                continue
            dd = _ensure_numeric(dd, [metric])
            base = dd[metric].dropna().mean()
            if pd.notna(base):
                baselines[(ds, expl)] = float(base)
    return baselines

def _hp_lineplot_relative(data: Dict[str, pd.DataFrame], hp: str, fname: str) -> None:
    metric = "test_f1_weighted"
    baselines = _baseline_defaults_by_explainer(data, metric=metric)

    recs = []
    for ds, d in data.items():
        views = _views_lock_others(d, hp)  # vary hp, others default
        for expl, dd in views.items():
            if dd is None or dd.empty:
                continue
            base = baselines.get((ds, expl), None)
            if base is None or base == 0:
                continue
            dd = _ensure_numeric(dd, [metric])
            for _, row in dd.iterrows():
                val = row[metric]
                if pd.isna(val):
                    continue
                rel = 100.0 * (val - base) / abs(base)
                recs.append({
                    "dataset": ds,
                    "explainer": expl,
                    "x": str(row[hp]),
                    "delta_pct": rel,
                })
    if not recs:
        return

    df = pd.DataFrame(recs)
    ordered = _order_for_param(hp, list(pd.unique(df["x"].tolist())))
    g = sns.relplot(
        data=df, x="x", y="delta_pct", hue="dataset",
        row="explainer", kind="line", marker="o",
        estimator="mean",
        # errorbar="sd",
        errorbar=None,
        height=3.2, aspect=2.0, facet_kws=dict(sharex=False)
    )
    for ax in g.axes.flat:
        ax.set_xlabel(hp)
        ax.set_ylabel("Δ% test_f1_weighted vs default")
        ax.axhline(0.0, ls="--", lw=1, color="0.5")
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels(ordered, rotation=20, ha="right")
    g.fig.suptitle(f"CFIRE: relative Δ% in test_f1 vs {hp} (others at default) — all explainers", y=1.02)
    _save(g.fig, fname)

def plot4_hparam_sweeps(data: Dict[str, pd.DataFrame]) -> None:
    _hp_lineplot_relative(data, "freq_threshold", "PLOT4A_rel_testF1_vs_freq_threshold.png")
    _hp_lineplot_relative(data, "max_dt_depth", "PLOT4B_rel_testF1_vs_max_dt_depth.png")
    _hp_lineplot_relative(data, "bin_config", "PLOT4C_rel_testF1_vs_bin_config.png")

# ------------------------------------------------------------------------------
# PLOT 5 — Feature activity ratios (defaults), show all local explainers
# ------------------------------------------------------------------------------
def plot5_bin_activity_ratios(data: Dict[str, pd.DataFrame]) -> None:
    metrics = ["bin_all_features_active_ratio", "bin_all_features_inactive_ratio"]
    recs = []
    for ds, d in data.items():
        views = _default_views_all(d)
        for expl, dd in views.items():
            if dd is None or dd.empty:
                continue
            dd = _ensure_numeric(dd, metrics)
            for m in metrics:
                for v in dd[m].dropna().tolist():
                    recs.append({"dataset": ds, "metric": m, "value": v, "explainer": expl})
    if not recs:
        return
    df = pd.DataFrame(recs)
    g = sns.catplot(
        data=df, x="dataset", y="value",
        row="explainer", col="metric",
        kind="box", sharey=False,
        height=3.8, aspect=1.4
    )
    g.set_titles("{row_name} — {col_name}")
    g.set_axis_labels("Dataset", "Value")
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
    g.fig.suptitle("Default CFIRE — Feature activity ratios across explainers", y=1.02)
    _save(g.fig, "PLOT5_bin_activity_boxplots.png")

# ------------------------------------------------------------------------------
# PLOT 6 — Literal count: TopK(2) vs Threshold(0.01) (others default), all explainers
# ------------------------------------------------------------------------------
def plot6_literal_count_topk_vs_threshold(data: Dict[str, pd.DataFrame]) -> None:
    desired = {
        "TopKBinarization(k=2)": "TopK=2",
        "ThresholdBinarization(threshold=0.01)": "Thresh=0.01",
    }
    recs = []
    for ds, d in data.items():
        dd = filter_other_params_to_default(d, "bin_config")
        dd = dd[dd["bin_config"].astype(str).isin(desired.keys())]
        if dd.empty:
            continue
        views = {
            "merged": merge_local_explainers(dd),
            "IG": dd[dd["expl_method"] == "IG"],
            "lime": dd[dd["expl_method"] == "lime"],
            "kernelshap": dd[dd["expl_method"] == "kernelshap"],
        }
        for expl, dv in views.items():
            if dv is None or dv.empty:
                continue
            dv = _ensure_numeric(dv, ["literal_count"])
            for _, row in dv.iterrows():
                recs.append({
                    "dataset": ds,
                    "explainer": expl,
                    "bin_config": desired[str(row["bin_config"])],
                    "literal_count": row["literal_count"],
                })
    if not recs:
        return
    df = pd.DataFrame(recs)
    g = sns.catplot(
        data=df, x="dataset", y="literal_count", hue="bin_config",
        row="explainer", kind="box",
        height=3.6, aspect=1.8, sharey=False
    )
    g.set_titles("{row_name}")
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
    g.fig.suptitle("Literal count — TopK(2) vs Threshold(0.01) (others at default), across explainers", y=1.02)
    _save(g.fig, "PLOT6_literal_count_topk_vs_threshold.png")

# ------------------------------------------------------------------------------
# PLOT 7 — CFIRE performance vs. model accuracy (scatter, means per dataset), limit axes
# ------------------------------------------------------------------------------
def plot7_cfire_vs_model_accuracy(data: Dict[str, pd.DataFrame]) -> None:
    recs = []
    for ds, d in data.items():
        dd = _default_views_all(d)["merged"]
        if dd is None or dd.empty:
            continue
        dd = _ensure_numeric(dd, ["test_f1_weighted"])
        mean_cfire = dd["test_f1_weighted"].dropna().mean()
        model_acc = MODEL_MEAN_ACCS.get(ds, None)
        if pd.isna(mean_cfire) or model_acc is None:
            continue
        recs.append({"dataset": ds, "model_acc": model_acc, "cfire_test_f1": float(mean_cfire)})
    if not recs:
        return
    df = pd.DataFrame(recs)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sns.scatterplot(data=df, x="model_acc", y="cfire_test_f1", hue="dataset", s=60, ax=ax, alpha=0.7)
    ax.set_title("Mean CFIRE test F1 vs Mean Model Accuracy (default config)")
    ax.set_xlabel("Mean model accuracy")
    ax.set_ylabel("Mean CFIRE test F1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "PLOT7_cfire_vs_model_accuracy.png")

# ------------------------------------------------------------------------------
# PLOT 8 — Scatter with x-jitter: max_dt_depth vs literal_count (others default), alpha
# ------------------------------------------------------------------------------
def plot8_depth_vs_literal_count(data: Dict[str, pd.DataFrame]) -> None:
    recs = []
    for ds, d in data.items():
        dd = filter_other_params_to_default(d, "max_dt_depth")
        dd = merge_local_explainers(dd)
        if dd.empty:
            continue
        dd = _ensure_numeric(dd, ["max_dt_depth", "literal_count"])
        for _, row in dd.iterrows():
            recs.append({
                "dataset": ds,
                "max_dt_depth": row["max_dt_depth"],
                "literal_count": row["literal_count"],
            })
    if not recs:
        return
    df = pd.DataFrame(recs)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.stripplot(
        data=df,
        x="max_dt_depth",
        y="literal_count",
        hue="dataset",
        dodge=True,
        jitter=True,
        alpha=0.6,
        ax=ax
    )
    ax.set_title("CFIRE: max_dt_depth vs literal_count (others at default)")
    ax.set_xlabel("max_dt_depth")
    ax.set_ylabel("literal_count")
    _save(fig, "PLOT8_depth_vs_literal_count_strip.png")


# THROWAWAY
def compare_depth_2_vs_7(data: Dict[str, pd.DataFrame]) -> None:
    metric = "test_f1_weighted"
    rows = []
    for ds, d in data.items():
        dd = filter_other_params_to_default(d, "max_dt_depth")
        if dd.empty:
            continue
        merged = merge_local_explainers(dd)
        if merged.empty:
            continue
        merged = _ensure_numeric(merged, ["max_dt_depth", metric])

        f2 = merged.loc[merged["max_dt_depth"] == 2, metric].dropna().mean()
        f7 = merged.loc[merged["max_dt_depth"] == 7, metric].dropna().mean()
        if pd.isna(f2) or pd.isna(f7):
            continue

        rel = 100.0 * (f2 - f7) / abs(f7)
        rows.append((ds, f2, f7, rel))

    rows.sort(key=lambda x: x[3], reverse=True)

    print("Dataset       | F1@depth=2 | F1@depth=7 | Δ% (2 vs 7, merged)")
    print("-" * 62)
    for ds, f2, f7, rel in rows:
        print(f"{ds:12s} | {f2:.3f}       | {f7:.3f}       | {rel:+.1f}%")


def table_val_vs_test_deviation(data: Dict[str, pd.DataFrame]) -> None:
    recs = []
    for ds, d in data.items():
        merged = _default_views_all(d)["merged"]
        if merged is None or merged.empty:
            continue
        merged = _ensure_numeric(merged, ["val_f1_weighted", "test_f1_weighted"])
        merged = merged.dropna(subset=["val_f1_weighted", "test_f1_weighted"])

        diffs = (merged["val_f1_weighted"] - merged["test_f1_weighted"]).tolist()
        if not diffs:
            continue

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        recs.append((ds, mean_diff, std_diff, len(diffs)))

    # print table
    print("Dataset      | N models | Mean(val - test) | Std")
    print("----------------------------------------------------")
    for ds, mean_d, std_d, n in recs:
        print(f"{ds:12s} | {n:8d} | {mean_d:+.3f}          | {std_d:.3f}")

def compare_depth2_vs7_with_valtest(data: Dict[str, pd.DataFrame]) -> None:
    metric = "test_f1_weighted"
    explainers = ["IG", "lime", "kernelshap", "merged"]

    for expl in explainers:
        rows = []
        for ds, d in data.items():
            # vary only max_dt_depth
            dd = filter_other_params_to_default(d, "max_dt_depth")
            if dd.empty:
                continue

            # pick explainer view
            if expl == "merged":
                dd = merge_local_explainers(dd)
            else:
                dd = dd[dd["expl_method"] == expl]

            if dd.empty:
                continue

            dd = _ensure_numeric(
                dd, ["max_dt_depth", metric, "val_f1_weighted"]
            )
            dd = dd.dropna(subset=[metric, "val_f1_weighted", "max_dt_depth"])

            f2 = dd.loc[dd["max_dt_depth"] == 2, metric].mean()
            f7 = dd.loc[dd["max_dt_depth"] == 7, metric].mean()
            if pd.isna(f2) or pd.isna(f7):
                continue

            delta = 100.0 * (f2 - f7) / abs(f7)

            # val-test deviation (defaults only!)
            d_def = filter_all_params_to_default(d)
            if expl == "merged":
                d_def = merge_local_explainers(d_def)
            else:
                d_def = d_def[d_def["expl_method"] == expl]

            if d_def.empty:
                continue
            d_def = _ensure_numeric(d_def, ["val_f1_weighted", metric])
            d_def = d_def.dropna(subset=["val_f1_weighted", metric])

            diffs = (d_def["val_f1_weighted"] - d_def[metric]).tolist()
            mean_diff = np.mean(diffs) if diffs else np.nan

            rows.append((ds, delta, mean_diff))

        # sort by Δ%
        rows.sort(key=lambda x: x[1], reverse=True)

        # print table
        print(f"\n=== Explainer: {expl} ===")
        print("Dataset      | Δ% (2 vs 7) | Mean(val - test)")
        print("------------------------------------------------")
        for ds, delta, mean_diff in rows:
            print(f"{ds:12s} | {delta:+.1f}%       | {mean_diff:+.3f}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    data = load_all()
    if not data:
        print("[WARN] No data found under results dir.")
        return

    print("[INFO] Comparing max_dt_depth=2 vs 7")
    # compare_depth_2_vs_7(data)
    # table_val_vs_test_deviation(data)
    compare_depth2_vs7_with_valtest(data)
    exit(0)

    print("[INFO] Generating PLOT1")
    plot1_val_vs_test_scatter(data)

    print("[INFO] Generating PLOT2")
    plot2_local_explainers_bars(data)

    print("[INFO] Generating PLOT3")
    plot3_iou_boxplots(data)

    print("[INFO] Generating PLOT4")
    plot4_hparam_sweeps(data)

    print("[INFO] Generating PLOT5")
    plot5_bin_activity_ratios(data)

    print("[INFO] Generating PLOT6")
    plot6_literal_count_topk_vs_threshold(data)

    print("[INFO] Generating PLOT7")
    plot7_cfire_vs_model_accuracy(data)

    print("[INFO] Generating PLOT8")
    plot8_depth_vs_literal_count(data)

    print(f"[OK] Plots saved under: {plots_root.resolve()}/")

if __name__ == "__main__":
    main()
