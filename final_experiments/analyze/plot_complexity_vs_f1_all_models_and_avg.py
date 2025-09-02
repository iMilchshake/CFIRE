from __future__ import annotations

import argparse
from pathlib import Path

# Headless-safe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import re
from final_experiments.analyze.utils import init_theme

DEFAULT_FREQ = 0.01
DEFAULT_DT_DEPTH = 7
DEFAULT_BIN_CONFIG_STR = "ThresholdBinarization(threshold=0.01)"

# Metrics to show on the y-axis (normalized)
METRICS = [
    "rule_count",
    "rule_size",
    "literal_count",
    "mean_nodes_per_sample",
    "avg_rules_matched_per_sample",
]

VARIANT_ORDER = ["Before", "After (Safe)", "After (Best)"]
VARIANT_TO_F1COL = {
    "Before": "test_f1_weighted_before",
    "After (Safe)": "test_f1_weighted_after_safe",
    "After (Best)": "test_f1_weighted_after_best",
}

MARKERS = {"Before": "o", "After (Safe)": "s", "After (Best)": "^"}

def parse_bin_param(bin_cfg: str) -> str:
    """Return the content inside parentheses for title, e.g.:
    'ThresholdBinarization(threshold=0.01)' -> 'threshold=0.01'
    'TopKBinarization(k=2)'                -> 'k=2'
    """
    if not isinstance(bin_cfg, str):
        return "threshold=0.01"
    m = re.search(r"\(([^)]*)\)", bin_cfg)
    return m.group(1) if m else "threshold=0.01"

def compute_default_root(script_file: str) -> Path:
    script_dir = Path(script_file).resolve().parent
    repo_root = script_dir.parent.parent  # ../../
    return repo_root / "experiments" / "2_grid" / "results"

def find_datasets(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])

def load_unified(ds_dir: Path) -> pd.DataFrame:
    csv_path = ds_dir / "unified_before_after_deltas.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run compute_before_after_deltas.py first.")
    return pd.read_csv(csv_path)

def filter_defaults(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["freq_threshold"] == DEFAULT_FREQ) & (df["max_dt_depth"] == DEFAULT_DT_DEPTH)
    if "bin_config" in df.columns:
        bin_str = df["bin_config"].astype(str).str.strip()
        mask &= (bin_str == DEFAULT_BIN_CONFIG_STR)
    return df.loc[mask].copy()

def to_long_norm_with_f1(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Long tidy DF with normalized metric (Before=100) and matching test_f1_weighted for each variant."""
    rows = []
    # Ensure F1 columns exist (we'll skip rows where not)
    need_f1 = {"test_f1_weighted_before", "test_f1_weighted_after_safe", "test_f1_weighted_after_best"}
    if not need_f1.issubset(df.columns):
        return pd.DataFrame(columns=["dataset","metric","expl_method","model_idx","variant","value_norm","f1","bin_param"])

    for m in METRICS:
        b, s, bt = f"{m}_before", f"{m}_after_safe", f"{m}_after_best"
        if not all(c in df.columns for c in (b, s, bt)):
            continue

        sub = df[["model_idx", "expl_method", "bin_config", b, s, bt,
                  "test_f1_weighted_before", "test_f1_weighted_after_safe", "test_f1_weighted_after_best"]].copy()

        # normalize metric: Before = 100
        base = sub[b].replace({0.0: np.nan})
        norm_s  = 100.0 * (sub[s]  / base)
        norm_bt = 100.0 * (sub[bt] / base)

        # Build rows per variant, carrying the matching test_f1_weighted value
        rows.append(pd.DataFrame({
            "dataset": dataset_name,
            "metric": m,
            "model_idx": sub["model_idx"],
            "expl_method": sub["expl_method"],
            "variant": "Before",
            "value_norm": 100.0,
            "f1": sub["test_f1_weighted_before"],
            "bin_param": sub["bin_config"].map(parse_bin_param),
        }))
        rows.append(pd.DataFrame({
            "dataset": dataset_name,
            "metric": m,
            "model_idx": sub["model_idx"],
            "expl_method": sub["expl_method"],
            "variant": "After (Safe)",
            "value_norm": norm_s,
            "f1": sub["test_f1_weighted_after_safe"],
            "bin_param": sub["bin_config"].map(parse_bin_param),
        }))
        rows.append(pd.DataFrame({
            "dataset": dataset_name,
            "metric": m,
            "model_idx": sub["model_idx"],
            "expl_method": sub["expl_method"],
            "variant": "After (Best)",
            "value_norm": norm_bt,
            "f1": sub["test_f1_weighted_after_best"],
            "bin_param": sub["bin_config"].map(parse_bin_param),
        }))

    if not rows:
        return pd.DataFrame(columns=["dataset","metric","expl_method","model_idx","variant","value_norm","f1","bin_param"])

    long_df = pd.concat(rows, ignore_index=True)
    # drop rows where normalization or f1 is NaN (e.g., base==0)
    long_df = long_df.dropna(subset=["value_norm", "f1"])
    long_df["variant"] = pd.Categorical(long_df["variant"], categories=VARIANT_ORDER, ordered=True)
    long_df["variant_order"] = long_df["variant"].cat.codes
    return long_df

def build_all_long(root: Path) -> tuple[pd.DataFrame, str]:
    frames = []
    bin_param_text = None
    for ds in find_datasets(root):
        try:
            df = load_unified(ds)
        except FileNotFoundError:
            continue
        dff = filter_defaults(df)
        if dff.empty:
            continue
        if bin_param_text is None and "bin_config" in dff.columns and not dff["bin_config"].empty:
            bin_param_text = parse_bin_param(str(dff["bin_config"].iloc[0]))
        long_df = to_long_norm_with_f1(dff, dataset_name=ds.name)
        if not long_df.empty:
            frames.append(long_df)
    if not frames:
        return pd.DataFrame(columns=["dataset","metric","model_idx","expl_method","variant","variant_order","value_norm","f1","bin_param"]), (bin_param_text or "threshold=0.01")
    return pd.concat(frames, ignore_index=True), (bin_param_text or "threshold=0.01")

def _per_model_paths(data: pd.DataFrame, palette_map: dict[str, tuple], **kwargs):
    """Draw faint lines per (dataset, model_idx) following VARIANT_ORDER in (f1, value_norm) space."""
    ax = plt.gca()
    # group by dataset + model to keep path color consistent
    for (dset, mid), grp in data.groupby(["dataset", "model_idx"]):
        g = grp.sort_values("variant_order")
        color = palette_map.get(dset, None)
        ax.plot(g["f1"], g["value_norm"], linewidth=0.7, alpha=0.25, color=color, zorder=1)

def plot_all_datasets_by_metric(df_long: pd.DataFrame, metric: str, out_dir: Path, bin_param_text: str) -> Path:
    data = df_long[df_long["metric"] == metric]
    if data.empty:
        raise ValueError(f"No data for metric '{metric}'.")

    # Build consistent hue palette
    hue_order = sorted(data["dataset"].unique().tolist())
    palette = sns.color_palette("deep", n_colors=len(hue_order))
    palette_map = dict(zip(hue_order, palette))


    g = sns.FacetGrid(
        data, col="expl_method", hue="dataset", hue_order=hue_order,
        col_wrap=3, sharex=True, sharey=True, height=4.0, palette=palette
    )

    # 1) Faint per-model paths (custom)
    g.map_dataframe(_per_model_paths, palette_map=palette_map)

    # 2) Scatter points with variant-specific markers
    g.map_dataframe(
        sns.scatterplot,
        x="f1", y="value_norm",
        style="variant", style_order=VARIANT_ORDER, markers=MARKERS,
        s=35, alpha=0.9, edgecolor="white", linewidth=0.3, zorder=2,
        legend=False,
    )

    # 3) Bold mean path per dataset (connect mean of each variant in order)
    # Compute means first:
    means = (
        data.groupby(["expl_method", "dataset", "variant"], as_index=False)
        .agg(f1=("f1", "mean"), value_norm=("value_norm", "mean"))
    )
    means["variant_order"] = pd.Categorical(means["variant"], categories=VARIANT_ORDER, ordered=True).codes
    for ax, expl in zip(g.axes.flat, g.col_names):
        sub = means[means["expl_method"] == expl]
        for dset in hue_order:
            sg = sub[sub["dataset"] == dset].sort_values("variant_order")
            if len(sg) >= 2:
                ax.plot(
                    sg["f1"], sg["value_norm"],
                    color=palette_map[dset], linewidth=2.2, marker="o", zorder=3
                )
            elif len(sg) == 1:
                ax.scatter(
                    sg["f1"], sg["value_norm"],
                    color=palette_map[dset], s=60, zorder=3
                )

    # Axes niceties
    g.set_ylabels("Normalized (Before = 100)")
    g.set_xlabels("test_f1_weighted")
    try:
        ncol = g._ncol
    except Exception:
        ncol = 3
    for i, ax in enumerate(g.axes.flatten()):
        if i % ncol != 0:
            ax.set_ylabel("")
        ax.axhline(100.0, linestyle="--", linewidth=0.8, zorder=0)
        ax.set_ylim(bottom=0)  # keep in view

    # Legend (datasets) outside; seaborn also tries to add style legend entries,
    # but FacetGrid merges—this keeps it readable.
    g.add_legend(title="Dataset")
    g.figure.subplots_adjust(right=0.82)
    if getattr(g, "_legend", None) is not None:
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        g._legend.set_loc("center left")
        g._legend.set_frame_on(False)

    g.figure.tight_layout()
    g.figure.suptitle(
        f"All datasets — {metric} vs test_f1_weighted  [freq={DEFAULT_FREQ}, dt={DEFAULT_DT_DEPTH}, {bin_param_text}]",
        y=1.03
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ALL__{metric}_vs_f1.pdf"
    g.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)
    return out_path

def main() -> int:
    init_theme()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None, help="Path to results root (default: ../../experiments/2_grid/results)")
    ap.add_argument("--out",  type=Path, default=Path("final_experiments/analyze/plots"), help="Output directory")
    args = ap.parse_args()

    root = args.root or compute_default_root(__file__)
    out_base = args.out

    # collect all datasets
    df_long, bin_param_text = build_all_long(root)
    if df_long.empty:
        print(f"[WARN] No data found across datasets with defaults freq={DEFAULT_FREQ}, dt={DEFAULT_DT_DEPTH}, threshold=0.01")
        return 1

    out_dir = out_base / "ALL"
    for m in METRICS:
        try:
            p = plot_all_datasets_by_metric(df_long, m, out_dir, bin_param_text)
            print(f"✓ wrote {p}")
        except Exception as e:
            print(f"! ERROR plotting {m} -> {e}")

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
