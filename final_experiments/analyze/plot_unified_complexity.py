from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import re

DEFAULT_FREQ = 0.01
DEFAULT_DT_DEPTH = 7
DEFAULT_BIN_CONFIG_STR = "ThresholdBinarization(threshold=0.01)"

METRICS = [
    "rule_count",
    "rule_size",
    "literal_count",
    "mean_nodes_per_sample",
    "avg_rules_matched_per_sample",
]

def parse_bin_param(bin_cfg: str) -> str:
    """Return content inside parentheses, e.g.:
    'ThresholdBinarization(threshold=0.01)' -> 'threshold=0.01'
    'TopKBinarization(k=2)' -> 'k=2'
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

def to_long_normalized(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    for m in METRICS:
        b, s, bt = f"{m}_before", f"{m}_after_safe", f"{m}_after_best"
        if not all(c in df.columns for c in (b, s, bt)):
            continue
        sub = df[["model_idx", "expl_method", "bin_config", b, s, bt]].copy()
        base = sub[b].replace({0.0: np.nan})
        norm_s = 100.0 * (sub[s] / base)
        norm_bt = 100.0 * (sub[bt] / base)
        rows.append(pd.DataFrame({
            "dataset": dataset_name,
            "metric": m,
            "model_idx": sub["model_idx"],
            "expl_method": sub["expl_method"],
            "variant": "Before",
            "value_norm": 100.0,
            "bin_param": sub["bin_config"].map(parse_bin_param),
        }))
        rows.append(pd.DataFrame({
            "dataset": dataset_name,
            "metric": m,
            "model_idx": sub["model_idx"],
            "expl_method": sub["expl_method"],
            "variant": "After (Safe)",
            "value_norm": norm_s,
            "bin_param": sub["bin_config"].map(parse_bin_param),
        }))
        rows.append(pd.DataFrame({
            "dataset": dataset_name,
            "metric": m,
            "model_idx": sub["model_idx"],
            "expl_method": sub["expl_method"],
            "variant": "After (Best)",
            "value_norm": norm_bt,
            "bin_param": sub["bin_config"].map(parse_bin_param),
        }))
    if not rows:
        return pd.DataFrame(columns=["dataset","metric","model_idx","expl_method","variant","value_norm","bin_param"])
    long_df = pd.concat(rows, ignore_index=True).dropna(subset=["value_norm"])
    long_df["variant"] = pd.Categorical(long_df["variant"], categories=["Before","After (Safe)","After (Best)"], ordered=True)
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
        long_df = to_long_normalized(dff, dataset_name=ds.name)
        if not long_df.empty:
            frames.append(long_df)
    if not frames:
        return pd.DataFrame(columns=["dataset","metric","model_idx","expl_method","variant","value_norm","bin_param"]), (bin_param_text or "threshold=0.01")
    return pd.concat(frames, ignore_index=True), (bin_param_text or "threshold=0.01")

def plot_all_datasets_by_metric(df_long: pd.DataFrame, metric: str, out_dir: Path, bin_param_text: str) -> Path:
    data = df_long[df_long["metric"] == metric]
    if data.empty:
        raise ValueError(f"No data available for metric '{metric}' across datasets.")
    sns.set_theme()
    g = sns.FacetGrid(data, col="expl_method", hue="dataset", col_wrap=3, sharey=True, height=4.0, palette="deep")

    # Layer 1: faint per-model lines
    g.map_dataframe(
        sns.lineplot,
        x="variant",
        y="value_norm",
        units="model_idx",
        estimator=None,
        linewidth=0.8,
        alpha=0.25,
    )

    # Layer 2: bold mean + CI per dataset color
    g.map_dataframe(
        sns.lineplot,
        x="variant",
        y="value_norm",
        errorbar="ci",
        linewidth=2.2,
        marker="o",
    )
    g.set_ylabels("Normalized (Before = 100)")

    # Keep y-axis label only on the left-most column of facets
    try:
        ncol = g._ncol
    except Exception:
        ncol = 3
    for i, ax in enumerate(g.axes.flatten()):
        if i % ncol != 0:
            ax.set_ylabel("")
        ax.axhline(100.0, linestyle="--", linewidth=0.8)

    # Legend outside (to the right)
    g.add_legend(title="Dataset")
    g.figure.subplots_adjust(right=0.82)
    if getattr(g, "_legend", None) is not None:
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        g._legend.set_loc("center left")
        g._legend.set_frame_on(False)

    g.figure.tight_layout()
    g.figure.suptitle(
        f"All datasets — {metric}  [freq={DEFAULT_FREQ}, dt={DEFAULT_DT_DEPTH}, {bin_param_text}]",
        y=1.03
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ALL__{metric}_normalized.png"
    g.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)
    return out_path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None, help="Path to results root (default: ../../experiments/2_grid/results)")
    ap.add_argument("--out", type=Path, default=Path("final_experiments/analyze/plots"), help="Output directory for figures")
    args = ap.parse_args()

    root = args.root or compute_default_root(__file__)
    out_base = args.out

    df_long, bin_param_text = build_all_long(root)
    if df_long.empty:
        print(f"[WARN] No data found across datasets with defaults freq={DEFAULT_FREQ}, dt={DEFAULT_DT_DEPTH}, threshold=0.01")
        return 1

    out_dir = out_base / "ALL"
    for m in METRICS:
        try:
            out_path = plot_all_datasets_by_metric(df_long, metric=m, out_dir=out_dir, bin_param_text=bin_param_text)
            print(f"✓ wrote {out_path}")
        except Exception as e:
            print(f"! ERROR plotting {m} -> {e}")

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
