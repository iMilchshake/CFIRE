from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_FREQ = 0.01
DEFAULT_DT_DEPTH = 7
DEFAULT_BIN_SNIPPET = "threshold=0.01"  # matches "ThresholdBinarization(threshold=0.01)"

DEFAULT_MARKERS = {
    "ig": "o",
    "lime": "^",
    "kernelshap": "s",
}


def find_datasets(results_dir: Path) -> List[str]:
    return sorted([p.name for p in results_dir.iterdir() if p.is_dir()])


def load_enriched_metrics(results_dir: Path, dataset: str) -> pd.DataFrame | None:
    base = results_dir / dataset
    enriched = base / "metrics_with_model_perf.csv"
    if not enriched.exists():
        logging.error("[%s] metrics_with_model_perf.csv not found. Run the augment script first.", dataset)
        return None
    df = pd.read_csv(enriched)
    if "model_test_acc" not in df.columns:
        logging.error("[%s] 'model_test_acc' missing in %s", dataset, enriched)
        return None
    return df


def filter_defaults(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "freq_threshold" in df.columns:
        df["freq_threshold"] = pd.to_numeric(df["freq_threshold"], errors="coerce")
        df = df[df["freq_threshold"] == DEFAULT_FREQ]
    else:
        logging.warning("Missing freq_threshold column; not filtering by it.")

    if "max_dt_depth" in df.columns:
        df["max_dt_depth"] = pd.to_numeric(df["max_dt_depth"], errors="coerce")
        df = df[df["max_dt_depth"] == DEFAULT_DT_DEPTH]
    else:
        logging.warning("Missing max_dt_depth column; not filtering by it.")

    if "bin_config" in df.columns:
        df = df[df["bin_config"].astype(str).str.contains(DEFAULT_BIN_SNIPPET, na=False)]
    else:
        logging.warning("Missing bin_config column; not filtering by it.")

    return df


def main():
    parser = argparse.ArgumentParser(description="Single scatter plot: all datasets (color) vs explainers (marker).")
    parser.add_argument("--grid-root", type=Path, default=Path("./experiments/2_grid"),
                        help="Path to the 2_grid directory (default: ./experiments/2_grid)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output image path (e.g., ./experiments/2_grid/plots/all.png). If omitted, a default is used.")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of datasets to include (default: all under <grid-root>/results)")
    parser.add_argument("--methods", nargs="*", default=["IG", "lime", "kernelshap"],
                        help="Explanation methods to include (default: IG lime kernelshap)")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results_dir = args.grid_root / "results"
    if not results_dir.exists():
        raise SystemExit(f"Results dir not found: {results_dir}")

    datasets = args.datasets or find_datasets(results_dir)
    if not datasets:
        raise SystemExit("No dataset directories found under results/.")

    out_path = args.out or (args.grid_root / "plots" / "all_datasets_model_vs_cfire.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare figure
    fig = plt.figure(figsize=(9, 7), dpi=args.dpi)
    ax = fig.add_subplot(111)

    # Marker per method (normalize to lower-case key)
    method_markers: Dict[str, str] = {}
    for m in args.methods:
        key = str(m).lower()
        method_markers[key] = DEFAULT_MARKERS.get(key, "o")  # fallback to circle

    # We'll gather handles per dataset for the color legend (first point we plot for each dataset)
    dataset_handles: Dict[str, any] = {}

    for dsi, ds in enumerate(datasets):
        df = load_enriched_metrics(results_dir, ds)
        if df is None:
            continue
        df = filter_defaults(df)
        if df.empty:
            logging.info("[%s] no rows after defaults filter; skipping.", ds)
            continue

        # Normalize expl_method to lower for filtering
        df["expl_method_norm"] = df["expl_method"].astype(str).str.lower()

        # For color consistency per dataset, plot each dataset's points in one color by
        # letting matplotlib use a single color for this dataset group. We'll draw in
        # per-method batches but re-use the first handle for the legend.
        first_handle = None

        for m in args.methods:
            mk = method_markers[str(m).lower()]
            sub = df[df["expl_method_norm"] == str(m).lower()]
            if sub.empty:
                continue

            # collapse to one point per (model_idx, method)
            g = sub.groupby(["model_idx"], as_index=False).agg(
                model_test_acc=("model_test_acc", "first"),
                cfire_test_acc=("test_acc", "mean"),
            )
            g = g.dropna(subset=["model_test_acc"])
            if g.empty:
                continue

            # Scatter for this subset. We do NOT set explicit colors, so all scatters for this dataset
            # will share the same implicitly chosen color as long as we consume the color cycle only
            # once per dataset. To ensure consistent color, grab it from the first scatter and reuse.
            if first_handle is None:
                # Make a tiny initial scatter to lock in the color for the dataset.
                first_handle = ax.scatter([], [], label=ds)

            # Extract the facecolor assigned to the placeholder handle
            facecolor = first_handle.get_facecolors()
            if facecolor is None or len(facecolor) == 0:
                # Fallback: plot and then read its color
                tmp = ax.scatter([g["model_test_acc"].iloc[0]], [g["cfire_test_acc"].iloc[0]], marker=mk, s=48, alpha=0.8)
                facecolor = tmp.get_facecolors()
                tmp.remove()

            # Use the determined facecolor for all points of this dataset
            color = facecolor[0] if len(facecolor) > 0 else None

            h = ax.scatter(g["model_test_acc"], g["cfire_test_acc"], marker=mk, s=48, alpha=0.8, c=[color])
            if ds not in dataset_handles:
                dataset_handles[ds] = first_handle or h

    ax.set_xlabel("Black-box model test accuracy")
    ax.set_ylabel("CFIRE test accuracy")
    ax.set_title("All datasets — model vs CFIRE test accuracy\n(defaults: freq 0.01, depth 7, bin threshold=0.01)")
    ax.grid(True, alpha=0.3)

    # Build legends
    # 1) Dataset legend (colors)
    if dataset_handles:
        ds_legend = ax.legend(
            handles=list(dataset_handles.values()),
            labels=list(dataset_handles.keys()),
            title="Datasets",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0.0
        )
        ax.add_artist(ds_legend)

    # 2) Method legend (markers)
    method_handles = []
    method_labels = []
    for m in args.methods:
        mk = method_markers[str(m).lower()]
        handle = ax.scatter([], [], marker=mk, s=48, label=m, c="k")
        method_handles.append(handle)
        method_labels.append(m)

    ax.legend(method_handles, method_labels, title="Explainers", bbox_to_anchor=(1.02, 0.0), loc="lower left", borderaxespad=0.0)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    logging.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
