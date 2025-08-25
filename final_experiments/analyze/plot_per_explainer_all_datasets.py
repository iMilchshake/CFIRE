from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_FREQ = 0.01
DEFAULT_DT_DEPTH = 7
DEFAULT_BIN_SNIPPET = "threshold=0.01"  # matches "ThresholdBinarization(threshold=0.01)"


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


def make_plot_for_method(results_dir: Path, datasets: List[str], method: str, outdir: Path, dpi: int = 150) -> Path | None:
    method_norm = method.lower()

    # Prepare figure
    fig = plt.figure(figsize=(9, 7), dpi=dpi)
    ax = fig.add_subplot(111)

    dataset_handles = {}

    # IMPORTANT for consistent colors across the three figures:
    # iterate datasets in the SAME ORDER for every method. We rely on the default
    # matplotlib color cycle and do not set explicit colors.
    for ds in datasets:
        df = load_enriched_metrics(results_dir, ds)
        if df is None:
            continue
        df = filter_defaults(df)
        if df.empty:
            logging.info("[%s] no rows after defaults filter; skipping.", ds)
            continue

        df["expl_method_norm"] = df["expl_method"].astype(str).str.lower()
        sub = df[df["expl_method_norm"] == method_norm]
        if sub.empty:
            continue

        # Collapse to one row per model_idx
        g = sub.groupby("model_idx", as_index=False).agg(
            model_test_acc=("model_test_acc", "first"),
            cfire_test_acc=("test_acc", "mean"),
        ).dropna(subset=["model_test_acc"])

        if g.empty:
            continue

        # One scatter call per dataset to ensure one color per dataset
        h = ax.scatter(g["model_test_acc"], g["cfire_test_acc"], label=ds, s=48, alpha=0.85)
        if ds not in dataset_handles:
            dataset_handles[ds] = h

    if not dataset_handles:
        logging.info("[method=%s] nothing to plot.", method)
        plt.close(fig)
        return None

    ax.set_xlabel("Black-box model test accuracy")
    ax.set_ylabel("CFIRE test accuracy")
    ax.set_title(f"{method} — All datasets (defaults: freq 0.01, depth 7, bin threshold=0.01)")
    ax.grid(True, alpha=0.3)

    # Dataset legend (colors)
    ds_legend = ax.legend(
        handles=list(dataset_handles.values()),
        labels=list(dataset_handles.keys()),
        title="Datasets",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0
    )
    ax.add_artist(ds_legend)

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{method}_all_datasets_model_vs_cfire.png"
    ds_legend = ax.legend(
        handles=list(dataset_handles.values()),
        labels=list(dataset_handles.keys()),
        title="Datasets",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0
    )
    ax.add_artist(ds_legend)

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{method}_all_datasets_model_vs_cfire.png"

    fig.savefig(out_path, bbox_inches="tight", bbox_extra_artists=(ds_legend,))
    plt.close(fig)
    plt.close(fig)

    logging.info("Wrote %s", out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Three separate diagrams (one per explainer), all datasets in each.")
    parser.add_argument("--grid-root", type=Path, default=Path("./experiments/2_grid"),
                        help="Path to the 2_grid directory (default: ./experiments/2_grid)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Directory to write plots (default: <grid-root>/plots)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of datasets to include (default: all under <grid-root>/results)")
    parser.add_argument("--methods", nargs="*", default=["IG", "lime", "kernelshap"],
                        help="Explainers to plot (default: IG lime kernelshap)")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results_dir = args.grid_root / "results"
    if not results_dir.exists():
        raise SystemExit(f"Results dir not found: {results_dir}")

    outdir = args.outdir or (args.grid_root / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or find_datasets(results_dir)
    datasets = sorted(datasets)  # fixed order = consistent color mapping across the three plots
    if not datasets:
        raise SystemExit("No dataset directories found under results/.")

    for m in args.methods:
        make_plot_for_method(results_dir, datasets, m, outdir, dpi=args.dpi)


if __name__ == "__main__":
    main()
