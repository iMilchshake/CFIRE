"""
augment_metrics_with_model_perf.py

Add the black-box model's test accuracy to each dataset's metrics.csv
as a new column `model_test_acc`, producing `metrics_with_model_perf.csv`.

Intended location: final_experiments/ (next to experiment.py and models.py)

Usage (from repo root):
    python final_experiments/augment_metrics_with_model_perf.py
        --grid-root ./experiments/2_grid
        --data-root ./data/cfire

Optional flags:
    --also-prune        # also enrich metrics_best_prune.csv and metrics_safe_prune.csv
    --overwrite         # overwrite existing *_with_model_perf.csv if present
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# --- Robust import of get_pretrained_models regardless of how the script is invoked ---
def _import_models_module():
    # Try relative import (works if called as `python -m final_experiments.augment_metrics_with_model_perf`)
    try:
        from .models import get_pretrained_models  # type: ignore
        return get_pretrained_models
    except Exception:
        pass
    # Try absolute package import
    try:
        from final_experiments.models import get_pretrained_models  # type: ignore
        return get_pretrained_models
    except Exception:
        pass
    # Fallback: local import (works if this file is placed in final_experiments/ and executed directly)
    from models import get_pretrained_models  # type: ignore
    return get_pretrained_models


get_pretrained_models = _import_models_module()


def build_model_perf_map(data_root: Path) -> Dict[int, float]:
    """
    For a given dataset directory (data_root / <dataset>), load all pretrained models
    and build a mapping: model_idx -> test_acc.
    NOTE: This does NOT slice by top-N; it returns all available models.
    """
    # get_pretrained_models expects the path to the dataset folder under ./data/cfire/<dataset>
    models, _X_val, _X_test = get_pretrained_models(data_root)
    model_perf_map = {m.model_idx: float(m.test_acc) for m in models}
    return model_perf_map


def enrich_one_csv(csv_path: Path, model_perf_map: Dict[int, float], overwrite: bool) -> Path | None:
    """
    Read a metrics-like CSV, add `model_test_acc` by mapping model_idx, and write
    a sibling file with `_with_model_perf` suffix.
    """
    if not csv_path.exists():
        logging.warning("missing file, skipping: %s", csv_path)
        return None

    out_path = csv_path.with_name(csv_path.stem + "_with_model_perf.csv")

    if out_path.exists() and not overwrite:
        logging.info("exists, skipping (use --overwrite to replace): %s", out_path)
        return out_path

    df = pd.read_csv(csv_path)
    if "model_idx" not in df.columns:
        logging.warning("no 'model_idx' column in %s — skipping.", csv_path)
        return None

    df["model_test_acc"] = df["model_idx"].map(model_perf_map)
    # Optional sanity check: warn if any model_idx could not be mapped
    n_missing = int(df["model_test_acc"].isna().sum())
    if n_missing > 0:
        missing = sorted(set(df.loc[df["model_test_acc"].isna(), "model_idx"].tolist()))
        logging.warning(
            "Unmapped model_idx in %s (no test_acc found for %s). They will be NaN.",
            csv_path.name, missing,
        )

    df.to_csv(out_path, index=False)
    logging.info("wrote %s", out_path)
    return out_path


def process_dataset(results_dir: Path, dataset: str, data_root: Path, also_prune: bool, overwrite: bool) -> Tuple[str, List[Path]]:
    """
    Enrich metrics files for one dataset and return the list of output paths.
    """
    dataset_results = results_dir / dataset
    if not dataset_results.exists():
        logging.warning("results directory missing for dataset %s", dataset)
        return dataset, []

    # Build model_perf_map for this dataset
    model_perf_map = build_model_perf_map(data_root / dataset)

    outputs: List[Path] = []
    outputs.append(enrich_one_csv(dataset_results / "metrics.csv", model_perf_map, overwrite))

    if also_prune:
        outputs.append(enrich_one_csv(dataset_results / "metrics_best_prune.csv", model_perf_map, overwrite))
        outputs.append(enrich_one_csv(dataset_results / "metrics_safe_prune.csv", model_perf_map, overwrite))

    return dataset, [p for p in outputs if p is not None]


def main():
    parser = argparse.ArgumentParser(description="Enrich metrics.csv with model test accuracy.")
    parser.add_argument("--grid-root", type=Path, default=Path("./experiments/2_grid"),
                        help="Path to the 2_grid directory (default: ./experiments/2_grid)")
    parser.add_argument("--data-root", type=Path, default=Path("./data/cfire"),
                        help="Path to the root of pretrained models (default: ./data/cfire)")
    parser.add_argument("--also-prune", action="store_true",
                        help="Also enrich metrics_best_prune.csv and metrics_safe_prune.csv")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing *_with_model_perf.csv files")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of datasets to process (default: all subdirs of <grid-root>/results)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results_dir = args.grid_root / "results"
    if not results_dir.exists():
        raise SystemExit(f"Results dir not found: {results_dir}")

    datasets = args.datasets
    if not datasets:
        # auto-discover datasets by subdirectories that contain metrics.csv
        datasets = sorted([p.name for p in results_dir.iterdir() if p.is_dir()])

    all_outputs: Dict[str, List[Path]] = {}
    for ds in datasets:
        dataset, outputs = process_dataset(results_dir, ds, args.data_root, args.also_prune, args.overwrite)
        all_outputs[dataset] = outputs

    # Summary
    print("\n=== Summary ===")
    for ds in sorted(all_outputs):
        outs = all_outputs[ds]
        if outs:
            for p in outs:
                print(f"[{ds}] -> {p}")
        else:
            print(f"[{ds}] -> (no files written)")

if __name__ == "__main__":
    main()
