from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd


PAIR_SPECS = [
    # (left_csv, right_csv, output_csv)
    ("metrics_with_model_perf.csv", "rule_log.csv", "unified_before_pruning_metrics.csv"),
    ("metrics_safe_prune.csv",      "rule_log_safe_prune.csv", "unified_after_safe_pruning_metrics.csv"),
    ("metrics_best_prune.csv",      "rule_log_best_prune.csv", "unified_after_best_pruning_metrics.csv"),
]

# Canonical join keys we expect across files.
REQUIRED_KEYS = [
    "model_idx", "cfire_config_idx", "cfire_seed",
    "expl_method", "freq_threshold", "bin_config", "max_dt_depth",
]


def choose_join_keys(df_left: pd.DataFrame, df_right: pd.DataFrame) -> List[str]:
    """Return the list of keys to join on (intersection of REQUIRED_KEYS present in both frames).
    If fewer than 2 keys are available, fall back to the full intersection of equal-named columns.
    """
    inter = [k for k in REQUIRED_KEYS if k in df_left.columns and k in df_right.columns]
    if len(inter) >= 2:
        return inter
    # Fallback: any shared columns
    fallback = [c for c in df_left.columns if c in df_right.columns]
    if len(fallback) == 0:
        raise ValueError("No shared columns between CSVs to join on.")
    return fallback


def merge_pair(left_path: Path, right_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    left_df  = pd.read_csv(left_path)
    right_df = pd.read_csv(right_path)
    keys = choose_join_keys(left_df, right_df)
    merged = left_df.merge(right_df, on=keys, how="inner", suffixes=("_metrics", "_rules"))
    # Reorder: keys first, then rest
    key_cols = [c for c in keys if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in key_cols]
    merged = merged[key_cols + other_cols]
    return merged, keys


def process_dataset(ds_dir: Path) -> None:
    print(f"\n== Dataset: {ds_dir.name}")
    for left_name, right_name, out_name in PAIR_SPECS:
        left = ds_dir / left_name
        right = ds_dir / right_name
        if not left.exists() or not right.exists():
            missing = []
            if not left.exists(): missing.append(left_name)
            if not right.exists(): missing.append(right_name)
            print(f"   - SKIP {out_name} (missing: {', '.join(missing)})")
            continue
        try:
            merged, keys = merge_pair(left, right)
        except Exception as e:
            print(f"   ! ERROR merging {left_name} × {right_name}: {e}")
            continue
        out_path = ds_dir / out_name
        merged.to_csv(out_path, index=False)
        print(f"   ✓ Wrote {out_name}  (rows={len(merged)}, join_keys={keys})")


def find_datasets(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])


def compute_default_root() -> Path:
    """Compute ../../experiments/2_grid/results relative to this script.
    Layout assumption:
        <repo-root>/final_experiments/analyze/merge_unified_metrics.py
        <repo-root>/experiments/2_grid/results/
    """
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        # Fallback if __file__ is not defined (rare), use CWD
        script_dir = Path.cwd()
    repo_root = script_dir.parent.parent  # ../../
    default_root = repo_root / "experiments" / "2_grid" / "results"
    return default_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None,
                    help="Path to the results root (default: ../../experiments/2_grid/results relative to this script)")
    args = ap.parse_args()

    root: Path = args.root or compute_default_root()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Results root not found or not a directory: {root}", file=sys.stderr)
        print("        Pass --root to override the default.", file=sys.stderr)
        return 2

    datasets = find_datasets(root)
    if not datasets:
        print(f"[WARN] No dataset subfolders found under {root}", file=sys.stderr)
        return 1

    for ds in datasets:
        process_dataset(ds)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
