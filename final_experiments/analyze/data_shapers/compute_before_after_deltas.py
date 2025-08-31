#!/usr/bin/env python3
from __future__ import annotations
"""
Build per-dataset files that combine BEFORE + AFTER (safe/best) values
for every numeric metric, and compute deltas to BEFORE.

Inputs (per dataset):
  - unified_before_pruning_metrics.csv
  - unified_after_safe_pruning_metrics.csv
  - unified_after_best_pruning_metrics.csv

Output:
  - unified_before_after_deltas.csv

Column naming:
  For each metric M, the output includes:
    M_before, M_after_safe, M_after_best, delta M_safe, delta M_best
  where deltas are (after - before).

Usage:
  python final_experiments/analyze/compute_before_after_deltas.py
  # or override root
  python final_experiments/analyze/compute_before_after_deltas.py --root path/to/experiments/2_grid/results
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

PREFERRED_KEYS = [
    "model_idx", "cfire_config_idx", "cfire_seed",
    "expl_method", "freq_threshold", "bin_config", "max_dt_depth",
]

@dataclass
class Inputs:
    before: pd.DataFrame
    after_safe: pd.DataFrame
    after_best: pd.DataFrame

def compute_default_root(script_file: str) -> Path:
    script_dir = Path(script_file).resolve().parent
    repo_root = script_dir.parent.parent.parent  # ../../
    return repo_root / "experiments" / "2_grid" / "results"

def read_inputs(ds_dir: Path) -> Inputs:
    return Inputs(
        before=pd.read_csv(ds_dir / "unified_before_pruning_metrics.csv"),
        after_safe=pd.read_csv(ds_dir / "unified_after_safe_pruning_metrics.csv"),
        after_best=pd.read_csv(ds_dir / "unified_after_best_pruning_metrics.csv"),
    )

def choose_join_keys(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> List[str]:
    pref = [k for k in PREFERRED_KEYS if k in a.columns and k in b.columns and k in c.columns]
    if len(pref) >= 2:
        return pref
    shared = [col for col in a.columns if col in b.columns and col in c.columns]
    if len(shared) < 2:
        raise ValueError("Not enough shared columns across inputs to join on (need >=2).");
    return shared

def numeric_common_metrics(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, keys: List[str]) -> List[str]:
    num_a = set(a.select_dtypes(include="number").columns)
    num_b = set(b.select_dtypes(include="number").columns)
    num_c = set(c.select_dtypes(include="number").columns)
    return sorted((num_a & num_b & num_c) - set(keys))

def merge_all(inputs: Inputs, keys: List[str]) -> pd.DataFrame:
    merged = inputs.before.merge(inputs.after_safe, on=keys, how="inner", suffixes=("_before", "_after_safe"))
    tmp_best = inputs.after_best.copy()
    best_renamed = {col: (f"{col}_after_best" if col not in keys else col) for col in tmp_best.columns}
    tmp_best = tmp_best.rename(columns=best_renamed)
    merged = merged.merge(tmp_best, on=keys, how="inner")
    return merged

def build_output(merged: pd.DataFrame, metrics: List[str], keys: List[str]) -> pd.DataFrame:
    # Start with the join keys as the left-most columns
    base = merged[keys].copy()
    frames = [base]          # pieces to concatenate once at the end
    col_order = list(keys)   # final column order

    for m in metrics:
        b, s, bt = f"{m}_before", f"{m}_after_safe", f"{m}_after_best"
        # fallbacks (usually unnecessary, but keep your original semantics)
        if b not in merged.columns:  b = m
        if s not in merged.columns:  s = m
        if bt not in merged.columns: bt = m

        # Build a small frame for this metric in the desired order.
        # Use .to_numpy() to avoid extra copies and keep it fast.
        df_m = pd.DataFrame({
            b: merged[b].to_numpy(),
            s: merged[s].to_numpy(),
            bt: merged[bt].to_numpy(),
            f"delta {m}_safe": (merged[s] - merged[b]).to_numpy(),
            f"delta {m}_best": (merged[bt] - merged[b]).to_numpy(),
        }, index=merged.index)

        frames.append(df_m)
        col_order.extend([b, s, bt, f"delta {m}_safe", f"delta {m}_best"])

    # Concatenate once → no fragmentation
    out = pd.concat(frames, axis=1)

    # Ensure the exact column order and (optionally) defragment the result
    out = out[col_order].copy()
    return out

def process_dataset(ds_dir: Path):
    inputs = read_inputs(ds_dir)
    keys = choose_join_keys(inputs.before, inputs.after_safe, inputs.after_best)
    metrics = numeric_common_metrics(inputs.before, inputs.after_safe, inputs.after_best, keys)
    merged = merge_all(inputs, keys)
    output = build_output(merged, metrics, keys)
    out_path = ds_dir / "unified_before_after_deltas.csv"
    output.to_csv(out_path, index=False)
    return out_path, metrics, keys

def find_datasets(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None,
                    help="Path to results root (default: ../../experiments/2_grid/results relative to this script)")
    args = ap.parse_args()

    root = args.root or compute_default_root(__file__)
    if not root.exists():
        print(f"[ERROR] Results root not found: {root}", file=sys.stderr)
        return 2

    datasets = find_datasets(root)
    if not datasets:
        print(f"[WARN] No dataset subfolders in {root}", file=sys.stderr)
        return 1

    for ds in datasets:
        try:
            out_path, metrics, keys = process_dataset(ds)
            print(f"✓ {ds.name}: wrote {out_path.name} | join_keys={keys} | #metrics={len(metrics)}")
            if metrics:
                print("  metrics:", metrics)
        except FileNotFoundError as e:
            print(f"- {ds.name}: SKIP (missing input) -> {e}")
        except Exception as e:
            print(f"! {ds.name}: ERROR -> {e}")

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
