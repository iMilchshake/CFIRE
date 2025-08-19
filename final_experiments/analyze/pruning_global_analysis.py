from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from final_experiments.analyze.utils import (
    ALL_PARAMS,
    load_csv_files,
    filter_all_params_to_default,
    filter_other_params_to_default,
)

# ---- CONFIG ----
METRICS = ["test_f1_weighted", "val_f1_weighted", "rule_size", "literal_count"]
# Preferred join keys provided by user; we’ll intersect with actual df columns at runtime.
JOIN_KEYS_PREFERRED = ["model_idx", "cfire_config_idx", "cfire_seed", "expl_method", "freq_threshold", "bin_config", "max_dt_depth"]

# Pretty printer (tabulate optional)
def _print_df(df: pd.DataFrame, title: str | None = None) -> None:
    if title:
        print(f"\n{title}")
    from tabulate import tabulate  # type: ignore
    print(tabulate(
        df.reset_index(),
        headers="keys",
        tablefmt="github",
        showindex=False,
        floatfmt=".2f"
    ))

def _relative_delta(pruned: pd.Series, base: pd.Series) -> pd.Series:
    base_abs = base.abs()
    diff = pruned - base
    out = diff.divide(base_abs.replace(0, np.nan))
    return out.replace([np.inf, -np.inf], np.nan)

def _read_three(results_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Return {dataset: {'base': df, 'safe': df, 'best': df}} for datasets having a base metrics.csv."""
    base = load_csv_files(results_dir, "metrics.csv")
    safe = load_csv_files(results_dir, "metrics_safe_prune.csv")
    best = load_csv_files(results_dir, "metrics_best_prune.csv")
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for ds, df in base.items():
        if ds in safe and ds in best:
            out[ds] = {"base": df, "safe": safe[ds], "best": best[ds]}
    return out

def _align_on_keys(base: pd.DataFrame, other: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, int, int]:
    """Inner-join base and other on available JOIN_KEYS_PREFERRED, suffixing metric columns from `other`."""
    keys = [k for k in JOIN_KEYS_PREFERRED if k in base.columns and k in other.columns]
    if not keys:
        raise ValueError(f"[{dataset_name}] No common join keys between base and other. Check CSV columns.")
    # Deduplicate potential duplicate rows before merge (keeps first)
    base_dedup = base.drop_duplicates(subset=keys)
    other_dedup = other.drop_duplicates(subset=keys)

    before_base = len(base_dedup)
    before_other = len(other_dedup)
    merged = pd.merge(
        base_dedup,
        other_dedup[[*keys, *[c for c in other_dedup.columns if c not in keys]]],
        on=keys,
        how="inner",
        suffixes=("", "_pruned"),
        validate="one_to_one",
    )
    matched = len(merged)
    # Coverage metrics: relative to smaller of the two sides
    denom = min(before_base, before_other)
    return merged, matched, denom

def _compute_deltas(merged: pd.DataFrame, variant_suffix: str) -> pd.DataFrame:
    out = merged.copy()
    for m in METRICS:
        base_col = m
        pruned_col = f"{m}_pruned"
        if base_col not in out.columns or pruned_col not in out.columns:
            continue
        out[f"{m}__{variant_suffix}"] = 100.0 * _relative_delta(out[pruned_col], out[base_col])
    return out

def _prepare_delta_rows(df_base: pd.DataFrame, df_safe: pd.DataFrame, df_best: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Tuple[int,int]]]:
    m_safe, matched_s, denom_s = _align_on_keys(df_base, df_safe, dataset_name)
    m_best, matched_b, denom_b = _align_on_keys(df_base, df_best, dataset_name)

    # Compute relative deltas vs base
    dsafe = _compute_deltas(m_safe, "safe")
    dbest = _compute_deltas(m_best, "best")

    # Keep only identifier + delta columns
    id_cols = [c for c in JOIN_KEYS_PREFERRED if c in dsafe.columns]
    delta_cols = [c for c in dsafe.columns if any(c.startswith(f"{m}__safe") for m in METRICS)]
    dsafe = dsafe[id_cols + delta_cols]

    id_cols_best = [c for c in JOIN_KEYS_PREFERRED if c in dbest.columns]
    delta_cols_best = [c for c in dbest.columns if any(c.startswith(f"{m}__best") for m in METRICS)]
    dbest = dbest[id_cols_best + delta_cols_best]

    # Merge safe+best deltas on identifiers
    keys = [k for k in JOIN_KEYS_PREFERRED if k in dsafe.columns and k in dbest.columns]
    d = pd.merge(dsafe, dbest, on=keys, how="outer", validate="one_to_one")

    coverage = {
        "safe": (matched_s, denom_s),
        "best": (matched_b, denom_b),
    }
    return d, coverage

def _maybe_filter_defaults(df: pd.DataFrame, defaults_only: bool) -> pd.DataFrame:
    return filter_all_params_to_default(df) if defaults_only else df

def _avg_over(df: pd.DataFrame, by_cols: List[str]) -> pd.Series:
    # Average only delta columns
    delta_cols = [c for c in df.columns if "__safe" in c or "__best" in c]
    return df[delta_cols].mean(numeric_only=True)

def _format_multiindex_columns() -> pd.MultiIndex:
    tuples = []
    for variant in ["safe", "best"]:
        for m in METRICS:
            tuples.append((variant, m))
    return pd.MultiIndex.from_tuples(tuples, names=["prune", "metric"])

def _collect_A_tables(per_ds_deltas: Dict[str, pd.DataFrame], defaults_only: bool) -> pd.DataFrame:
    rows = {}
    for ds, d in per_ds_deltas.items():
        d_use = _maybe_filter_defaults(d, defaults_only)
        rows[ds] = _avg_over(d_use, by_cols=[])
    out = pd.DataFrame.from_dict(rows, orient="index")
    # Reorder columns to (variant, metric)
    col_order = [f"{m}__safe" for m in METRICS] + [f"{m}__best" for m in METRICS]
    out = out.reindex(columns=col_order)
    out.columns = _format_multiindex_columns()
    out.index.name = "dataset"
    return out

def _collect_B_tables(per_ds_deltas: Dict[str, pd.DataFrame], defaults_only: bool) -> pd.DataFrame:
    # average over datasets and models -> group by expl_method
    frames = []
    for ds, d in per_ds_deltas.items():
        d_use = _maybe_filter_defaults(d, defaults_only)
        d_use = d_use.copy()
        d_use["__dataset"] = ds
        frames.append(d_use)
    all_d = pd.concat(frames, ignore_index=True)
    grp = all_d.groupby("expl_method", dropna=False)
    rows = grp.apply(lambda g: g[[c for c in g.columns if "__safe" in c or "__best" in c]].mean(numeric_only=True))
    rows.index.name = "expl_method"
    # Reorder and relabel columns
    col_order = [f"{m}__safe" for m in METRICS] + [f"{m}__best" for m in METRICS]
    rows = rows.reindex(columns=col_order)
    rows.columns = _format_multiindex_columns()
    return rows

def _collect_C_tables(per_ds_deltas: Dict[str, pd.DataFrame], defaults_only: bool) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    # concat all with dataset labels
    frames = []
    for ds, d in per_ds_deltas.items():
        d_use = _maybe_filter_defaults(d, defaults_only)
        d_use = d_use.copy()
        d_use["__dataset"] = ds
        frames.append(d_use)
    all_d = pd.concat(frames, ignore_index=True)

    for hp in ALL_PARAMS:
        # For C1: fix other params to default; C2: use all rows
        if defaults_only:
            # keep rows where other params are default: we can reuse helper by filtering base-like df
            # Here rows already contain only deltas; we emulate the same by string comparing to defaults via utils
            # Simplest: reconstruct a mask by equality to the value mode per hp? Instead, reuse filter_other_params_to_default:
            # That util expects originals with ALL_PARAMS present; we still have them. Good.
            d_hp = filter_other_params_to_default(all_d, hp)
        else:
            d_hp = all_d

        # group by the selected hyperparameter value (stringified to avoid unhashable objects)
        key_series = d_hp[hp].astype(str)
        grp = d_hp.groupby(key_series, dropna=False)
        table = grp[[c for c in d_hp.columns if "__safe" in c or "__best" in c]].mean(numeric_only=True)
        # order columns
        col_order = [f"{m}__safe" for m in METRICS] + [f"{m}__best" for m in METRICS]
        table = table.reindex(columns=col_order)
        table.index.name = hp
        table.columns = _format_multiindex_columns()
        out[hp] = table
    return out

def main(results_dir: Path) -> None:
    datasets = _read_three(results_dir)
    if not datasets:
        raise FileNotFoundError("No datasets with all three files found.")

    per_ds_deltas: Dict[str, pd.DataFrame] = {}
    coverage_msgs = []

    for ds, trio in datasets.items():
        d, coverage = _prepare_delta_rows(trio["base"], trio["safe"], trio["best"], ds)
        per_ds_deltas[ds] = d
        # coverage reporting
        for variant, (matched, denom) in coverage.items():
            pct = 0.0 if denom == 0 else 100.0 * matched / denom
            coverage_msgs.append(f"[{ds}] join '{variant}': matched={matched}/{denom} ({pct:.1f}%)")

    # ----- A1 / A2 -----
    A1 = _collect_A_tables(per_ds_deltas, defaults_only=True)
    A2 = _collect_A_tables(per_ds_deltas, defaults_only=False)

    # ----- B1 / B2 -----
    B1 = _collect_B_tables(per_ds_deltas, defaults_only=True)
    B2 = _collect_B_tables(per_ds_deltas, defaults_only=False)

    # ----- C1 / C2 -----
    C1 = _collect_C_tables(per_ds_deltas, defaults_only=True)
    C2 = _collect_C_tables(per_ds_deltas, defaults_only=False)

    # ---- OUTPUT ----
    print("\n=== Join coverage (inner joins vs base) ===")
    for msg in coverage_msgs:
        print(msg)

    _print_df(A1, title="\n[A1] Per-dataset mean relative deltas (defaults only)")
    _print_df(A2, title="\n[A2] Per-dataset mean relative deltas (all params)")

    _print_df(B1, title="\n[B1] Per-explainer mean relative deltas (defaults only)")
    _print_df(B2, title="\n[B2] Per-explainer mean relative deltas (all params)")

    print("\n[C1] Hyperparameter-wise mean relative deltas (defaults only)")
    for hp, tbl in C1.items():
        _print_df(tbl, title=f"  • {hp}")

    print("\n[C2] Hyperparameter-wise mean relative deltas (all params)")
    for hp, tbl in C2.items():
        _print_df(tbl, title=f"  • {hp}")

if __name__ == "__main__":
    main(Path("./experiments/2_grid/results/"))
