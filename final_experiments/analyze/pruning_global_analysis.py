from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from final_experiments.analyze.utils import (
    ALL_PARAMS,
    load_csv_files,
    filter_all_params_to_default,
    filter_other_params_to_default,
)

# ---- CONFIG ----
METRICS = ["val_acc", "test_acc", "test_f1_weighted", "val_f1_weighted", "rule_size", "literal_count"] #, "max_iou", "mean_iou"]
JOIN_CSV_KEYS = [
    "model_idx",
    "cfire_config_idx",
    "cfire_seed",
    "expl_method",
    "freq_threshold",
    "bin_config",
    "max_dt_depth",
]

def _format_percent_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where all numeric cells are formatted as 'xx.xx%'; NaN → '—'."""
    out = df.copy()
    for col in out.columns:
        if is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}%")
    return out

def _fmt_mean_std_series(mean_s: pd.Series, std_s: pd.Series) -> pd.Series:
    """Return strings 'xx.xx% ± yy.yy%' per index; NaN → '—'."""
    out = []
    for k in mean_s.index:
        m = mean_s[k]
        s = std_s[k] if k in std_s.index else np.nan
        out.append("—" if pd.isna(m) else f"{m:.2f}% ± {s:.2f}%")
    return pd.Series(out, index=mean_s.index)

def _print_df(df: pd.DataFrame, title: str | None = None) -> None:
    if title:
        print(f"\n{title}")
    from tabulate import tabulate  # type: ignore
    to_show = _format_percent_df_for_display(df).reset_index()
    print(tabulate(
        to_show,
        headers="keys",
        tablefmt="github",
        showindex=False,
    ))


def _print_df_abs(df, title=None):
    if title: print(f"\n{title}")
    from tabulate import tabulate
    print(tabulate(df.reset_index(), headers="keys", tablefmt="github", showindex=False))

def _collect_abs_change(datasets: Dict[str, Dict[str, pd.DataFrame]], defaults_only: bool) -> pd.DataFrame:
    """Absolute change base→safe / base→best (mean ± std)."""
    rows = {}
    for ds, trio in datasets.items():
        df_base = _maybe_filter_defaults(trio["base"], defaults_only)
        row = {}
        for variant in ["safe", "best"]:
            df_var = _maybe_filter_defaults(trio[variant], defaults_only)
            # align on common rows by index length (assumes matched CSVs)
            common_cols = [m for m in METRICS if m in df_base.columns and m in df_var.columns]
            diffs = df_var[common_cols] - df_base[common_cols]
            mean_s, std_s = diffs.mean(numeric_only=True), diffs.std(numeric_only=True)
            row.update({(variant, m): v for m, v in _fmt_mean_std_series_abs(mean_s, std_s).items()})
        rows[ds] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["variant", "metric"])
    out.index.name = "dataset"
    return out.T

def _fmt_mean_std_series_abs(mean_s, std_s):
    return pd.Series(
        ["—" if pd.isna(mean_s[k]) else f"{mean_s[k]:.2f} ± {std_s.get(k, np.nan):.2f}" for k in mean_s.index],
        index=mean_s.index,
    )

def _read_three(results_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Return {dataset: {'base': df, 'safe': df, 'best': df}} for datasets with all three files."""
    base = load_csv_files(results_dir, "metrics.csv")
    safe = load_csv_files(results_dir, "metrics_safe_prune.csv")
    best = load_csv_files(results_dir, "metrics_best_prune.csv")
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for ds, df in base.items():
        if ds in safe and ds in best:
            out[ds] = {"base": df, "safe": safe[ds], "best": best[ds]}
    return out

def _relative_delta(pruned: pd.Series, base: pd.Series) -> pd.Series:
    """Return percentage change in % (not fraction)."""
    base_abs = base.abs()
    diff = pruned - base
    out = diff.divide(base_abs.replace(0, np.nan))
    out = out.replace([np.inf, -np.inf], np.nan)
    return 100.0 * out  # in percent

def _align_on_keys(base: pd.DataFrame, other: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, int, int]:
    """Inner-join base and other on available JOIN_KEYS_PREFERRED, suffix metrics from `other` with '_pruned'."""
    keys = [k for k in JOIN_CSV_KEYS if k in base.columns and k in other.columns]
    if not keys:
        raise ValueError(f"[{dataset_name}] No common join keys between base and other. Check CSV columns.")
    base_dedup = base.drop_duplicates(subset=keys)
    other_dedup = other.drop_duplicates(subset=keys)

    before_base = len(base_dedup)
    before_other = len(other_dedup)

    other_keep = [c for c in other_dedup.columns if c not in keys]
    merged = pd.merge(
        base_dedup,
        other_dedup[keys + other_keep],
        on=keys,
        how="inner",
        suffixes=("", "_pruned"),
        validate="one_to_one",
    )
    matched = len(merged)
    denom = min(before_base, before_other)
    return merged, matched, denom

def _compute_deltas(merged: pd.DataFrame, variant_suffix: str) -> pd.DataFrame:
    """Compute % deltas for METRICS; output columns named f'{metric}__{variant}'."""
    out = merged.copy()
    for m in METRICS:
        base_col = m
        pruned_col = f"{m}_pruned"
        if base_col not in out.columns or pruned_col not in out.columns:
            continue
        out[f"{m}__{variant_suffix}"] = _relative_delta(out[pruned_col], out[base_col])
    return out

def _prepare_delta_rows(
        df_base: pd.DataFrame, df_safe: pd.DataFrame, df_best: pd.DataFrame, dataset_name: str
) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]]]:
    m_safe, matched_s, denom_s = _align_on_keys(df_base, df_safe, dataset_name)
    m_best, matched_b, denom_b = _align_on_keys(df_base, df_best, dataset_name)

    # % relative deltas vs base
    dsafe = _compute_deltas(m_safe, "safe")
    dbest = _compute_deltas(m_best, "best")

    # Keep only identifiers + delta columns (no duplicate non-key cols)
    id_cols_safe = [c for c in JOIN_CSV_KEYS if c in dsafe.columns]
    delta_cols_safe = [c for c in dsafe.columns if any(c == f"{m}__safe" for m in METRICS)]
    dsafe = dsafe[id_cols_safe + delta_cols_safe]

    id_cols_best = [c for c in JOIN_CSV_KEYS if c in dbest.columns]
    delta_cols_best = [c for c in dbest.columns if any(c == f"{m}__best" for m in METRICS)]
    dbest = dbest[id_cols_best + delta_cols_best]

    # Merge safe+best deltas on common identifier columns (prevents duplicate label errors)
    merge_keys = [k for k in JOIN_CSV_KEYS if k in dsafe.columns and k in dbest.columns]
    d = pd.merge(dsafe, dbest, on=merge_keys, how="inner", validate="one_to_one")

    coverage = {"safe": (matched_s, denom_s), "best": (matched_b, denom_b)}
    return d, coverage

def _maybe_filter_defaults(df: pd.DataFrame, defaults_only: bool) -> pd.DataFrame:
    return filter_all_params_to_default(df) if defaults_only else df

def _avg_over(df: pd.DataFrame) -> pd.Series:
    delta_cols = [c for c in df.columns if "__safe" in c or "__best" in c]
    return df[delta_cols].mean(numeric_only=True)

def _collect_A_tables(per_ds_deltas: Dict[str, pd.DataFrame], defaults_only: bool) -> pd.DataFrame:
    """A1/A2: per-dataset; average over models & explainers, show mean ± std."""
    rows = {}
    for ds, d in per_ds_deltas.items():
        d_use = _maybe_filter_defaults(d, defaults_only)
        delta_cols = [c for c in d_use.columns if "__safe" in c or "__best" in c]
        mean_s = d_use[delta_cols].mean(numeric_only=True)
        std_s  = d_use[delta_cols].std(numeric_only=True)
        rows[ds] = _fmt_mean_std_series(mean_s, std_s)
    out = pd.DataFrame.from_dict(rows, orient="index")

    col_order = [f"{m}__safe" for m in METRICS] + [f"{m}__best" for m in METRICS]
    out = out.reindex(columns=col_order)

    tuples = []
    for variant in ["safe", "best"]:
        for m in METRICS:
            tuples.append((variant, m))
    out.columns = pd.MultiIndex.from_tuples(tuples, names=["prune", "metric"])
    out.index.name = "dataset"
    return out.T

def _collect_B_tables(per_ds_deltas: Dict[str, pd.DataFrame], defaults_only: bool) -> pd.DataFrame:
    """B1/B2: per explainer; avg over datasets & models, show mean ± std."""
    frames = []
    for ds, d in per_ds_deltas.items():
        d_use = _maybe_filter_defaults(d, defaults_only)
        frames.append(d_use.assign(__dataset=ds))
    all_d = pd.concat(frames, ignore_index=True)

    delta_cols = [c for c in all_d.columns if "__safe" in c or "__best" in c]
    grp = all_d.groupby("expl_method", dropna=False)
    means = grp[delta_cols].mean(numeric_only=True)
    stds  = grp[delta_cols].std(numeric_only=True)

    # merge into "mean ± std" strings
    rows = means.copy()
    for col in delta_cols:
        rows[col] = _fmt_mean_std_series(means[col], stds[col])

    rows.index.name = "expl_method"
    col_order = [f"{m}__safe" for m in METRICS] + [f"{m}__best" for m in METRICS]
    rows = rows.reindex(columns=col_order)

    tuples = []
    for variant in ["safe", "best"]:
        for m in METRICS:
            tuples.append((variant, m))
    rows.columns = pd.MultiIndex.from_tuples(tuples, names=["prune", "metric"])
    return rows.T

def _collect_C1_fixed_others(per_ds_deltas: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    C1: For each hyperparam hp:
      - FIX all other params to defaults
      - average over datasets/models/explainers
    """
    all_d = pd.concat([d.assign(__dataset=ds) for ds, d in per_ds_deltas.items()], ignore_index=True)
    out: Dict[str, pd.DataFrame] = {}
    for hp in ALL_PARAMS:
        d_hp = filter_other_params_to_default(all_d, hp)  # leaves hp free
        key = d_hp[hp].astype(str)
        grp = d_hp.groupby(key, dropna=False)

        delta_cols = [c for c in d_hp.columns if "__safe" in c or "__best" in c]
        means = grp[delta_cols].mean(numeric_only=True)
        stds  = grp[delta_cols].std(numeric_only=True)

        tbl = means.copy()
        for col in delta_cols:
            tbl[col] = _fmt_mean_std_series(means[col], stds[col])

        col_order = [f"{m}__safe" for m in METRICS] + [f"{m}__best" for m in METRICS]
        tbl = tbl.reindex(columns=col_order)

        tuples = []
        for variant in ["safe", "best"]:
            for m in METRICS:
                tuples.append((variant, m))
        tbl.columns = pd.MultiIndex.from_tuples(tuples, names=["prune", "metric"])
        tbl.index.name = hp
        out[hp] = tbl.T
    return out

def _collect_C2_avg_others(per_ds_deltas: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    C2: For each hyperparam hp:
      - DO NOT fix others
      - group by hp; average across all other params (and datasets/models/explainers)
    """
    all_d = pd.concat([d.assign(__dataset=ds) for ds, d in per_ds_deltas.items()], ignore_index=True)
    out: Dict[str, pd.DataFrame] = {}
    for hp in ALL_PARAMS:
        key = all_d[hp].astype(str)
        grp = all_d.groupby(key, dropna=False)

        delta_cols = [c for c in all_d.columns if "__safe" in c or "__best" in c]
        means = grp[delta_cols].mean(numeric_only=True)
        stds  = grp[delta_cols].std(numeric_only=True)

        tbl = means.copy()
        for col in delta_cols:
            tbl[col] = _fmt_mean_std_series(means[col], stds[col])

        col_order = [f"{m}__safe" for m in METRICS] + [f"{m}__best" for m in METRICS]
        tbl = tbl.reindex(columns=col_order)

        tuples = []
        for variant in ["safe", "best"]:
            for m in METRICS:
                tuples.append((variant, m))
        tbl.columns = pd.MultiIndex.from_tuples(tuples, names=["prune", "metric"])
        tbl.index.name = hp
        out[hp] = tbl.T
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
        for variant, (matched, denom) in coverage.items():
            pct = 0.0 if denom == 0 else 100.0 * matched / denom
            coverage_msgs.append(f"[{ds}] join '{variant}': matched={matched}/{denom} ({pct:.1f}%)")

    # A1 / A2
    A1 = _collect_A_tables(per_ds_deltas, defaults_only=True)
    A2 = _collect_A_tables(per_ds_deltas, defaults_only=False)

    # B1 / B2
    B1 = _collect_B_tables(per_ds_deltas, defaults_only=True)
    B2 = _collect_B_tables(per_ds_deltas, defaults_only=False)

    # C1 / C2 (new, correct semantics)
    C1 = _collect_C1_fixed_others(per_ds_deltas)   # vary hp, fix others to defaults
    C2 = _collect_C2_avg_others(per_ds_deltas)     # vary hp, average all others

    # ---- OUTPUT ----
    print("\n=== Join coverage (inner joins vs base) ===")
    for msg in coverage_msgs:
        print(msg)

    _print_df(A1, title="\n[A1] Per-dataset mean relative deltas (defaults only)")
    _print_df(A2, title="\n[A2] Per-dataset mean relative deltas (all params)")

    A1_abschg = _collect_abs_change(datasets, defaults_only=True)
    A2_abschg = _collect_abs_change(datasets, defaults_only=False)
    _print_df_abs(A1_abschg, title="\n[A1-abschg] Absolute change base→safe/best (defaults only)")
    _print_df_abs(A2_abschg, title="\n[A2-abschg] Absolute change base→safe/best (all params)")

    _print_df(B1, title="\n[B1] Per-explainer mean relative deltas (defaults only)")
    _print_df(B2, title="\n[B2] Per-explainer mean relative deltas (all params)")

    print("\n[C1] Hyperparameter-wise mean relative deltas (fixed others to defaults)")
    for hp, tbl in C1.items():
        _print_df(tbl, title=f"param={hp}")

    print("\n[C2] Hyperparameter-wise mean relative deltas (averaged over others)")
    for hp, tbl in C2.items():
        _print_df(tbl, title=f"param={hp}")

if __name__ == "__main__":
    main(Path("./experiments/2_grid/results/"))
