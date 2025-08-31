# grid_pruning_aggregate_pruning_effects.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from final_experiments.analyze.utils import load_csv_files  # reuse your loader

# ---- config ----
results_dir = Path("./experiments/2_grid/results/")
out_path = Path("./experiments/2_grid/results_pruning_aggregate.txt")

# Only these are treated as identifiers (adapt if your repo uses more):
ID_COLUMNS_DEFAULT = [
    "model_idx",
    "cfire_config_idx",
    "cfire_seed",
    "expl_method",
    "freq_threshold",
    "bin_config",
    "max_dt_depth",
]

PERF_METRICS: List[str] = ["val_acc", "val_f1_weighted", "test_f1_weighted", "test_acc"]
COMPLEXITY_METRICS: List[str] = ["rule_size", "literal_count"]
ALL_METRICS = PERF_METRICS + COMPLEXITY_METRICS


def _parse_mean(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        for sep in ("±", "+/-", "+–", "+-"):
            if sep in s:
                s = s.split(sep)[0].strip()
                break
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    return series.map(_parse_mean).astype(float)


def _summarize(series: pd.Series) -> Tuple[float, float, int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan"), float("nan"), 0
    mean = s.mean()
    var = s.var(ddof=1) if len(s) > 1 else 0.0
    return float(mean), float(var), int(len(s))


def _available_id_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in ID_COLUMNS_DEFAULT if c in df.columns]


def _three_way_merge(
        df_b: pd.DataFrame, df_best: pd.DataFrame, df_safe: pd.DataFrame, id_cols: List[str], metrics: List[str]
) -> pd.DataFrame:
    """Create a single table with *_before, *_after_best, *_after_safe without losing columns."""
    m_b = [m for m in metrics if m in df_b.columns]
    m_best = [m for m in metrics if m in df_best.columns]
    m_safe = [m for m in metrics if m in df_safe.columns]

    left = df_b[id_cols + m_b].copy().rename(columns={m: f"{m}_before" for m in m_b})
    right_best = df_best[id_cols + m_best].copy().rename(columns={m: f"{m}_after_best" for m in m_best})
    right_safe = df_safe[id_cols + m_safe].copy().rename(columns={m: f"{m}_after_safe" for m in m_safe})

    merged = pd.merge(left, right_best, on=id_cols, how="inner", validate="one_to_one")
    merged = pd.merge(merged, right_safe, on=id_cols, how="inner", validate="one_to_one")
    return merged


def main() -> None:
    base: Dict[str, pd.DataFrame] = load_csv_files(results_dir, csv_file_name="metrics.csv")
    best: Dict[str, pd.DataFrame] = load_csv_files(results_dir, csv_file_name="metrics_best_prune.csv")
    safe: Dict[str, pd.DataFrame] = load_csv_files(results_dir, csv_file_name="metrics_safe_prune.csv")

    all_rows: List[pd.DataFrame] = []
    datasets_used: List[str] = []
    total_candidates = 0

    for ds in sorted(base.keys()):
        if ds not in best or ds not in safe:
            continue

        df_b, df_best, df_safe = base[ds].copy(), best[ds].copy(), safe[ds].copy()
        id_cols = list(dict.fromkeys(_available_id_cols(df_b) + _available_id_cols(df_best) + _available_id_cols(df_safe)))
        if not id_cols:
            continue

        total_candidates += len(df_b)

        merged = _three_way_merge(df_b, df_best, df_safe, id_cols, ALL_METRICS)

        # Performance: decrease = before - after (absolute percentage points)
        for m in PERF_METRICS:
            b = _to_numeric(merged.get(f"{m}_before"))
            ab = _to_numeric(merged.get(f"{m}_after_best"))
            as_ = _to_numeric(merged.get(f"{m}_after_safe"))
            merged[f"decrease_best_{m}"] = b - ab
            merged[f"decrease_safe_{m}"] = b - as_

        # Complexity: change = after - before (negative => smaller after pruning)
        for m in COMPLEXITY_METRICS:
            b = _to_numeric(merged.get(f"{m}_before"))
            ab = _to_numeric(merged.get(f"{m}_after_best"))
            as_ = _to_numeric(merged.get(f"{m}_after_safe"))
            merged[f"change_best_{m}"] = ab - b
            merged[f"change_safe_{m}"] = as_ - b

        merged["__dataset__"] = ds
        all_rows.append(merged)
        datasets_used.append(ds)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not all_rows:
        out_path.write_text("No datasets with complete (before/best/safe) CSVs found.\n", encoding="utf-8")
        print(f"Wrote: {out_path.resolve()}")
        return

    big = pd.concat(all_rows, axis=0, ignore_index=True)

    # Build summary text
    lines: List[str] = []
    lines.append("========= Pruning Effect: Aggregated Across All Datasets =========")
    lines.append(f"Datasets included: {len(sorted(set(datasets_used)))} -> {', '.join(sorted(set(datasets_used)))}")
    lines.append(f"Matched configurations (rows): {len(big)} out of {total_candidates} base rows\n")
    lines.append("Performance decrease = (before - after) in absolute percentage points (e.g., 0.90→0.85 ⇒ 0.05).")
    lines.append("Complexity change   = (after - before); negative means pruning reduced size.")
    lines.append("Variance is the unbiased sample variance (ddof=1).\n")

    # Performance table
    perf_header = ["metric", "mean_decrease_best", "var_decrease_best", "N_best",
                   "mean_decrease_safe", "var_decrease_safe", "N_safe"]
    perf_rows = []
    for m in PERF_METRICS:
        mean_b, var_b, n_b = _summarize(big[f"decrease_best_{m}"])
        mean_s, var_s, n_s = _summarize(big[f"decrease_safe_{m}"])
        perf_rows.append([m, mean_b, var_b, n_b, mean_s, var_s, n_s])
    perf_df = pd.DataFrame(perf_rows, columns=perf_header)

    # Complexity table
    comp_header = ["metric", "mean_change_best", "var_change_best", "N_best",
                   "mean_change_safe", "var_change_safe", "N_safe"]
    comp_rows = []
    for m in COMPLEXITY_METRICS:
        mean_b, var_b, n_b = _summarize(big[f"change_best_{m}"])
        mean_s, var_s, n_s = _summarize(big[f"change_safe_{m}"])
        comp_rows.append([m, mean_b, var_b, n_b, mean_s, var_s, n_s])
    comp_df = pd.DataFrame(comp_rows, columns=comp_header)

    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.6f}".format):
        lines.append("---- Performance (decrease: before - after) ----")
        lines.append(perf_df.to_string(index=False))
        lines.append("")
        lines.append("---- Complexity (change: after - before) ----")
        lines.append(comp_df.to_string(index=False))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote aggregated pruning summary to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
