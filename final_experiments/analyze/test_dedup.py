from pathlib import Path
from typing import Dict
import pandas as pd
from tabulate import tabulate

from final_experiments.analyze.utils import (
    load_csv_files,
    merge_local_explainers,
    filter_all_params_to_default,
)

results_dir = Path("./experiments/2_grid/results/")

JOIN_COLS = [
    "model_idx", "cfire_config_idx", "cfire_seed", "expl_method",
    "freq_threshold", "bin_config", "max_dt_depth"
]

FOCUS_METRICS = [
    "val_f1_weighted",
    "test_f1_weighted",
    "rule_size",
    "literal_count",
    "unique_literal_count",
]

# toggle dropping of rows with zero change in rule_size & literal_count
DROP_ZERO_CHANGE_ROWS = True


def fmt_rel_percent(mean_change: float, mean_full: float) -> str:
    if pd.isna(mean_change) or pd.isna(mean_full):
        return "—%"
    if mean_full == 0.0:
        if mean_change > 0:
            return "+inf%"
        if mean_change < 0:
            return "-inf%"
        return "0.00%"
    return f"{(mean_change / mean_full) * 100.0:+.2f}%"


def stat_with_minmax_and_rel(df: pd.DataFrame, metric: str) -> str:
    """
    df slice must contain f"{metric}_change" and f"{metric}_full".
    Output: mean±std [rel%] (min=..., max=...)
    """
    change_col = f"{metric}_change"
    base_col = f"{metric}_full"
    vals = pd.to_numeric(df.get(change_col, pd.Series(dtype=float)), errors="coerce").dropna()
    if vals.empty:
        return "—"
    mean = vals.mean()
    std = vals.std()
    vmin = vals.min()
    vmax = vals.max()
    base_vals = pd.to_numeric(df.get(base_col, pd.Series(dtype=float)), errors="coerce").dropna()
    base_mean = base_vals.mean() if not base_vals.empty else float("nan")
    rel = fmt_rel_percent(mean, base_mean)
    return f"{mean:.2f}±{std:.2f} [{rel}] (min={vmin:.2f}, max={vmax:.2f})"


def main() -> None:
    metrics: Dict[str, pd.DataFrame] = load_csv_files(results_dir, csv_file_name="metrics.csv")
    metrics_dedup: Dict[str, pd.DataFrame] = load_csv_files(results_dir, csv_file_name="metrics_dedup.csv")

    for dataset, _ in metrics.items():
        if dataset not in metrics_dedup:
            print(f"\n[skip] {dataset}: not in dedup")
            continue

        # restrict to default hyperparameters
        df_full = filter_all_params_to_default(metrics[dataset])
        df_dedup = filter_all_params_to_default(metrics_dedup[dataset])

        if df_full.empty or df_dedup.empty:
            print(f"\n[skip] {dataset}: no rows after filtering to defaults")
            continue

        metric_cols = [m for m in FOCUS_METRICS if m in df_full.columns and m in df_dedup.columns]
        if "val_acc" in df_full.columns and "val_acc" in df_dedup.columns:
            metric_cols = ["val_acc"] + metric_cols
        if not metric_cols:
            print(f"\n[skip] {dataset}: no overlapping focus metrics")
            continue

        df_merged = pd.merge(
            df_full[JOIN_COLS + metric_cols],
            df_dedup[JOIN_COLS + metric_cols],
            on=JOIN_COLS,
            suffixes=("_full", "_dedup"),
            how="inner"
        )
        if df_merged.empty:
            print(f"\n[skip] {dataset}: no matching rows after merge")
            continue

        for m in metric_cols:
            if m == "val_acc":
                continue
            df_merged[f"{m}_change"] = df_merged[f"{m}_dedup"] - df_merged[f"{m}_full"]

        def split_keep_drop(df_view: pd.DataFrame):
            if df_view.empty:
                return df_view, 0, 0
            total = len(df_view)
            if not DROP_ZERO_CHANGE_ROWS:
                return df_view, 0, total
            rs = df_view.get("rule_size_change")
            lc = df_view.get("literal_count_change")
            if rs is None or lc is None:
                return df_view, 0, total
            mask_keep = ~(rs.fillna(0).eq(0) & lc.fillna(0).eq(0))
            return df_view.loc[mask_keep], (~mask_keep).sum(), total

        df_for_merge = df_merged[[*JOIN_COLS, *[f"{m}_dedup" for m in metric_cols]]].rename(
            columns={f"{m}_dedup": m for m in metric_cols}
        )

        results: Dict[str, tuple[dict, int, int]] = {}

        # merged
        merged_idx = merge_local_explainers(df_for_merge).index
        df_merged_sel = df_merged.loc[merged_idx]
        kept, dropped, total = split_keep_drop(df_merged_sel)
        if not kept.empty:
            stats = {
                m: stat_with_minmax_and_rel(kept, m)
                for m in FOCUS_METRICS
                if f"{m}_change" in kept and f"{m}_full" in kept
            }
            results["Merged"] = (stats, dropped, total)

        # locals
        for m in df_merged["expl_method"].unique():
            df_view = df_merged[df_merged["expl_method"] == m]
            kept, dropped, total = split_keep_drop(df_view)
            if kept.empty:
                continue
            stats = {
                met: stat_with_minmax_and_rel(kept, met)
                for met in FOCUS_METRICS
                if f"{met}_change" in kept and f"{met}_full" in kept
            }
            results[m] = (stats, dropped, total)

        print(f"\n=== Dataset: {dataset} ===")
        if results:
            df_out = pd.DataFrame({k: v[0] for k, v in results.items()}).T
            df_out = df_out[FOCUS_METRICS]
            print(tabulate(df_out.T, headers="keys", tablefmt="github", showindex=True))
            for expl, (_, dropped, total) in results.items():
                if total > 0 and DROP_ZERO_CHANGE_ROWS:
                    print(f"{expl}: dropped {dropped}/{total} rows with no rule_size/literal_count change")
        else:
            print("no explainers with rule_size/literal_count change")

        if "t_dedup" in df_dedup.columns:
            t_vals = pd.to_numeric(df_dedup["t_dedup"], errors="coerce").dropna()
            if t_vals.empty:
                print("t_dedup (s): —")
            else:
                print(f"t_dedup (s): {t_vals.mean():.2f}±{t_vals.std():.2f} (min={t_vals.min():.2f}, max={t_vals.max():.2f})")


if __name__ == "__main__":
    main()
