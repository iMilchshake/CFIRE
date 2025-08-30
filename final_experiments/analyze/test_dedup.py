from pathlib import Path
from typing import Dict
import pandas as pd
from tabulate import tabulate
import numpy as np

from final_experiments.analyze.utils import (
    load_csv_files,
    merge_local_explainers,
    filter_all_params_to_default,
)

results_dir = Path("./experiments/2_grid/results/")

# NEW: where CSVs will be written
export_dir = Path("./experiments/2_grid/analysis/dedup/")
export_dir.mkdir(parents=True, exist_ok=True)

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
    mean_rounded = round(mean_change, 2)
    return f"{(mean_rounded / mean_full) * 100.0:+.2f}%"


def stat_with_minmax_and_rel(df: pd.DataFrame, metric: str) -> str:
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

    all_rows = []
    all_time_diffs = []

    # single collector for raw rows (with boolean 'drop' column, includes locals + Merged)
    raw_rows = []

    for dataset, _ in metrics.items():
        if dataset not in metrics_dedup:
            print(f"\n[skip] {dataset}: not in dedup")
            continue

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
            df_full[JOIN_COLS + metric_cols + ["t_compose_rules"]],
            df_dedup[JOIN_COLS + metric_cols + ["t_dedup"]],
            on=JOIN_COLS,
            suffixes=("_full", "_dedup"),
            how="inner",
        )

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

        tt = df_merged[["t_dedup", "t_compose_rules"]].apply(pd.to_numeric, errors="coerce").dropna()
        diffs = tt["t_dedup"] - tt["t_compose_rules"]
        print(f"t_dedup (s): {tt['t_dedup'].mean():.2f}±{tt['t_dedup'].std():.2f} "
              f"(min={tt['t_dedup'].min():.2f}, max={tt['t_dedup'].max():.2f})")
        print(f"Δ time abs (dedup - compose_rules): {diffs.mean():.2f}±{diffs.std():.2f} "
              f"(min={diffs.min():.2f}, max={diffs.max():.2f})")

        all_time_diffs.append(diffs)
        all_rows.append(df_merged)

        # === NEW: percent changes (raw), row-wise ===
        pct_cols = []
        for base in ["val_f1_weighted", "test_f1_weighted", "rule_size", "literal_count"]:
            if f"{base}_full" in df_merged.columns and f"{base}_change" in df_merged.columns:
                col = f"pct_{base}"
                df_merged[col] = (df_merged[f"{base}_change"] / df_merged[f"{base}_full"]) * 100.0
                pct_cols.append(col)
        if pct_cols:
            df_merged[pct_cols] = df_merged[pct_cols].replace([np.inf, -np.inf], np.nan)

            # locals no-drop
            df_nodrop = df_merged[["expl_method", *pct_cols]].copy()
            df_nodrop.insert(0, "dataset", dataset)
            df_nodrop.insert(1, "drop", False)
            raw_rows.append(df_nodrop)

            # merged no-drop
            mask_merged = df_merged.index.isin(merged_idx)
            df_m_nodrop = df_merged.loc[mask_merged, pct_cols].copy()
            if not df_m_nodrop.empty:
                df_m_nodrop.insert(0, "dataset", dataset)
                df_m_nodrop.insert(1, "drop", False)
                df_m_nodrop.insert(2, "expl_method", "Merged")
                raw_rows.append(df_m_nodrop)

            # locals drop
            rs = df_merged.get("rule_size_change")
            lc = df_merged.get("literal_count_change")
            if rs is not None and lc is not None:
                mask_keep = ~(rs.fillna(0).eq(0) & lc.fillna(0).eq(0))
            else:
                mask_keep = pd.Series(True, index=df_merged.index)
            kept_drop = df_merged.loc[mask_keep]
            if not kept_drop.empty:
                df_drop = kept_drop[["expl_method", *pct_cols]].copy()
                df_drop.insert(0, "dataset", dataset)
                df_drop.insert(1, "drop", True)
                raw_rows.append(df_drop)

            # merged drop
            kept_m_drop = df_merged.loc[mask_merged & mask_keep, pct_cols].copy()
            if not kept_m_drop.empty:
                kept_m_drop.insert(0, "dataset", dataset)
                kept_m_drop.insert(1, "drop", True)
                kept_m_drop.insert(2, "expl_method", "Merged")
                raw_rows.append(kept_m_drop)

    # === write exports ===
    def write_exports(df_list: list[pd.DataFrame]) -> None:
        target_cols = ["dataset", "drop", "expl_method",
                       "pct_val_f1_weighted", "pct_test_f1_weighted",
                       "pct_rule_size", "pct_literal_count"]

        df_all_raw = pd.concat(df_list, axis=0, ignore_index=True) if df_list else pd.DataFrame(columns=target_cols)
        cols = [c for c in target_cols if c in df_all_raw.columns]
        df_all_raw = df_all_raw[cols]

        df_all_raw.to_csv(export_dir / "raw_pct.csv", index=False)

        if not df_all_raw.empty:
            by_dataset = df_all_raw.groupby(["drop", "dataset"], as_index=False).mean(numeric_only=True)
            by_expl = df_all_raw.groupby(["drop", "expl_method"], as_index=False).mean(numeric_only=True)
        else:
            by_dataset = pd.DataFrame(columns=["drop", "dataset"])
            by_expl = pd.DataFrame(columns=["drop", "expl_method"])

        by_dataset.to_csv(export_dir / "agg_by_dataset.csv", index=False)
        by_expl.to_csv(export_dir / "agg_by_explainer.csv", index=False)

    write_exports(raw_rows)

    # === final print: aggregate by explainer (with drop flag) ===
    # === final print: aggregate by explainer (mean±std), split by drop ===
    df_all_raw = pd.concat(raw_rows, axis=0, ignore_index=True)

    pct_cols = [
        "pct_val_f1_weighted", "pct_test_f1_weighted",
        "pct_rule_size", "pct_literal_count"
    ]

    means = df_all_raw.groupby(["drop", "expl_method"])[pct_cols].mean()
    stds  = df_all_raw.groupby(["drop", "expl_method"])[pct_cols].std()

    fmt = (means.round(2).astype(str) + "±" + stds.round(2).astype(str)).reset_index()
    fmt = fmt.sort_values(["drop", "expl_method"])

    counts = (
        df_all_raw.groupby(["expl_method", "drop"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[False, True], fill_value=0)
    )
    total = counts[False]
    kept = counts[True]
    unchanged_ratio = ((total - kept) / total.replace(0, pd.NA)).astype(float).round(3)
    ratio_df = unchanged_ratio.rename("unchanged_ratio").reset_index()

    fmt = fmt.merge(ratio_df, on="expl_method", how="left").sort_values(
        ["drop", "expl_method"]
    )

    print("\n=== Aggregate percent changes by explainer (split by drop) ===")
    print(tabulate(
        fmt[["drop", "expl_method", *pct_cols, "unchanged_ratio"]],
        headers="keys", tablefmt="github", showindex=False
    ))

    diffs_all = pd.concat(all_time_diffs, axis=0)
    print(f"\nΔ_t_abs (dedup - compose_rules) global: "
          f"{diffs_all.mean():.4f}±{diffs_all.std():.4f} "
          f"(min={diffs_all.min():.4f}, max={diffs_all.max():.4f})")


if __name__ == "__main__":
    main()
