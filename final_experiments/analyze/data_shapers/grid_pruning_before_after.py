from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Tuple, List

import pandas as pd

from final_experiments.analyze.utils import (
    ALL_PARAMS,
    filter_other_params_to_default,
    merge_local_explainers,
    get_metric_table,
    load_csv_files,
)

results_dir = Path("./experiments/2_grid/results/")
out_path = Path("./experiments/2_grid/results_grid2_pruning.txt")

METRIC_SELECTION: List[str] = [
    "val_acc",
    "val_f1_weighted",
    "test_f1_weighted",
    "test_acc",
    "rule_size",
    "rule_count",
    "literal_count",
    "unique_literal_count",
    "max_iou",
    "mean_iou",
    "t_rule_candidates",
    "t_compose_rules",
    "missing_class_rules",
    "missing_pred_val",
    "missing_pred_test",
    "mean_coverage_ratio",
    "mean_single_coverage_ratio",
    "mean_nodes_per_sample",
    "mean_duplicate_nodes_ratio",
    "total_frequent_node_count",
    "attr_mean_absolute_attribution",
    "attr_attribution_variance",
    "attr_sparsity",
    "attr_class_separation",
    "bin_mean_active_features_per_sample",
    "bin_mean_active_features_ratio",
    "bin_mean_feature_activation_ratio",
    "bin_features_inactive_ratio",
    "bin_mean_feature_class_specificity",
    "bin_mean_within_class_jaccard",
    "bin_mean_across_class_jaccard",
    "bin_class_separation_score",
    "bin_all_features_active_ratio",
    "bin_all_features_inactive_ratio",
]
EXPLAINERS = ["IG", "lime", "kernelshap", "merged"]


def _parse_mean(value) -> Optional[float]:
    """Extract the mean from strings like '0.76±0.03' or return float if numeric."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        for sep in ("±", "+/-", "+–", "+-"):
            if sep in s:
                s = s.split(sep)[0].strip()
                break
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _with_subcolumns(
        base_table: pd.DataFrame,
        best_table: pd.DataFrame,
        safe_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every parameter value column, create a 5-subcolumn block:
      before | after_best | after_safe | Δbest | Δsafe
    Δ columns are differences of MEANS only (signed, +/−).
    """
    metrics_index = base_table.index
    all_cols = list(dict.fromkeys(list(base_table.columns) + list(best_table.columns) + list(safe_table.columns)))

    base_table = base_table.reindex(columns=all_cols)
    best_table = best_table.reindex(columns=all_cols)
    safe_table = safe_table.reindex(columns=all_cols)

    blocks = []
    for col in all_cols:
        before = base_table[col]
        after_b = best_table[col]
        after_s = safe_table[col]

        d_best, d_safe = [], []
        for b, ab, as_ in zip(before.tolist(), after_b.tolist(), after_s.tolist()):
            mb = _parse_mean(b)
            mab = _parse_mean(ab)
            mas = _parse_mean(as_)
            db = (mab - mb) if (mb is not None and mab is not None) else None
            ds = (mas - mb) if (mb is not None and mas is not None) else None
            d_best.append("" if db is None else f"{db:+.3f}")
            d_safe.append("" if ds is None else f"{ds:+.3f}")

        block = pd.DataFrame(
            {
                (str(col), "before"): before,
                (str(col), "after_best"): after_b,
                (str(col), "after_safe"): after_s,
                (str(col), "Δbest"): d_best,
                (str(col), "Δsafe"): d_safe,
            },
            index=metrics_index,
        )
        blocks.append(block)

    out = pd.concat(blocks, axis=1)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def _subset_for_expl(
        df_base: pd.DataFrame, df_best: pd.DataFrame, df_safe: pd.DataFrame, expl: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if expl == "merged":
        return df_base, df_best, df_safe
    return (
        df_base[df_base["expl_method"] == expl],
        df_best[df_best["expl_method"] == expl],
        df_safe[df_safe["expl_method"] == expl],
    )


def _analyze_one_dataset(name: str,
                         base_df: pd.DataFrame,
                         best_df: pd.DataFrame,
                         safe_df: pd.DataFrame) -> str:
    """Builds the text section for a single dataset"""
    lines: List[str] = []
    lines.append(f"\n\n{'='*15} ANALYSIS FOR DATASET: '{name}' {'='*15}")

    for param in ALL_PARAMS:
        df_base = filter_other_params_to_default(base_df, param)
        df_best = filter_other_params_to_default(best_df, param)
        df_safe = filter_other_params_to_default(safe_df, param)

        df_cfire_base = merge_local_explainers(df_base)
        df_cfire_best = merge_local_explainers(df_best)
        df_cfire_safe = merge_local_explainers(df_safe)

        for expl in EXPLAINERS:
            if expl == "merged":
                sub_b, sub_best, sub_safe = df_cfire_base, df_cfire_best, df_cfire_safe
            else:
                sub_b, sub_best, sub_safe = _subset_for_expl(df_base, df_best, df_safe, expl)

            tbl_b = get_metric_table(sub_b, param, METRIC_SELECTION)
            tbl_best = get_metric_table(sub_best, param, METRIC_SELECTION)
            tbl_safe = get_metric_table(sub_safe, param, METRIC_SELECTION)

            combined = _with_subcolumns(tbl_b, tbl_best, tbl_safe)

            lines.append(f"\n ======= [{name}] param={param} | expl={expl} ======")
            try:
                lines.append(combined.to_string())
            except Exception:
                # fallback — shouldn't happen, but keeps the file writing robust
                lines.append(str(combined))

    return "\n".join(lines)


def main() -> None:
    # Each dataset dir contains: metrics.csv, metrics_best_prune.csv, metrics_safe_prune.csv
    base: Dict[str, pd.DataFrame] = load_csv_files(results_dir, csv_file_name="metrics.csv")
    best: Dict[str, pd.DataFrame] = load_csv_files(results_dir, csv_file_name="metrics_best_prune.csv")
    safe: Dict[str, pd.DataFrame] = load_csv_files(results_dir, csv_file_name="metrics_safe_prune.csv")

    blocks: List[str] = []
    for dataset in sorted(base.keys()):
        if dataset not in best or dataset not in safe:
            print(f"[WARN] Skipping dataset '{dataset}' — best/safe prune files missing.")
            continue
        blocks.append(_analyze_one_dataset(dataset, base[dataset], best[dataset], safe[dataset]))

    text = "\n".join(blocks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote pruning-aware grid to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
